"""DDPM and EDM objectives and samplers."""

import math

import torch
from torch.nn import functional as F


def _extract(values, timestep, shape):
    result = values.to(timestep.device)[timestep]
    return result.reshape(timestep.shape[0], *((1,) * (len(shape) - 1)))


class DDPMProcess:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, device=None):
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()

    def training_loss(self, model, image, profile):
        timestep = torch.randint(0, self.steps, (image.shape[0],), device=image.device)
        noise = torch.randn_like(image)
        noised = (
            _extract(self.sqrt_alpha_bars, timestep, image.shape) * image
            + _extract(self.sqrt_one_minus_alpha_bars, timestep, image.shape) * noise
        )
        prediction = model(noised, timestep.float() / max(self.steps - 1, 1), profile)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, model, profile, shape, sampler="ddim", sampling_steps=None, noise=None):
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError("DDPM supports 'ddpm' and 'ddim' samplers")
        batch = shape[0]
        image = torch.randn(shape, device=profile.device) if noise is None else noise.clone()
        sampling_steps = sampling_steps or self.steps
        times = torch.linspace(self.steps - 1, 0, sampling_steps, device=profile.device).long()

        for index, timestep in enumerate(times):
            t = torch.full((batch,), timestep, device=profile.device, dtype=torch.long)
            prediction = model(image, t.float() / max(self.steps - 1, 1), profile)
            alpha_bar = _extract(self.alpha_bars, t, image.shape)
            sqrt_alpha_bar = alpha_bar.sqrt()
            sqrt_one_minus = (1 - alpha_bar).sqrt()
            predicted_x0 = (image - sqrt_one_minus * prediction) / sqrt_alpha_bar.clamp_min(1e-5)
            predicted_x0 = predicted_x0.clamp(-1.5, 1.5)

            if index == len(times) - 1:
                image = predicted_x0
                continue

            next_timestep = torch.full(
                (batch,), times[index + 1], device=profile.device, dtype=torch.long
            )
            next_alpha_bar = _extract(self.alpha_bars, next_timestep, image.shape)
            if sampler == "ddpm":
                beta = _extract(self.betas, t, image.shape)
                alpha = _extract(self.alphas, t, image.shape)
                mean = (image - beta * prediction / sqrt_one_minus.clamp_min(1e-5)) / alpha.sqrt()
                variance = ((1 - next_alpha_bar) / (1 - alpha_bar) * beta).clamp_min(1e-20)
                image = mean + variance.sqrt() * torch.randn_like(image)
            else:
                direction = (1 - next_alpha_bar).sqrt() * prediction
                image = next_alpha_bar.sqrt() * predicted_x0 + direction

        return image


class EDMProcess:
    def __init__(
        self,
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        p_mean=-1.2,
        p_std=1.2,
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std

    def denoise(self, model, image, sigma, profile):
        sigma = sigma.reshape(-1, 1, 1, 1)
        sigma_data = self.sigma_data
        c_in = 1 / (sigma.square() + sigma_data**2).sqrt()
        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / (sigma.square() + sigma_data**2).sqrt()
        c_noise = sigma.log() / 4
        residual = model(c_in * image, c_noise.flatten(), profile)
        return c_skip * image + c_out * residual

    def training_loss(self, model, image, profile):
        sigma = (torch.randn(image.shape[0], device=image.device) * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(image) * sigma[:, None, None, None]
        denoised = self.denoise(model, image + noise, sigma, profile)
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        return (weight[:, None, None, None] * (denoised - image).square()).mean()

    @torch.no_grad()
    def sample(self, model, profile, shape, sampler="heun", sampling_steps=40, noise=None):
        if sampler not in {"euler", "heun"}:
            raise ValueError("EDM supports 'euler' and 'heun' samplers")
        batch = shape[0]
        image = torch.randn(shape, device=profile.device) if noise is None else noise.clone()
        ramp = torch.linspace(0, 1, sampling_steps, device=profile.device)
        min_power = self.sigma_min ** (1 / self.rho)
        max_power = self.sigma_max ** (1 / self.rho)
        sigmas = (max_power + ramp * (min_power - max_power)).pow(self.rho)
        image = image * sigmas[0]

        for index in range(len(sigmas) - 1):
            sigma = sigmas[index].expand(batch)
            next_sigma = sigmas[index + 1]
            denoised = self.denoise(model, image, sigma, profile)
            derivative = (image - denoised) / sigma[:, None, None, None]
            image_next = image + (next_sigma - sigma)[:, None, None, None] * derivative

            if sampler == "heun" and next_sigma > self.sigma_min:
                next_denoised = self.denoise(model, image_next, next_sigma.expand(batch), profile)
                next_derivative = (image_next - next_denoised) / next_sigma
                image = image + (next_sigma - sigma)[:, None, None, None] * (
                    derivative + next_derivative
                ) / 2
            else:
                image = image_next

        return image
