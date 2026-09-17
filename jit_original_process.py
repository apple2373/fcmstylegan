"""JiT paper-style training objective and ODE sampler."""

import torch
import torch.nn.functional as F


# Relationship to JiT/denoiser.py
#
# This class intentionally mirrors the algorithmic parts of the upstream JiT
# Denoiser: logit-normal timestep sampling via P_mean/P_std, interpolation
# from Gaussian noise to data, clean-image prediction, conversion of that
# prediction to velocity, velocity MSE training, and Euler/Heun ODE sampling
# with interval-limited classifier-free guidance.
#
# It is not a direct call to the upstream Denoiser because that implementation
# assumes RGB images, integer ImageNet class labels, and owns the complete
# training wrapper. This project instead uses one-channel images and continuous
# (3, 128) profiles. Profile projection, learned null-profile dropout, model
# construction, EMA, FID, checkpointing, and logging therefore live in the
# project adapter/trainer. The diffusion path below is JiT-method faithful,
# while the conditioning and task-specific plumbing are intentionally adapted.

class JiTOriginalProcess:
    def __init__(self, P_mean=-0.8, P_std=0.8, noise_scale=1.0, t_eps=5e-2,
                 cfg=1.0, interval_min=0.0, interval_max=1.0,
                 t_distribution="logit_normal"):
        if t_distribution not in {"logit_normal", "uniform"}:
            raise ValueError(
                "t_distribution must be 'logit_normal' or 'uniform'"
            )
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.cfg = cfg
        self.cfg_interval = (interval_min, interval_max)
        self.t_distribution = t_distribution

    def sample_t(self, batch, device):
        if self.t_distribution == "uniform":
            return torch.rand(batch, device=device)
        logits = torch.randn(batch, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(logits)

    def training_loss(self, model, image, profile):
        t = self.sample_t(image.shape[0], image.device)
        t_view = t[:, None, None, None]
        noise = torch.randn_like(image) * self.noise_scale
        z = t_view * image + (1.0 - t_view) * noise
        target = (image - z) / (1.0 - t_view).clamp_min(self.t_eps)
        x_pred = model(z, t, profile)
        prediction = (x_pred - z) / (1.0 - t_view).clamp_min(self.t_eps)
        return (target - prediction).square().mean(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def _velocity(self, model, z, t, profile):
        x_cond = model(z, t.flatten(), profile)
        x_uncond = model(z, t.flatten(), profile, unconditional=True)
        denominator = (1.0 - t).clamp_min(self.t_eps)
        v_cond = (x_cond - z) / denominator
        v_uncond = (x_uncond - z) / denominator
        low, high = self.cfg_interval
        mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(mask, self.cfg, torch.ones_like(t))
        return v_uncond + scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(self, model, profile, shape, sampler="heun", sampling_steps=50, noise=None):
        if sampler not in {"euler", "heun"}:
            raise ValueError("JiT original process supports 'euler' and 'heun'")
        z = (torch.randn(shape, device=profile.device) * self.noise_scale
             if noise is None else noise.clone())
        times = torch.linspace(0.0, 1.0, sampling_steps + 1, device=z.device)
        for step in range(sampling_steps):
            t = times[step].expand(shape[0]).view(-1, 1, 1, 1)
            t_next = times[step + 1]
            velocity = self._velocity(model, z, t, profile)
            if sampler == "heun" and step < sampling_steps - 1:
                z_euler = z + (t_next - times[step]) * velocity
                next_t = t_next.expand(shape[0]).view(-1, 1, 1, 1)
                next_velocity = self._velocity(model, z_euler, next_t, profile)
                z = z + 0.5 * (t_next - times[step]) * (velocity + next_velocity)
            else:
                z = z + (t_next - times[step]) * velocity
        return z
