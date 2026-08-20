"""Evaluate pretrained conditional generators on the held-out test split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from calc_inception import load_patched_inception_v3
from checkpoint_utils import load_checkpoint_args
from diffusion_model import ConditionalUNet
from diffusion_process import DDPMProcess, EDMProcess
from fid import calc_fid
from model import Generator as StyleGANGenerator
from sysmex_task1_dataset import SysmexTask1Dataset
from train_dcgan import Generator as DCGANGenerator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_value(config, cli_value, name, default=None):
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def split_dataset(dataset, split_name):
    indices = [
        index for index, row in enumerate(dataset.rows)
        if str(row.get("split", "")).strip().lower() == split_name
    ]
    if not indices:
        raise ValueError(f"No samples found for split {split_name!r}")
    return Subset(dataset, indices)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_generator(model_name, checkpoint, config, cli, device):
    if model_name == "stylegan":
        generator = StyleGANGenerator(
            size=int(config_value(config, cli.size, "size", 128)),
            style_dim=int(config_value(config, cli.latent, "latent", 512)),
            n_mlp=int(config_value(config, cli.n_mlp, "n_mlp", 8)),
            channel_multiplier=int(config_value(config, cli.channel_multiplier, "channel_multiplier", 2)),
            out_channels=1,
            profile_encoder=config_value(config, cli.profile_encoder, "profile_encoder", "cnn"),
        ).to(device)
        state = checkpoint["g_ema"]
    elif model_name == "dcgan":
        generator = DCGANGenerator(
            latent_dim=int(config_value(config, cli.latent, "latent", 128)),
            base_channels=int(config_value(config, cli.base_channels, "base_channels", 256)),
            profile_encoder=config_value(config, cli.profile_encoder, "profile_encoder", "mlp"),
        ).to(device)
        state = checkpoint.get("generator_ema") or checkpoint["generator"]
    elif model_name == "diffusion":
        generator = ConditionalUNet(
            profile_encoder=config_value(config, cli.profile_encoder, "profile_encoder", "cnn"),
            backbone=config_value(config, cli.backbone, "backbone", "compact"),
            base_channels=config_value(config, cli.base_channels, "base_channels", None),
            dropout=float(config_value(config, cli.dropout, "dropout", 0.0)),
        ).to(device)
        state = checkpoint["ema"]
    else:
        raise ValueError(f"Unknown model {model_name!r}")

    generator.load_state_dict(state)
    generator.eval()
    return generator


def make_diffusion_process(config, cli, device):
    objective = config_value(config, cli.objective, "objective", "ddpm")
    sampler = config_value(config, cli.sampler, "sampler", None)
    steps = config_value(config, cli.sample_steps, "sample_steps", None)
    if objective == "ddpm":
        process = DDPMProcess(device=device)
        sampler = "ddim" if sampler in (None, "auto") else sampler
        steps = int(steps or (50 if sampler == "ddim" else process.steps))
    elif objective == "edm":
        process = EDMProcess()
        sampler = "heun" if sampler in (None, "auto") else sampler
        steps = int(steps or 40)
    else:
        raise ValueError(f"Unknown diffusion objective {objective!r}")
    return process, objective, sampler, steps


@torch.inference_mode()
def generate(generator, model_name, profiles, config, cli, device, noise=None):
    shape = (profiles.shape[0], 1, 128, 128)
    if model_name == "stylegan":
        latent = int(config_value(config, cli.latent, "latent", 512))
        latent_noise = torch.randn(profiles.shape[0], latent, device=device)
        return generator([latent_noise], profile=profiles)[0]
    if model_name == "dcgan":
        latent = int(config_value(config, cli.latent, "latent", 128))
        latent_noise = torch.randn(profiles.shape[0], latent, device=device)
        return generator(latent_noise, profiles)
    process, _, sampler, steps = make_diffusion_process(config, cli, device)
    return process.sample(
        generator, profiles, shape, sampler=sampler, sampling_steps=steps, noise=noise
    )


def as_zero_one(images):
    return images.detach().float().clamp(-1, 1).add(1).div(2)


def inception_input(images):
    return images if images.shape[1] == 3 else images.repeat(1, 3, 1, 1)


def calculate_fid(real_features, fake_features):
    real = torch.cat(real_features).numpy()
    fake = torch.cat(fake_features).numpy()
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("FID requires at least two real and generated samples")
    return float(calc_fid(
        fake.mean(axis=0), np.cov(fake, rowvar=False),
        real.mean(axis=0), np.cov(real, rowvar=False),
    ))


def paired_metrics(real_images, fake_images, lpips_model=None, lpips_batch=16):
    real = real_images.cpu().numpy()[:, 0]
    fake = fake_images.cpu().numpy()[:, 0]
    psnr_values, ssim_values = [], []
    for target, prediction in zip(real, fake):
        mse = np.mean((target - prediction) ** 2)
        psnr_values.append(float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)))
        ssim_values.append(float(structural_similarity(target, prediction, data_range=1.0)))

    result = {
        "psnr_mean": float(np.mean(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
    }
    if lpips_model is not None:
        values = []
        for start in range(0, len(real_images), lpips_batch):
            real_rgb = real_images[start:start + lpips_batch].repeat(1, 3, 1, 1) * 2 - 1
            fake_rgb = fake_images[start:start + lpips_batch].repeat(1, 3, 1, 1) * 2 - 1
            values.append(lpips_model(real_rgb, fake_rgb).detach().flatten().cpu())
        result["lpips_mean"] = float(torch.cat(values).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("stylegan", "dcgan", "diffusion"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datasplit", required=True)
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--split_column", default=None)
    parser.add_argument("--brightfield_postfix", default=None)
    parser.add_argument("--profile_prefix", default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latent", type=int, default=None)
    parser.add_argument("--n_mlp", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--channel_multiplier", type=int, default=None)
    parser.add_argument("--profile_encoder", choices=("cnn", "mlp"), default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--backbone", choices=("compact", "adm"), default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--objective", choices=("ddpm", "edm"), default=None)
    parser.add_argument("--sampler", choices=("auto", "ddpm", "ddim", "euler", "heun"), default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--lpips_batch", type=int, default=16)
    cli = parser.parse_args()

    seed_everything(cli.seed)
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(cli.ckpt)
    overrides = vars(cli).copy()
    config = load_checkpoint_args(cli.ckpt, overrides=overrides)

    dataset_kwargs = {
        "brightfield_postfix": config_value(config, cli.brightfield_postfix, "brightfield_postfix", "_brightfield_crop_masked_normalized_avebg_pad128.png"),
    }
    if cli.profile_prefix is not None or "profile_prefix" in config:
        dataset_kwargs["profile_prefix"] = config_value(config, cli.profile_prefix, "profile_prefix", "vertical")
    dataset = SysmexTask1Dataset(cli.datasplit, cli.preprocessed_root, **dataset_kwargs)
    split_column = config_value(config, cli.split_column, "split_column", "split")
    indices = [i for i, row in enumerate(dataset.rows) if str(row.get(split_column, "")).strip().lower() == cli.split.lower()]
    if not indices:
        raise ValueError(f"No samples found for split {cli.split!r} using column {split_column!r}")
    if cli.num_samples is not None:
        indices = indices[:cli.num_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=cli.batch, shuffle=False, num_workers=0)

    generator = build_generator(cli.model, checkpoint, config, cli, device)
    inception = load_patched_inception_v3().to(device).eval()
    real_features, fake_features = [], []
    real_images, fake_images = [], []
    lpips_model = None
    if not cli.no_lpips:
        from lpips import PerceptualLoss
        lpips_model = PerceptualLoss(net="alex", use_gpu=device.type == "cuda", gpu_ids=[device.index or 0])

    for batch in tqdm(loader, desc="evaluating"):
        real = batch["image"].to(device)
        profiles = batch["profile"].to(device)
        fake = generate(generator, cli.model, profiles, config, cli, device)
        real01, fake01 = as_zero_one(real), as_zero_one(fake)
        real_features.append(inception(inception_input(real01))[0].flatten(1).cpu())
        fake_features.append(inception(inception_input(fake01))[0].flatten(1).cpu())
        real_images.append(real01.cpu())
        fake_images.append(fake01.cpu())

    real_images = torch.cat(real_images)
    fake_images = torch.cat(fake_images)
    result = {
        "model": cli.model,
        "checkpoint": str(Path(cli.ckpt).resolve()),
        "split": cli.split,
        "num_samples": len(real_images),
        "seed": cli.seed,
        "fid": calculate_fid(real_features, fake_features),
        **paired_metrics(real_images, fake_images, lpips_model, cli.lpips_batch),
    }
    if cli.model == "diffusion":
        _, objective, sampler, steps = make_diffusion_process(config, cli, device)
        result.update({"objective": objective, "sampler": sampler, "sample_steps": steps})
    print(json.dumps(result, indent=2))
    if cli.output:
        Path(cli.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
