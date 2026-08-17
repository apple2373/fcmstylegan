import argparse
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import random
import shutil
import subprocess
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from calc_inception import load_patched_inception_v3
from fid import calc_fid
from reproducibility import seed_everything, seed_worker

from sysmex_task1_dataset import SysmexTask1Dataset
from diff_augment import AUGMENT_FNS, DiffAugment

IMAGE_SIZE = 128
PROFILE_DIM = 3 * 128


def maybe_compile(model, compile_mode):
    mode = str(compile_mode).lower()
    if mode in {"none", "off", "false", "0"}:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model, mode=mode)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update EMA parameters and copy BatchNorm buffers from the live model."""
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.mul_(decay).add_(model_parameters[name], alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


class ProfileEncoder(nn.Module):
    """Encode the three 128-bin FCM tracks for DCGAN conditioning."""

    def __init__(self, out_dim, mode="mlp"):
        super().__init__()
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(PROFILE_DIM, out_dim),
                nn.LeakyReLU(0.2, True),
            )
        elif mode == "cnn":
            self.net = nn.Sequential(
                nn.Conv1d(3, 64, 5, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(64, 128, 5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, 256, 5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, out_dim),
            )
        else:
            raise ValueError(f"Unknown profile encoder {mode!r}; choose 'mlp' or 'cnn'")

    def forward(self, profile):
        if profile.ndim != 3 or tuple(profile.shape[1:]) != (3, 128):
            raise ValueError(
                f"profile must have shape (batch, 3, 128), got {tuple(profile.shape)}"
            )
        return self.net(profile)


class Generator(nn.Module):
    def __init__(self, latent_dim=128, base_channels=512, profile_encoder="mlp"):
        super().__init__()
        if base_channels < 16 or base_channels % 16:
            raise ValueError("base_channels must be a multiple of 16 and at least 16")
        c4, c8, c16, c32, c64 = (
            base_channels, base_channels // 2, base_channels // 4,
            base_channels // 8, base_channels // 16,
        )
        self.profile = ProfileEncoder(256, profile_encoder)
        self.project = nn.Sequential(
            nn.Linear(latent_dim + 256, c4 * 4 * 4),
            nn.BatchNorm1d(c4 * 4 * 4), nn.ReLU(True)
        )
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(c4, c8, 4, 2, 1, bias=False), nn.BatchNorm2d(c8), nn.ReLU(True),
            nn.ConvTranspose2d(c8, c16, 4, 2, 1, bias=False), nn.BatchNorm2d(c16), nn.ReLU(True),
            nn.ConvTranspose2d(c16, c32, 4, 2, 1, bias=False), nn.BatchNorm2d(c32), nn.ReLU(True),
            nn.ConvTranspose2d(c32, c64, 4, 2, 1, bias=False), nn.BatchNorm2d(c64), nn.ReLU(True),
            nn.ConvTranspose2d(c64, 1, 4, 2, 1), nn.Tanh()
        )

    def forward(self, noise, profile):
        condition = self.profile(profile)
        x = self.project(torch.cat((noise, condition), dim=1)).view(-1, self.project[0].out_features // 16, 4, 4)
        return self.blocks(x)


class Discriminator(nn.Module):
    def __init__(self, base_channels=512, profile_encoder="mlp"):
        super().__init__()
        if base_channels < 16 or base_channels % 16:
            raise ValueError("base_channels must be a multiple of 16 and at least 16")
        c4, c8, c16, c32, c64 = (
            base_channels, base_channels // 2, base_channels // 4,
            base_channels // 8, base_channels // 16,
        )
        self.image = nn.Sequential(
            nn.Conv2d(1, c64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c64, c32, 4, 2, 1, bias=False), nn.BatchNorm2d(c32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c32, c16, 4, 2, 1, bias=False), nn.BatchNorm2d(c16), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c16, c8, 4, 2, 1, bias=False), nn.BatchNorm2d(c8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c8, c4, 4, 2, 1, bias=False), nn.BatchNorm2d(c4), nn.LeakyReLU(0.2, True)
        )
        self.profile = ProfileEncoder(c4, profile_encoder)
        self.output = nn.Linear(c4 * 4 * 4 + c4, 1)

    def forward(self, image, profile):
        image_features = self.image(image).flatten(1)
        profile_features = self.profile(profile)
        return self.output(torch.cat((image_features, profile_features), dim=1)).view(-1)


def split_dataset(dataset, split_column):
    if split_column not in dataset.rows[0]:
        raise ValueError(f"CSV has no split column {split_column!r}")
    indices = {name: [] for name in ("train", "val", "test")}
    for index, row in enumerate(dataset.rows):
        split = str(row[split_column]).strip().lower()
        if split not in indices:
            raise ValueError(f"unexpected split {row[split_column]!r}")
        indices[split].append(index)
    subsets = {name: data.Subset(dataset, values) for name, values in indices.items()}
    if not all(len(subsets[name]) for name in indices):
        raise ValueError("CSV must contain non-empty train, val, and test splits")
    return subsets


def sample_profiles(dataset, count, device):
    if len(dataset) < count:
        raise ValueError(f"Validation set has {len(dataset)} samples, but n_sample={count}")
    indices = torch.randperm(len(dataset))[:count].tolist()
    return torch.stack([dataset[index]["profile"] for index in indices]).to(device)


def infinite(loader):
    while True:
        yield from loader


@torch.inference_mode()
def calculate_validation_fid(generator, inception, loader, device, latent_dim, max_samples=None):
    real_features = []
    fake_features = []
    sample_count = 0
    generator.eval()
    inception.eval()

    for batch in loader:
        if max_samples is not None:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            batch["image"] = batch["image"][:remaining]
            batch["profile"] = batch["profile"][:remaining]

        real = batch["image"].to(device, non_blocking=True)
        profile = batch["profile"].to(device, non_blocking=True)
        fake = generator(
            torch.randn(real.shape[0], latent_dim, device=device), profile
        )
        sample_count += real.shape[0]

        real = real.clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
        fake = fake.clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
        real_features.append(inception(real)[0].flatten(1).cpu())
        fake_features.append(inception(fake)[0].flatten(1).cpu())

    if not real_features:
        raise RuntimeError("validation FID cannot be computed with an empty validation set")

    real_features = torch.cat(real_features).numpy()
    fake_features = torch.cat(fake_features).numpy()
    if real_features.shape[0] < 2:
        raise RuntimeError("validation FID requires at least two validation samples")

    return calc_fid(
        np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False),
        np.mean(real_features, axis=0), np.cov(real_features, rowvar=False),
    )


def validate_diff_aug_policy(policy):
    """Normalize and validate a comma-separated DiffAugment policy."""
    names = [name.strip() for name in policy.split(",") if name.strip()]
    unknown = sorted(set(names) - set(AUGMENT_FNS))
    if unknown:
        available = ", ".join(sorted(AUGMENT_FNS))
        raise ValueError(
            f"Unknown DiffAugment policy {unknown}; choose from: {available}"
        )
    return ",".join(names)


def main():
    parser = argparse.ArgumentParser(description="Conditional Sysmex Task 1 DCGAN trainer")
    parser.add_argument("--datasplit", required=True)
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument(
        "--brightfield_postfix",
        default="_brightfield_crop_masked_normalized_avebg_pad128.png",
        help="suffix of the preprocessed brightfield image files",
    )
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument(
        "--profile_encoder",
        choices=("mlp", "cnn"),
        default="mlp",
        help="profile conditioning encoder (default: mlp; use cnn to match train.py)",
    )
    parser.add_argument("--base_channels", type=int, default=256,
                        help="channels at the 4x4 layer; 512 gives 512->256->128->64->32, "
                             "256 gives 256->128->64->32->16")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--fid_every", type=int, default=5000,
                        help="compute validation FID every N iterations; 0 disables FID")
    parser.add_argument("--fid_batch", type=int, default=None,
                        help="validation FID batch size; defaults to --batch")
    parser.add_argument("--fid_samples", type=int, default=None,
                        help="maximum validation samples used for FID")
    parser.add_argument(
        "--diff_aug_policy", "--diff_aug", dest="diff_aug_policy", default="",
        metavar="POLICY",
        help="comma-separated DiffAugment policy (color, translation, cutout); empty disables it",
    )
    parser.add_argument("--ema", action="store_true",
                        help="use an exponential moving average generator for evaluation and samples")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay; only used with --ema (default: 0.999)")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile_mode", default="none",
                        help="torch.compile mode; none disables compilation")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--exp_dir", default="experiments/dcgan/dafault")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="fixed random seed for repeatable runs; unset keeps stochastic behavior",
    )
    args = parser.parse_args()
    print(args)
    if args.size != IMAGE_SIZE:
        raise ValueError(f"This DCGAN architecture requires --size {IMAGE_SIZE}")
    if args.base_channels < 16 or args.base_channels % 16:
        raise ValueError("base_channels must be a multiple of 16 and at least 16")
    args.diff_aug_policy = validate_diff_aug_policy(args.diff_aug_policy)
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")

    channel_widths = [
        args.base_channels,
        args.base_channels // 2,
        args.base_channels // 4,
        args.base_channels // 8,
        args.base_channels // 16,
    ]
    print("Generator:     " + " → ".join(map(str, channel_widths + [1])))
    print("Discriminator: " + " → ".join(map(str, [1] + list(reversed(channel_widths)))))

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    use_bf16 = args.bf16 and device.type == "cuda"
    dataset = SysmexTask1Dataset(
        args.datasplit,
        args.preprocessed_root,
        brightfield_postfix=args.brightfield_postfix,
    )
    subsets = split_dataset(dataset, args.split_column)
    train_set, val_set = subsets["train"], subsets["val"]
    train_loader_generator = torch.Generator()
    val_loader_generator = torch.Generator()
    if args.seed is not None:
        train_loader_generator.manual_seed(args.seed)
        val_loader_generator.manual_seed(args.seed + 1)
    loader = data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=train_loader_generator if args.seed is not None else None
    )
    val_loader = data.DataLoader(
        val_set, batch_size=args.fid_batch or args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=val_loader_generator if args.seed is not None else None
    )
    generator_base = Generator(
        args.latent, args.base_channels, profile_encoder=args.profile_encoder
    ).to(device)
    discriminator_base = Discriminator(
        args.base_channels, profile_encoder=args.profile_encoder
    ).to(device)
    generator_ema = None
    if args.ema:
        generator_ema = Generator(
            args.latent, args.base_channels, profile_encoder=args.profile_encoder
        ).to(device)
        generator_ema.load_state_dict(generator_base.state_dict())
        generator_ema.eval()
        for parameter in generator_ema.parameters():
            parameter.requires_grad_(False)
    generator = maybe_compile(generator_base, args.compile_mode)
    discriminator = maybe_compile(discriminator_base, args.compile_mode)
    g_optim = optim.Adam(generator_base.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_optim = optim.Adam(discriminator_base.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    start = 0
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        generator_base.load_state_dict(checkpoint["generator"])
        discriminator_base.load_state_dict(checkpoint["discriminator"])
        g_optim.load_state_dict(checkpoint["g_optim"])
        d_optim.load_state_dict(checkpoint["d_optim"])
        if generator_ema is not None:
            if checkpoint.get("generator_ema") is not None:
                generator_ema.load_state_dict(checkpoint["generator_ema"])
            else:
                generator_ema.load_state_dict(generator_base.state_dict())
        start = checkpoint.get("iteration", 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.exp_dir, timestamp)
    os.makedirs(os.path.join(run_dir, "sample"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoint"), exist_ok=True)
    fid_log_path = os.path.join(run_dir, "validation_fid.jsonl")
    print("run_dir:", run_dir)
    if args.fid_every > 0:
        open(fid_log_path, "a", encoding="utf-8").close()
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__)))

    try:
        git_status = subprocess.check_output(
            ["git", "status"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_status.txt"), "w", encoding="utf-8") as file:
            file.write(git_status)

        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_hash.txt"), "w", encoding="utf-8") as file:
            file.write(git_hash.strip())

        git_diff = subprocess.check_output(
            ["git", "diff"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_diff.txt"), "w", encoding="utf-8") as file:
            file.write(git_diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with open(os.path.join(run_dir, "git_info_error.txt"), "w", encoding="utf-8") as file:
            file.write("Git information could not be retrieved. (Not a git repository or git not installed)\n")

    writer = SummaryWriter(log_dir=run_dir)
    inception = None

    fixed_profile = sample_profiles(val_set, args.n_sample, device)
    fixed_noise = torch.randn(args.n_sample, args.latent, device=device)
    batches = infinite(loader)
    progress = tqdm(range(start, args.iter), dynamic_ncols=True)
    for iteration in progress:
        batch = next(batches)
        real = batch["image"].to(device, non_blocking=True)
        profile = batch["profile"].to(device, non_blocking=True)
        noise = torch.randn(real.shape[0], args.latent, device=device)

        d_optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            fake = generator(noise, profile).detach()
            real_score = discriminator(
                DiffAugment(real, policy=args.diff_aug_policy), profile
            )
            fake_score = discriminator(
                DiffAugment(fake, policy=args.diff_aug_policy), profile
            )
            d_loss = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
            d_loss += F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
        d_loss.backward()
        d_optim.step()

        g_optim.zero_grad(set_to_none=True)
        noise = torch.randn(real.shape[0], args.latent, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            fake = generator(noise, profile)
            g_loss = F.binary_cross_entropy_with_logits(
                discriminator(
                    DiffAugment(fake, policy=args.diff_aug_policy), profile
                ), torch.ones(real.shape[0], device=device)
            )
        g_loss.backward()
        g_optim.step()
        if generator_ema is not None:
            update_ema(generator_ema, generator_base, args.ema_decay)
        progress.set_description(f"d: {d_loss.item():.4f}; g: {g_loss.item():.4f}")
        step = iteration + 1
        writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
        writer.add_scalar("Loss/Generator", g_loss.item(), step)
        writer.add_scalar("Score/Real", real_score.mean().item(), step)
        writer.add_scalar("Score/Fake", fake_score.mean().item(), step)

        if args.fid_every > 0 and step % args.fid_every == 0:
            if inception is None:
                inception = load_patched_inception_v3().to(device).eval()
            validation_fid = calculate_validation_fid(
                generator_ema if generator_ema is not None else generator,
                inception, val_loader, device, args.latent, args.fid_samples
            )
            writer.add_scalar("Validation/FID", validation_fid, step)
            with open(fid_log_path, "a", encoding="utf-8") as fid_log:
                fid_log.write(json.dumps({"iteration": step, "fid": validation_fid}) + "\n")
            print(f"validation FID: {validation_fid:.6f}")
            generator.train()


        if args.sample_every > 0 and step % args.sample_every == 0:
            sample_generator = generator_ema if generator_ema is not None else generator
            sample_generator.eval()
            with torch.no_grad():
                samples = sample_generator(fixed_noise, fixed_profile)
            utils.save_image(samples, os.path.join(run_dir, "sample", f"{step:06d}.png"),
                             nrow=int(args.n_sample ** 0.5), normalize=True, value_range=(-1, 1))
            generator.train()
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            checkpoint_state = {
                "iteration": step,
                "generator": generator_base.state_dict(),
                "discriminator": discriminator_base.state_dict(),
                "g_optim": g_optim.state_dict(),
                "d_optim": d_optim.state_dict(),
                "args": vars(args),
            }
            if generator_ema is not None:
                checkpoint_state["generator_ema"] = generator_ema.state_dict()
            torch.save(
                checkpoint_state,
                os.path.join(run_dir, "checkpoint", f"{step:06d}.pt"),
            )

    writer.close()


if __name__ == "__main__":
    main()
