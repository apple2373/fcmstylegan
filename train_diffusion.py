"""Train conditional DDPM/ADM or EDM diffusion models on Sysmex images."""

import argparse
import copy
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from tqdm import tqdm

from calc_inception import load_patched_inception_v3
from diffusion_model import ConditionalUNet
from diffusion_process import DDPMProcess, EDMProcess
from fid import calc_fid
from reproducibility import seed_everything, seed_worker
from run_utils import save_git_metadata
from sysmex_task1_dataset import SysmexTask1Dataset


def maybe_compile(model, compile_mode):
    mode = str(compile_mode).lower()
    if mode in {"none", "off", "false", "0"}:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model, mode=mode)


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
        raise ValueError(f"validation set has {len(dataset)} samples, but n_sample={count}")
    indices = torch.randperm(len(dataset))[:count].tolist()
    return torch.stack([dataset[index]["profile"] for index in indices]).to(device)


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


@torch.inference_mode()
def calculate_validation_fid(
    model,
    process,
    inception,
    loader,
    device,
    sampler,
    sampling_steps,
    max_samples=None,
):
    real_features = []
    fake_features = []
    sample_count = 0
    model.eval()
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
        fake = process.sample(
            model,
            profile,
            real.shape,
            sampler=sampler,
            sampling_steps=sampling_steps,
        )
        sample_count += real.shape[0]

        real_rgb = real.clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
        fake_rgb = fake.clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
        real_features.append(inception(real_rgb)[0].flatten(1).cpu())
        fake_features.append(inception(fake_rgb)[0].flatten(1).cpu())

    if not real_features:
        raise RuntimeError("validation FID cannot be computed with an empty validation set")
    real_features = torch.cat(real_features).numpy()
    fake_features = torch.cat(fake_features).numpy()
    if real_features.shape[0] < 2:
        raise RuntimeError("validation FID requires at least two validation samples")

    return calc_fid(
        np.mean(fake_features, axis=0),
        np.cov(fake_features, rowvar=False),
        np.mean(real_features, axis=0),
        np.cov(real_features, rowvar=False),
    )


def make_run_dir(exp_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(exp_dir, timestamp)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(exp_dir, f"{timestamp}_{suffix:02d}")
        suffix += 1
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Conditional Sysmex diffusion trainer")
    parser.add_argument("--datasplit", required=True)
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument("--split_column", default="split")
    parser.add_argument(
        "--brightfield_postfix",
        default="_brightfield_crop_masked_normalized_avebg_pad128.png",
    )
    parser.add_argument(
        "--backbone", choices=("compact", "adm"), default="compact",
        help="denoiser architecture: compact U-Net or ADM-style U-Net",
    )
    parser.add_argument(
        "--objective", choices=("ddpm", "edm"), default="ddpm",
        help="training objective: discrete DDPM or continuous EDM",
    )
    parser.add_argument(
        "--sampler",
        choices=("auto", "ddpm", "ddim", "euler", "heun"),
        default="auto",
        help="sampler; auto uses DDIM (50 steps) for DDPM and Heun (40 steps) for EDM",
    )
    parser.add_argument("--profile_encoder", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=32, help="training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--fid_every", type=int, default=5000)
    parser.add_argument("--fid_batch", type=int, default=None)
    parser.add_argument("--fid_samples", type=int, default=None)
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="override sampling steps; defaults to 50 for DDIM and 40 for EDM Heun",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--compile_mode",
        default="none",
        help='torch.compile mode; "none" disables compilation',
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--exp_dir", default="experiments/diffusion")
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.objective == "ddpm":
        process = DDPMProcess(device=device)
        # DDIM is much faster for routine samples/FID; full DDPM remains
        # available explicitly as a slower reference sampler.
        sampler = "ddim" if args.sampler == "auto" else args.sampler
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError("DDPM objective requires --sampler ddpm or ddim")
        sampling_steps = args.sample_steps or (50 if sampler == "ddim" else process.steps)
    else:
        process = EDMProcess()
        sampler = "heun" if args.sampler == "auto" else args.sampler
        if sampler not in {"euler", "heun"}:
            raise ValueError("EDM objective requires --sampler euler or heun")
        sampling_steps = args.sample_steps or 40

    dataset = SysmexTask1Dataset(
        args.datasplit,
        args.preprocessed_root,
        brightfield_postfix=args.brightfield_postfix,
    )
    subsets = split_dataset(dataset, args.split_column)
    train_set, val_set = subsets["train"], subsets["val"]

    train_generator = torch.Generator()
    val_generator = torch.Generator()
    if args.seed is not None:
        train_generator.manual_seed(args.seed)
        val_generator.manual_seed(args.seed + 1)

    loader = data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=train_generator if args.seed is not None else None,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.fid_batch or args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=val_generator if args.seed is not None else None,
    )

    model_base = ConditionalUNet(
        profile_encoder=args.profile_encoder,
        backbone=args.backbone,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    # Keep the base model uncompiled so optimizer, EMA, and checkpoint keys
    # remain identical to the ordinary model (without _orig_mod prefixes).
    model = maybe_compile(model_base, args.compile_mode)
    ema = copy.deepcopy(model_base).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model_base.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start = 0

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model_base.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint.get("iteration", 0)

    run_dir = make_run_dir(args.exp_dir)
    sample_dir = os.path.join(run_dir, "sample")
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    fid_log_path = os.path.join(run_dir, "validation_fid.jsonl")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.fid_every > 0:
        open(fid_log_path, "a", encoding="utf-8").close()
    print("run_dir:", run_dir)
    print("device:", device, "objective:", args.objective, "backbone:", args.backbone)

    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__)))
    save_git_metadata(run_dir)

    writer = SummaryWriter(log_dir=run_dir)
    fixed_profile = sample_profiles(val_set, args.n_sample, device)
    fixed_noise = torch.randn(args.n_sample, 1, 128, 128, device=device)
    inception = None
    use_bf16 = args.bf16 and device.type == "cuda"
    progress = tqdm(range(start, args.iter), dynamic_ncols=True)
    batches = iter(loader)

    for iteration in progress:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        image = batch["image"].to(device, non_blocking=True)
        profile = batch["profile"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = process.training_loss(model, image, profile)
        loss.backward()
        optimizer.step()
        update_ema(ema, model_base, decay=0.9999)

        step = iteration + 1
        progress.set_description(f"loss: {loss.item():.5f}")
        writer.add_scalar("Loss/Train", loss.item(), step)

        if args.sample_every > 0 and step % args.sample_every == 0:
            samples = process.sample(
                ema, fixed_profile, fixed_noise.shape, sampler=sampler,
                sampling_steps=sampling_steps, noise=fixed_noise,
            )
            utils.save_image(
                samples.clamp(-1, 1),
                os.path.join(sample_dir, f"{step:06d}.jpg"),
                nrow=max(1, int(args.n_sample ** 0.5)),
                normalize=True,
                value_range=(-1, 1),
            )

        if args.fid_every > 0 and step % args.fid_every == 0:
            if inception is None:
                inception = load_patched_inception_v3().to(device).eval()
            validation_fid = calculate_validation_fid(
                ema, process, inception, val_loader, device, sampler,
                sampling_steps, args.fid_samples,
            )
            writer.add_scalar("Validation/FID", validation_fid, step)
            with open(fid_log_path, "a", encoding="utf-8") as fid_log:
                fid_log.write(json.dumps({"iteration": step, "fid": validation_fid}) + "\n")
            print(f"validation FID: {validation_fid:.6f}")

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(
                {
                    "iteration": step,
                    "model": model_base.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                os.path.join(checkpoint_dir, f"{step:06d}.pt"),
            )

    writer.close()


if __name__ == "__main__":
    main()
