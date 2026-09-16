"""Train profile-conditioned JiT with DDPM or flow matching."""
import argparse
import copy
import json
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
import torch
torch.set_float32_matmul_precision("high")
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from tqdm import tqdm
from calc_inception import load_patched_inception_v3
from diffusion_process import DDPMProcess, EDMProcess, FlowMatchingProcess
from reproducibility import seed_everything, seed_worker
from run_utils import save_git_metadata
from sysmex_task1_dataset import SysmexTask1Dataset
from train_diffusion import (
    calculate_validation_fid, make_run_dir, sample_profiles, split_dataset, update_ema,
)

def main():
    parser = argparse.ArgumentParser(description="Profile-conditioned JiT trainer")
    parser.add_argument("--datasplit", required=True)
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--brightfield_postfix", default="_brightfield_crop_masked_normalized_avebg_pad128.png")
    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--objective", choices=("ddpm", "edm", "flow_matching"), default="ddpm")
    parser.add_argument("--sampler", choices=("auto", "ddpm", "ddim", "euler", "heun"), default="auto")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--fid_every", type=int, default=5000)
    parser.add_argument("--fid_batch", type=int, default=None)
    parser.add_argument("--fid_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile_mode", default="none")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--exp_dir", default="experiments/jit")
    args = parser.parse_args()
    print(args)
    from jit_model import ConditionalJiT

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.objective == "ddpm":
        process = DDPMProcess(device=device)
        sampler = "ddim" if args.sampler == "auto" else args.sampler
        if sampler not in {"ddpm", "ddim"}: raise ValueError("DDPM requires --sampler ddpm or ddim")
        sampling_steps = args.sample_steps or (50 if sampler == "ddim" else process.steps)
    elif args.objective == "edm":
        process = EDMProcess()
        sampler = "heun" if args.sampler == "auto" else args.sampler
        if sampler not in {"euler", "heun"}: raise ValueError("EDM requires --sampler euler or heun")
        sampling_steps = args.sample_steps or 40
    else:
        process = FlowMatchingProcess()
        sampler = "euler" if args.sampler == "auto" else args.sampler
        if sampler not in {"euler", "heun"}: raise ValueError("flow_matching requires --sampler euler or heun")
        sampling_steps = args.sample_steps or 50
    dataset = SysmexTask1Dataset(args.datasplit, args.preprocessed_root, brightfield_postfix=args.brightfield_postfix)
    subsets = split_dataset(dataset, args.split_column)
    train_set, val_set = subsets["train"], subsets["val"]
    train_generator = torch.Generator()
    val_generator = torch.Generator()
    if args.seed is not None:
        train_generator.manual_seed(args.seed)
        val_generator.manual_seed(args.seed + 1)

    loader = data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=train_generator if args.seed is not None else None,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=args.fid_batch or args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=val_generator if args.seed is not None else None,
    )
    model_base = ConditionalJiT(
        args.model, args.img_size, args.attn_dropout, args.proj_dropout,
    ).to(device)
    print("Model =", model_base)
    n_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    model = torch.compile(model_base, mode=args.compile_mode) if args.compile_mode.lower() not in {"none", "off", "false", "0"} else model_base
    ema = copy.deepcopy(model_base).eval()
    for parameter in ema.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model_base.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint.get("iteration", 0)
        print("Resumed checkpoint from", args.ckpt)
    else:
        print("Training from scratch")
    run_dir = make_run_dir(args.exp_dir)
    sample_dir = os.path.join(run_dir, "sample")
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    fid_log_path = os.path.join(run_dir, "validation_fid.jsonl")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.fid_every > 0:
        open(fid_log_path, "a", encoding="utf-8").close()
    print("run_dir:", run_dir)
    print("device:", device, "objective:", args.objective, "model:", args.model)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as file: json.dump(vars(args), file, indent=2)
    shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__))); save_git_metadata(run_dir)
    writer = SummaryWriter(log_dir=run_dir)
    fixed_profile = sample_profiles(val_set, args.n_sample, device)
    fixed_noise = torch.randn(args.n_sample, 1, args.img_size, args.img_size, device=device)
    use_bf16 = args.bf16 and device.type == "cuda"
    inception = None
    print(f"Start training for {args.iter} iterations")
    start_wall_time = datetime.now(timezone.utc)
    start_perf_time = time.perf_counter()
    timing_path = os.path.join(run_dir, "timing.txt")
    with open(timing_path, "w", encoding="utf-8") as timing_file:
        timing_file.write(f"start_time_utc: {start_wall_time.isoformat(timespec='seconds')}\n")
    progress, batches = tqdm(range(start, args.iter), dynamic_ncols=True), iter(loader)
    for iteration in progress:
        try: batch = next(batches)
        except StopIteration: batches = iter(loader); batch = next(batches)
        image, profile = batch["image"].to(device, non_blocking=True), batch["profile"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16): loss = process.training_loss(model, image, profile)
        loss.backward(); optimizer.step(); update_ema(ema, model_base, decay=0.9999)
        step = iteration + 1; progress.set_description(f"loss: {loss.item():.5f}"); writer.add_scalar("Loss/Train", loss.item(), step)
        if args.sample_every > 0 and step % args.sample_every == 0:
            samples = process.sample(ema, fixed_profile, fixed_noise.shape, sampler=sampler, sampling_steps=sampling_steps, noise=fixed_noise)
            utils.save_image(samples.clamp(-1, 1), os.path.join(sample_dir, f"{step:06d}.jpg"), nrow=max(1, int(args.n_sample ** 0.5)), normalize=True, value_range=(-1, 1))
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save({"iteration": step, "model": model_base.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}, os.path.join(checkpoint_dir, f"{step:06d}.pt"))
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
    end_wall_time = datetime.now(timezone.utc)
    elapsed = timedelta(seconds=int(time.perf_counter() - start_perf_time))
    with open(timing_path, "a", encoding="utf-8") as timing_file:
        timing_file.write(f"end_time_utc: {end_wall_time.isoformat(timespec='seconds')}\n")
        timing_file.write(f"elapsed: {elapsed}\n")
    writer.close()
    print("Training time:", elapsed)

if __name__ == "__main__": main()
