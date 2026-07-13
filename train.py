import argparse
import math
import random
import os
import time

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

import json
import subprocess
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
     
try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def maybe_compile(model, compile_mode):
    mode = str(compile_mode).lower()

    if mode in {"none", "off", "false", "0"}:
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    return torch.compile(model, mode=mode)


def make_run_dir(exp_dir, wait_seconds=5):

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(exp_dir, timestamp)

        if not os.path.exists(run_dir):
            return run_dir

        if get_rank() == 0:
            print(f"run dir exists, waiting {wait_seconds}s: {run_dir}")

        time.sleep(wait_seconds)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(
    args,
    loader,
    generator,
    discriminator,
    g_optim,
    d_optim,
    g_ema,
    device,
    sample_dir,
    checkpoint_dir,
    writer,
    generator_base,
    discriminator_base,
    g_ema_base,
):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
                fake_img, _ = augment(fake_img, ada_aug_p)

            else:
                real_img_aug = real_img

            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator_base.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            with torch.amp.autocast("cuda", enabled=False):
                real_pred = discriminator_base(real_img_aug)
                r1_loss = d_r1_loss(real_pred, real_img)

            discriminator_base.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)

            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)

            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator_base.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            with torch.amp.autocast("cuda", enabled=False):
                fake_img, latents = generator_base(noise, return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )

            generator_base.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_base, generator_base, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )
            
            # Log metrics to TensorBoard with custom structured names
            writer.add_scalar("Loss/Generator", g_loss_val, i)
            writer.add_scalar("Loss/Discriminator", d_loss_val, i)
            writer.add_scalar("Stats/Augment", ada_aug_p, i)
            writer.add_scalar("Stats/Rt", r_t_stat, i)
            writer.add_scalar("Loss/R1", r1_val, i)
            writer.add_scalar("Loss/Path Length Regularization", path_loss_val, i)
            writer.add_scalar("Stats/Mean Path Length", mean_path_length, i)
            writer.add_scalar("Score/Real Score", real_score_val, i)
            writer.add_scalar("Score/Fake Score", fake_score_val, i)
            writer.add_scalar("Stats/Path Length", path_length_val, i)

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                        g_ema.eval()
                        sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        os.path.join(sample_dir, f"{str(i).zfill(6)}.jpg"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": generator_base.state_dict(),
                        "d": discriminator_base.state_dict(),
                        "g_ema": g_ema_base.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    os.path.join(checkpoint_dir, f"{str(i).zfill(6)}.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="none",
        help='torch.compile mode: "none" disables compile; otherwise use "default", "reduce-overhead", or "max-autotune"',
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="high",
        choices=("highest", "high", "medium"),
        help='float32 matmul precision for CUDA (default: high)',
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="experiments",
        help="root directory for training runs",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="train with bfloat16 autocast",
    )
    parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="number of workers for data loading",
    )
    
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator_base = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator_base = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema_base = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema_base.eval()
    accumulate(g_ema_base, generator_base, 0)

    generator = maybe_compile(generator_base, args.compile_mode)
    discriminator = maybe_compile(discriminator_base, args.compile_mode)
    g_ema = maybe_compile(g_ema_base, args.compile_mode)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator_base.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator_base.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator_base.load_state_dict(ckpt["g"])
        discriminator_base.load_state_dict(ckpt["d"])
        g_ema_base.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 1. Generate timestamp and define directory paths
    run_dir = make_run_dir(args.exp_dir)
    print("run_dir:",run_dir)
    sample_dir = os.path.join(run_dir, "sample")
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    writer = SummaryWriter(log_dir=run_dir)

    # 2. Execute save operations only on the main process (Rank 0)
    if get_rank() == 0:
        # Automatically create directories (including the parent 'experiments' directory if it doesn't exist)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # (1) Save args as a JSON file
        print(args)
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

        # (2) Copy the currently running train.py file for backup
        shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__)))

        # (3) Retrieve and save Git information (status, hash, diff)
        try:
            # Get git status
            git_status = subprocess.check_output(["git", "status"], stderr=subprocess.STDOUT).decode("utf-8")
            with open(os.path.join(run_dir, "git_status.txt"), "w", encoding="utf-8") as f:
                f.write(git_status)

            # Get git commit hash (latest commit ID)
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8")
            with open(os.path.join(run_dir, "git_hash.txt"), "w", encoding="utf-8") as f:
                f.write(git_hash.strip())

            # Get git diff (changes since the last commit)
            git_diff = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode("utf-8")
            with open(os.path.join(run_dir, "git_diff.txt"), "w", encoding="utf-8") as f:
                f.write(git_diff)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Skip if it is not a git repository or git command is not installed
            with open(os.path.join(run_dir, "git_info_error.txt"), "w") as f:
                f.write("Git information could not be retrieved. (Not a git repository or git not installed)")

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(
        args,
        loader,
        generator,
        discriminator,
        g_optim,
        d_optim,
        g_ema,
        device,
        sample_dir,
        checkpoint_dir,
        writer,
        generator_base,
        discriminator_base,
        g_ema_base,
    )
