import argparse

import torch
from torchvision import utils
from tqdm import tqdm

from brightfield_dataset import BrightFieldProfileDataset
from model import Generator


def generate(args, g_ema, device, mean_latent, dataset):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            item = dataset[i % len(dataset)]
            profile = item["profile"].unsqueeze(0).repeat(args.sample, 1, 1).to(device)
            sample, _ = g_ema(
                [sample_z],
                profile=profile,
                truncation=args.truncation,
                truncation_latent=mean_latent,
            )
            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate brightfield images conditioned on FCM profiles")
    parser.add_argument("--csv", required=True, help="task1_dataset_split.csv")
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument("--id_column", default="cell_id")
    parser.add_argument("--mode", choices=("pad", "resize"), default="pad")
    parser.add_argument("--orientation", choices=("horizontal", "vertical"), default="horizontal")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--pics", type=int, default=20)
    parser.add_argument("--truncation", type=float, default=1)
    parser.add_argument("--truncation_mean", type=int, default=4096)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    dataset = BrightFieldProfileDataset(
        args.csv, args.preprocessed_root, id_column=args.id_column,
        mode=args.mode, orientation=args.orientation, normalized=args.normalized,
    )
    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
        channel_multiplier=args.channel_multiplier, out_channels=1,
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    g_ema.load_state_dict(checkpoint["g_ema"])

    mean_latent = None
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)

    generate(args, g_ema, device, mean_latent, dataset)
