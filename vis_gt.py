"""Visualize the ground-truth images used as train_dcgan.py fixed profiles.

The validation samples are selected with the same seeded ``torch.randperm``
call as ``train_dcgan.py``.  This makes the output correspond to the profiles
used for saved DCGAN sample grids when the same seed and arguments are used.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision import utils

from reproducibility import seed_everything
from sysmex_task1_dataset import SysmexTask1Dataset


def split_dataset(dataset, split_column):
    """Return dataset indices grouped by train/val/test, like train_dcgan.py."""
    if split_column not in dataset.rows[0]:
        raise ValueError(f"CSV has no split column {split_column!r}")

    indices = {name: [] for name in ("train", "val", "test")}
    for index, row in enumerate(dataset.rows):
        split = str(row[split_column]).strip().lower()
        if split not in indices:
            raise ValueError(f"unexpected split {row[split_column]!r}")
        indices[split].append(index)

    if not all(indices[name] for name in indices):
        raise ValueError("CSV must contain non-empty train, val, and test splits")
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Save GT images corresponding to train_dcgan.py fixed_profile"
    )
    parser.add_argument("--datasplit", required=True)
    parser.add_argument("--preprocessed_root", required=True)
    parser.add_argument("--brightfield_postfix", default=(
        "_brightfield_crop_masked_normalized_avebg_pad128.png"
    ))
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument(
        "--seed", type=int, default=123,
        help="must match train_dcgan.py --seed (default: 123)",
    )
    parser.add_argument(
        "--nrow", type=int, default=None,
        help="images per row; defaults to int(sqrt(n_sample))",
    )
    parser.add_argument("--output", default="fixed_profile_gt.jpg")
    args = parser.parse_args()

    if args.n_sample <= 0:
        raise ValueError("--n_sample must be positive")
    if args.nrow is not None and args.nrow <= 0:
        raise ValueError("--nrow must be positive")

    # Keep this before sampling, exactly as in train_dcgan.py.
    seed_everything(args.seed)
    dataset = SysmexTask1Dataset(
        args.datasplit,
        args.preprocessed_root,
        brightfield_postfix=args.brightfield_postfix,
    )
    subsets = split_dataset(dataset, args.split_column)
    val_indices = subsets["val"]
    if len(val_indices) < args.n_sample:
        raise ValueError(
            f"Validation set has {len(val_indices)} samples, "
            f"but n_sample={args.n_sample}"
        )

    # This is the same selection as sample_profiles(val_set, ...).
    selected_positions = torch.randperm(len(val_indices))[:args.n_sample].tolist()
    selected_indices = [val_indices[position] for position in selected_positions]
    samples = [dataset[index] for index in selected_indices]
    images = torch.stack([sample["image"] for sample in samples])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrow = args.nrow or max(1, int(math.sqrt(args.n_sample)))
    utils.save_image(images, output_path, nrow=nrow, normalize=True, value_range=(-1, 1))

    ids = [sample["metadata"].get("cell_id") for sample in samples]
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "datasplit": str(args.datasplit),
                "preprocessed_root": str(args.preprocessed_root),
                "seed": args.seed,
                "n_sample": args.n_sample,
                "selected_cell_ids": ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.n_sample} GT images to {output_path}")
    print(f"saved selected cell IDs to {metadata_path}")


if __name__ == "__main__":
    main()
