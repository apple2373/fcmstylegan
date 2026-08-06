#!/usr/bin/env python3
"""Compute robust nonzero profile scales from the training split only."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from brightfield_dataset import BrightFieldProfileDataset


def compute_q99(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    values = [[], [], []]

    for batch_index, batch in enumerate(loader):
        profile = batch["profile"].float()  # (batch, 3, 128)

        for channel in range(3):
            active = profile[:, channel, :].abs() > 1e-12
            values[channel].append(profile[:, channel, :][active].cpu())

        if batch_index % 100 == 0:
            print(f"Processed batch {batch_index}")

    scale = torch.empty(3, dtype=torch.float64)
    count = torch.empty(3, dtype=torch.long)

    for channel in range(3):
        channel_values = torch.cat(values[channel]).double()
        if channel_values.numel() == 0:
            raise RuntimeError(
                f"Profile channel {channel} contains no nonzero values"
            )

        scale[channel] = torch.quantile(channel_values, 0.99)
        count[channel] = channel_values.numel()

        if scale[channel] <= 0:
            raise RuntimeError(f"Q99 scale for channel {channel} is not positive")

    return scale, count


def main() -> None:
    # Edit these values directly when using a different dataset/configuration.
    datasplit = "./data/task1_dataset_split.csv"
    preprocessed_root = "./data/Task1FCMPreprocessed/"
    id_column = "cell_id"
    split_column = "split"
    mode = "pad"
    orientation = "horizontal"
    normalized = True
    batch_size = 256
    num_workers = 4
    dataset = BrightFieldProfileDataset(
        datasplit,
        preprocessed_root,
        id_column=id_column,
        mode=mode,
        orientation=orientation,
        normalized=normalized,
    )

    if split_column not in dataset.rows[0]:
        raise ValueError(f"CSV has no split column {split_column!r}")

    split_indices = {"train": [], "val": [], "test": []}
    for index, row in enumerate(dataset.rows):
        split = row[split_column].strip().lower()
        if split not in split_indices:
            raise ValueError(
                f"Unexpected split {row[split_column]!r}; "
                "expected train, val, or test"
            )
        split_indices[split].append(index)

    train_dataset = Subset(dataset, split_indices["train"])
    val_dataset = Subset(dataset, split_indices["val"])
    test_dataset = Subset(dataset, split_indices["test"])

    if not len(train_dataset):
        raise ValueError("The train split is empty")

    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    scale, count = compute_q99(loader)
    names = ("SSC", "CD45", "mask")

    print("\nNonzero counts:")
    for name, value in zip(names, count):
        print(f"  {name}: {int(value.item())}")

    print("\nNonzero 99th percentile scales:")
    for name, value in zip(names, scale):
        print(f"  {name}: {value.item():.9f}")

    print("\nPython constant:")
    print(f"PROFILE_SCALE = {tuple(scale.tolist())}")


if __name__ == "__main__":
    main()
