import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
import pandas as pd
from PIL import Image

# %run -i plot_test_results.py #use with -i option

# Edit these settings before running with %run.

# `ROOT` will be provided by notebook side
# ROOT = Path("~/projects/fcmstylegan/experiments/archive/dcgan_baseline_20260817_231036").expanduser()
# ROOT = Path("~/projects/fcmstylegan/experiments/archive/stylegan_baseline_20260819_170800").expanduser()


PREPROCESSED_ROOT = Path("~/projects/fcmstylegan/data/task1_processed").expanduser()
TOP_K = 1
METRICS = ["psnr", "ssim", "psnr_masked", "ssim_masked", "lpips", "lpips_crop"]

GROUPS = ["Best", "Median", "Worst"]

GENERATED_DIR = ROOT / "test_eval_imgs"
PER_IMAGE_FILE = ROOT / "test_eval_imgs.jsonl"
STAT_FILE = ROOT / "test_eval_stat.json"
IMAGE_SUFFIX = "_brightfield_crop_masked_normalized_avebg_pad128.png"
MASK_SUFFIX = "_mask_crop_pad128.png"

higher_is_better = {
    "psnr": True, "ssim": True, "psnr_masked": True,
    "ssim_masked": True, "lpips": False, "lpips_crop": False,
}


def safe_id(image_id):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id))
    return name.strip("._") or "image"


def selected_samples(data, metric, top_k):
    valid = data.dropna(subset=[metric]).copy()
    if higher_is_better[metric]:
        best = valid.nlargest(top_k, metric)
        worst = valid.nsmallest(top_k, metric)
    else:
        best = valid.nsmallest(top_k, metric)
        worst = valid.nlargest(top_k, metric)
    median_value = valid[metric].median()
    distances = (valid[metric] - median_value).abs()
    median = valid.loc[distances.nsmallest(top_k).index].sort_values(metric)
    return {"Best": best, "Median": median, "Worst": worst}


def crop_to_mask(image, mask_path, margin=8, resampling=Image.Resampling.BILINEAR):
    mask = Image.open(mask_path).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(image.width, right + margin)
    bottom = min(image.height, bottom + margin)
    return image.crop((left, top, right, bottom)).resize(image.size, resampling)

def show_image(ax, path, title, mask_path=None, cropped=False, mask=False, masked=False):
    ax.axis("off")
    if path.exists():
        # Preserve 16-bit/float image data; converting to L can wash out GT images.
        image = Image.open(path)
        if cropped:
            image = crop_to_mask(
                image, mask_path,
                resampling=Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR,
            )
        image_array = np.asarray(image)
        if masked:
            mask_image = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
            mask_array = np.asarray(mask_image) > 0
            image_array = np.where(mask_array, image_array, image_array.min())
        ax.imshow(image_array, cmap="gray")
        ax.set_title(title, fontsize=9)
    else:
        ax.text(0.5, 0.5, f"Missing:\n{path}", ha="center", va="center", fontsize=8)


with STAT_FILE.open() as file:
    aggregate = json.load(file)

df = pd.read_json(PER_IMAGE_FILE, lines=True)
available_metrics = [metric for metric in METRICS if metric in df.columns]
df[available_metrics] = df[available_metrics].apply(pd.to_numeric, errors="coerce")

metric_decimals = {
    "psnr": 2, "ssim": 4, "psnr_masked": 2,
    "ssim_masked": 4, "lpips": 4, "lpips_crop": 4,
}

summary = pd.DataFrame({
    "mean": df[available_metrics].mean(),
    "median": df[available_metrics].median(),
    "std": df[available_metrics].std(),
    "min": df[available_metrics].min(),
    "max": df[available_metrics].max(),
    "count": df[available_metrics].count(),
})
summary = summary.apply(
    lambda column: column.map(
        lambda item: f"{item:.{metric_decimals.get(column.name, 4)}f}"
        if pd.notna(item) else ""
    )
)

display(pd.DataFrame([aggregate]).T.rename(columns={0: "value"}))
display(summary)

def plot_cell(cell_id, metric=None, title=None, title_metrics=None, show=True):
    """Display all metrics and image comparisons for one cell ID."""
    matches = df[df["image_id"].astype(str) == str(cell_id)]
    if matches.empty:
        raise KeyError(f"Cell ID not found: {cell_id}")
    sample = matches.iloc[0]

    if metric is not None and metric not in available_metrics:
        raise ValueError(f"Unknown metric {metric!r}; choose from {available_metrics}")

    sample_table = sample[available_metrics].to_frame().T
    sample_table.index = [str(cell_id)]
    for column in available_metrics:
        decimals = metric_decimals.get(column, 4)
        sample_table[column] = sample_table[column].map(
            lambda item: f"{item:.{decimals}f}" if pd.notna(item) else ""
        )
    display(sample_table)

    stem = str(cell_id)
    generated_path = GENERATED_DIR / f"{safe_id(cell_id)}.png"
    gt_path = PREPROCESSED_ROOT / f"{stem}{IMAGE_SUFFIX}"
    mask_path = PREPROCESSED_ROOT / f"{stem}{MASK_SUFFIX}"

    if title_metrics is None:
        title_metrics = []
    elif isinstance(title_metrics, str):
        title_metrics = [title_metrics]
    invalid_metrics = [name for name in title_metrics if name not in available_metrics]
    if invalid_metrics:
        raise ValueError(f"Unknown title metric(s): {invalid_metrics}; choose from {available_metrics}")

    if title is None:
        label = None if metric is None else f"{metric.upper()}|value={sample[metric]:.4f}"
    else:
        label = title

    if title_metrics:
        metric_line = " | ".join(
            f"{name.upper()}={sample[name]:.{metric_decimals.get(name, 4)}f}"
            for name in title_metrics
        )
        title_lines = ([label] if label else []) + [metric_line, f"id={cell_id}"]
    else:
        title_lines = [label, f"id={cell_id}"] if label else [f"id={cell_id}"]

    fig, axes = plt.subplots(3, 3, figsize=(6, 7.2))
    fig.suptitle("\n".join(title_lines), fontsize=12)
    show_image(axes[0, 0], generated_path, "Generated")
    show_image(axes[0, 1], gt_path, "Ground truth")
    show_image(axes[0, 2], mask_path, "Ground-truth mask", mask=True)
    show_image(axes[1, 0], generated_path, "Generated (masked)", mask_path, masked=True)
    show_image(axes[1, 1], gt_path, "Ground truth (masked)", mask_path, masked=True)
    show_image(axes[1, 2], mask_path, "Ground-truth mask (masked)", mask=True)
    show_image(axes[2, 0], generated_path, "Generated (cropped)", mask_path, True)
    show_image(axes[2, 1], gt_path, "Ground truth (cropped)", mask_path, True)
    show_image(
        axes[2, 2], mask_path, "Ground-truth mask (cropped)",
        mask_path, True, True,
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


# One figure per selected image: full comparison row and cropped comparison row.
for metric in available_metrics:
    for group, samples in selected_samples(df, metric, TOP_K).items():
        if group not in GROUPS:
            continue
        for rank, (_, sample) in enumerate(samples.iterrows(), start=1):
            image_id = sample["image_id"]
            value = sample[metric]
            metric_label = metric.upper()
            if group == "Best":
                label = f"Top{rank} {metric_label}"
            elif group == "Worst":
                label = f"Bottom{rank} {metric_label}"
            else:
                label = f"{metric_label}={value:.4f}"
            fig, _ = plot_cell(
                image_id, metric=metric,
                title=f"{label}|value={value:.4f}",
            )
            plt.close(fig)
            display(HTML("<hr style=\"border: 1px solid #999;\">"))
        display(HTML('<hr style="border: 3px double #333;">'))
    display(HTML('<hr style="border: 3px double #000;">'))
