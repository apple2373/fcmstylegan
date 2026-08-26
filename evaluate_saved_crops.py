"""Evaluate saved generated images after a configurable square center crop."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from sysmex_task1_dataset import SysmexTask1Dataset


DEFAULT_CROP_SIZE = 70


def safe_image_id(image_id: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id)).strip("._") or "image"


def center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}")
    height, width = image.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image is smaller than the {crop_size}x{crop_size} crop: {image.shape}"
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top:top + crop_size, left:left + crop_size]


def load_real_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=np.float32)
    if image.ndim == 3:
        if image.shape[2] != 1:
            raise ValueError(f"Expected a grayscale image, got {image.shape} for {path}")
        image = image[:, :, 0]
    return np.clip(image / 65535.0, 0.0, 1.0)


def load_generated_image(path: Path) -> np.ndarray:
    source = Image.open(path)
    source_dtype = np.asarray(source).dtype
    image = np.asarray(source, dtype=np.float32)
    if image.ndim == 3:
        image = image[:, :, 0]
    # New evaluate.py outputs uint16 PNGs; support older uint8 outputs too.
    scale = 65535.0 if source_dtype == np.uint16 else 255.0
    return np.clip(image / scale, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save_images_dir", required=True,
                        help="directory containing PNGs saved by evaluate.py")
    parser.add_argument("--datasplit", required=True,
                        help="dataset CSV or CSV ZIP used by evaluate.py")
    parser.add_argument("--preprocessed_root", required=True,
                        help="directory containing the real brightfield images")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--id_column", default="cell_id")
    parser.add_argument(
        "--brightfield_postfix",
        default="_brightfield_crop_masked_normalized_avebg_pad128.png",
    )
    parser.add_argument("--crop_size", type=int, default=DEFAULT_CROP_SIZE,
                        help="square center-crop size in pixels (default: 70)")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="aggregate metrics JSON path")
    parser.add_argument("--per_image_output", default=None,
                        help="per-image metrics JSONL path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lpips_batch", type=int, default=128)
    cli = parser.parse_args()

    generated_dir = Path(cli.save_images_dir)
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated image directory not found: {generated_dir}")

    dataset = SysmexTask1Dataset(
        cli.datasplit,
        cli.preprocessed_root,
        id_column=cli.id_column,
        brightfield_postfix=cli.brightfield_postfix,
        load_json_metadata=False,
    )
    indices = [
        index for index, row in enumerate(dataset.rows)
        if str(row.get(cli.split_column, "")).strip().lower() == cli.split.lower()
    ]
    if not indices:
        raise ValueError(
            f"No samples found for split {cli.split!r} using column {cli.split_column!r}"
        )
    if cli.num_samples is not None:
        indices = indices[:cli.num_samples]

    records = []
    real_batch, fake_batch = [], []
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from lpips import PerceptualLoss
    lpips_model = PerceptualLoss(
        net="alex", use_gpu=device.type == "cuda", gpu_ids=[device.index or 0]
    )

    missing = []
    for index in tqdm(indices, desc="loading center crops"):
        image_id = dataset.rows[index].get(dataset.id_column, index)
        generated_path = generated_dir / f"{safe_image_id(image_id)}.png"
        if not generated_path.is_file():
            missing.append(str(generated_path))
            continue
        real = center_crop(load_real_image(dataset._image_path(str(image_id))), cli.crop_size)
        fake = center_crop(load_generated_image(generated_path), cli.crop_size)
        real_batch.append(torch.from_numpy(real).unsqueeze(0))
        fake_batch.append(torch.from_numpy(fake).unsqueeze(0))

        error = (real - fake) ** 2
        mse = float(error.mean())
        psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
        ssim = float(structural_similarity(real, fake, data_range=1.0))
        records.append({"image_id": str(image_id), "psnr": psnr, "ssim": ssim})

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} generated image(s), for example:\n{preview}"
        )
    if not records:
        raise ValueError("No image pairs were found")

    real_tensor = torch.stack(real_batch)
    fake_tensor = torch.stack(fake_batch)
    lpips_values = []
    with torch.inference_mode():
        for start in range(0, len(records), cli.lpips_batch):
            real_rgb = real_tensor[start:start + cli.lpips_batch].repeat(1, 3, 1, 1).to(device)
            fake_rgb = fake_tensor[start:start + cli.lpips_batch].repeat(1, 3, 1, 1).to(device)
            lpips_values.append(lpips_model(real_rgb * 2 - 1, fake_rgb * 2 - 1).flatten().cpu())
    lpips_values = torch.cat(lpips_values).numpy()
    for record, value in zip(records, lpips_values):
        record["lpips"] = float(value)

    result = {
        "crop": "center",
        "crop_size": [cli.crop_size, cli.crop_size],
        "split": cli.split,
        "num_samples": len(records),
        "psnr_mean": float(np.mean([record["psnr"] for record in records])),
        "ssim_mean": float(np.mean([record["ssim"] for record in records])),
        "lpips_mean": float(lpips_values.mean()),
    }
    output_path = Path(cli.output or f"center_crop_{cli.crop_size}_metrics.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    per_image_output = cli.per_image_output or str(
        output_path.with_name(output_path.stem + "_per_image.jsonl")
    )
    with Path(per_image_output).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    print(json.dumps(result, indent=2))
    print(f"per-image metrics: {per_image_output}")


if __name__ == "__main__":
    main()
