# Checkpoint argument metadata

## What changed

`train.py` now stores StyleGAN training arguments as a plain dictionary:

```python
"args": vars(args)
```

DCGAN and diffusion checkpoints already use this dictionary format.

`checkpoint_utils.py` provides `load_checkpoint_args()` for evaluation and
other tools that need to reconstruct the configuration used for a checkpoint.

## Compatibility

The resolver supports:

- New checkpoints with `args` stored as a dictionary.
- Older StyleGAN checkpoints with `args` stored as an `argparse.Namespace`.
- Checkpoints without embedded arguments when the run directory contains
  `args.json`.

For a checkpoint at:

```text
experiments/stylegan/<run>/checkpoint/010000.pt
```

 the fallback metadata file is:

```text
experiments/stylegan/<run>/args.json
```

Checkpoint files are loaded as trusted PyTorch files because legacy
StyleGAN checkpoints may contain a serialized `argparse.Namespace`.

## Manual overrides

Explicit values can be supplied through the `overrides` mapping. They always
win over checkpoint and `args.json` values:

```python
from checkpoint_utils import load_checkpoint_args

config = load_checkpoint_args(
    "experiments/stylegan/<run>/checkpoint/010000.pt",
    overrides={
        "datasplit": "data/task1_dataset_split.csv",
        "preprocessed_root": "data/task1_processed",
        "split": "test",
        "num_samples": 5000,
    },
)
```

The resolution order is:

```text
sibling args.json -> embedded checkpoint args -> manual overrides
```

Dataset paths, evaluation split, sample count, seed, and sampler settings
should normally be supplied as manual evaluation overrides. Architecture
settings such as `profile_encoder`, `base_channels`, `backbone`, `objective`,
and `size` should match the training checkpoint unless the model loader can
infer them independently.
## Masked metrics and per-image output

Use `--use_mask` to load `SysmexTask1Dataset` masks and add
`psnr_masked_mean` and `ssim_masked_mean`. These metrics use only foreground
pixels. Per-image metrics are always saved as JSONL; when `--output eval.json`
is supplied, the default path is `eval_per_image.jsonl`. It can be overridden
with `--per_image_output`.

```bash
python evaluate.py \
  --model dcgan \
  --ckpt path/to/checkpoint.pt \
  --datasplit data/task1_dataset_split.csv \
  --preprocessed_root data/task1_processed \
  --use_mask \
  --output dcgan_eval.json
```
Generated images can be saved as individual grayscale PNG files with:

```bash
--save_images_dir generated_dcgan
```

Filenames are derived from the dataset cell IDs.
When `--use_mask` is enabled, LPIPS reports both:

- `lpips_mean`: full-image LPIPS, including background.
- `lpips_crop_mean`: LPIPS on both images cropped to the real-image mask
  bounding box with an 8-pixel context margin, then resized consistently.

The per-image JSONL contains `lpips` and `lpips_crop` for the same two values.
With `--use_mask`, the evaluator also reports `fid_crop`. It uses the real
foreground mask bounding box with the same 8-pixel context margin as cropped
LPIPS, applies that crop to both real and generated images, and computes FID
on the resized crops. Standard `fid` remains the full-image metric.
