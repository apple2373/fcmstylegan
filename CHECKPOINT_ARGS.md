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
