# JiT and flow-matching experiments

## Why the existing diffusion implementations cannot be reused unchanged

The project dataset contains grayscale images shaped `(1, 128, 128)` and continuous profiles shaped `(3, 128)`. The existing U-Net accepts `(image, noise_level, profile)` and predicts DDPM noise or an EDM residual.

The upstream JiT code is designed for RGB ImageNet images. Its model accepts `(image, timestep, class_label)`, uses a class embedding, and its bundled `Denoiser` owns a different continuous-time velocity objective and ODE sampler. Therefore, replacing the U-Net with the unmodified JiT model would mix incompatible input channels, conditioning, loss targets, and samplers.

The upstream `JiT/` source is kept in place and imported by `jit_model.py`; it is not copied. JiT also requires the dependencies listed in `JiT/environment.yaml` (notably `einops`). Install that environment or install those dependencies into the active training environment before running `train_jit.py`. The adapter changes the input to one channel and replaces the class embedding with the project's flattened profile MLP.

## Profile encoder

The default `mlp` encoder is `Flatten -> Linear(3 * 128, hidden_dim) -> SiLU`. It is now the default for `train_diffusion.py`.

## Flow matching

`FlowMatchingProcess` uses `x_t = t * image + (1 - t) * noise` and trains the model to predict `image - noise`. Sampling integrates the predicted velocity from `t=0` to `t=1` using Euler or Heun steps. This objective is independent of the backbone, so it can train both the U-Net and JiT.

## Experiments

1. **JiT + DDPM:** `train_jit.py --objective ddpm`; JiT predicts epsilon and DDPM/DDIM sampling is used.
2. **JiT + EDM:** `train_jit.py --objective edm`; JiT is used as the EDM residual network and Euler/Heun sampling is used.
3. **U-Net + flow matching:** `train_diffusion.py --objective flow_matching`; the U-Net predicts velocity and Euler/Heun sampling is used.
4. **JiT + flow matching:** `train_jit.py --objective flow_matching`; JiT predicts velocity and Euler/Heun sampling is used.

For fair comparisons, keep the dataset split, profile encoder, image size, seed, batch size, EMA, fixed profiles, fixed noise, and evaluation procedure constant.

## Example commands

The commands below use the same dataset and GPU convention as the current
U-Net experiment. Each experiment has its own output directory.

### 1. JiT + DDPM

```bash
CUDA_VISIBLE_DEVICES=3 python train_jit.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --model JiT-B/16 \
  --objective ddpm \
  --sampler ddim \
  --batch 32 \
  --bf16 \
  --compile_mode default \
  --fid_samples 1000 \
  --exp_dir experiments/diffusion/jit_ddpm
```

JiT uses DDPM training and DDIM sampling here. Use `--sampler ddpm` if you
want the full ancestral DDPM sampler instead of DDIM.

### 2. U-Net + flow matching

```bash
CUDA_VISIBLE_DEVICES=3 python train_diffusion.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --backbone compact \
  --objective flow_matching \
  --profile_encoder mlp \
  --sampler heun \
  --batch 128 \
  --bf16 \
  --compile_mode default \
  --fid_samples 1000 \
  --exp_dir experiments/diffusion/unet_flow_matching
```

Use `--sampler euler` for the cheaper first-order flow-matching sampler.

### 3. JiT + EDM

```bash
CUDA_VISIBLE_DEVICES=3 python train_jit.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --model JiT-B/16 \
  --objective edm \
  --sampler heun \
  --batch 32 \
  --bf16 \
  --compile_mode default \
  --fid_samples 1000 \
  --exp_dir experiments/diffusion/jit_edm
```

JiT is used as the EDM residual model here.

### 4. JiT + flow matching

```bash
CUDA_VISIBLE_DEVICES=3 python train_jit.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --model JiT-B/16 \
  --objective flow_matching \
  --sampler heun \
  --batch 32 \
  --bf16 \
  --compile_mode default \
  --fid_samples 1000 \
  --exp_dir experiments/diffusion/jit_flow_matching
```

For an initial smoke test, add `--iter 100 --sample_every 50 --fid_every 0` to
any command. JiT generally needs a smaller batch than the compact U-Net because
of its transformer memory usage.
## Original JiT-style training

For the paper-style JiT procedure, use `train_jit_original.py`. It is separate
from `train_jit.py` so the controlled objective comparisons remain unambiguous.
It uses JiT's logit-normal timestep sampling, clean-image prediction followed
by velocity conversion, profile dropout with a learned null embedding, and the
JiT-style ODE sampler. It writes the same `args.json`, TensorBoard scalars,
`validation_fid.jsonl`, checkpoints, samples, and `timing.txt` as the other
trainers.

```bash
CUDA_VISIBLE_DEVICES=3 python train_jit_original.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root /dev/shm/satoshi.tsutsui/data/task1_processed/ \
  --model JiT-B/16 \
  --sampler heun \
  --num_sampling_steps 50 \
  --cfg 1.0 \
  --batch 32 \
  --bf16 \
  --compile_mode default \
  --fid_samples 1000 \
  --exp_dir experiments/diffusion_sweep/jit_original_seed0 \
  --seed 0
```

