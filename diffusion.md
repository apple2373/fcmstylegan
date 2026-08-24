# Diffusion model roadmap

This project generates 128×128 single-channel images conditioned on a continuous profile with shape `(3, 128)`. The dataset is domain-specific and likely much smaller than ImageNet, so the most useful roadmap is one that isolates the effects of the backbone, diffusion parameterization, sampler, and conditioning method.

## Important distinction

These are separate choices:

- **Backbone:** small U-Net, ADM-style U-Net, or DiT.
- **Training/parameterization:** vanilla DDPM or EDM.
- **Sampler:** ancestral DDPM, DDIM, or an EDM ODE/SDE sampler such as Heun.
- **Conditioning:** profile injection into the denoiser, optionally with guidance.

ADM is mainly a U-Net architecture. EDM is a training, preconditioning, and sampling design. They can be combined as an EDM-trained ADM U-Net; EDM does not inherently require ADM. The official EDM implementation supports DDPM++, NCSN++, and ADM backbones: [EDM implementation](https://github.com/NVlabs/edm).

## Recommended roadmap

| Phase | Model/configuration | Purpose |
|---|---|---|
| 1 | Conditional DDPM with a compact U-Net | Minimal working diffusion baseline, comparable to DCGAN |
| 2 | Conditional ADM-style U-Net with DDPM training | Strong convolutional baseline from *Diffusion Models Beat GANs* |
| 3 | The same ADM-style U-Net with EDM training/preconditioning | Test whether EDM improves this dataset |
| 4 | Classifier-free guidance | Improve conditional fidelity if profile adherence is weak |
| 5 | Conditional DiT | Scaling experiment, only worthwhile with enough data/compute |

The first implementation should probably combine phases 1 and 2 as a configurable compact ADM-style U-Net. Then phase 3 can reuse the same backbone and change mainly the noise parameterization, loss, and sampler.

## Phase 1: conditional DDPM baseline

Train a denoiser to predict the noise added to an image:

```text
clean image x0 + profile
          ↓
add noise at timestep t
          ↓
conditional U-Net predicts ε
          ↓
MSE(predicted_noise, true_noise)
```

Recommended conditioning:

- Encode the profile with a 1D CNN, matching the CNN option already available in the GAN trainers.
- Embed the timestep.
- Project the profile and timestep embeddings into each residual block using FiLM or adaptive group normalization.
- Use one input and output image channel.

An MLP profile encoder can remain as an ablation, but the CNN should be the primary encoder because the profile bins have local structure.

This is based on the original DDPM formulation: [DDPM paper](https://arxiv.org/abs/2006.11239).

Use a fixed validation subset, fixed evaluation profiles, and fixed sampling noise when comparing checkpoints. DDIM can be used to reduce evaluation cost without changing DDPM training: [DDIM paper](https://arxiv.org/abs/2010.02502).

## Phase 2: ADM-style U-Net

The paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) is the right reference for the stronger convolutional baseline. Its ADM model improves the U-Net with design choices such as residual blocks, adaptive normalization, multi-resolution attention, and improved up/downsampling.

For this project, use a smaller version rather than reproducing the ImageNet-scale model exactly:

- Residual blocks at each resolution.
- Timestep and profile conditioning through adaptive normalization.
- Attention only at selected lower resolutions, such as 32×32 and 16×16, if memory allows.
- Keep the model width and number of blocks modest to match the dataset size.

This should be the main diffusion counterpart to the StyleGAN experiment. The paper's reported ImageNet results demonstrate the architecture at scale, but those FIDs should not be expected on this biological dataset.

## Phase 3: EDM with the ADM backbone

Keep the phase-2 ADM-style network fixed and change the diffusion design:

- EDM noise-level sampling and preconditioning.
- EDM loss weighting.
- Continuous noise parameterization.
- Heun or another EDM sampler for evaluation.

This makes the comparison meaningful: improvements can be attributed mostly to EDM rather than simultaneously changing the network. EDM systematically separates these design choices and reports much faster high-quality sampling: [EDM paper](https://arxiv.org/abs/2206.00364).

## Phase 4: classifier-free guidance

The original ADM paper uses classifier guidance for class labels, which requires a separate classifier trained on noisy images. That is not a natural fit for the continuous profile condition.

Instead, use classifier-free guidance:

- Randomly drop the profile condition for a fraction of training examples.
- At sampling time, run the denoiser with and without the profile.
- Combine the two predictions with a guidance scale.

This is optional. It can improve profile fidelity, but excessive guidance may reduce diversity or produce unrealistic cells. Measure both image quality and profile consistency.

## Phase 5: conditional DiT

A DiT replaces the U-Net with Transformer blocks operating on image patches. For this dataset:

- Patchify the noisy 128×128 image with patch size 4 or 8.
- Add timestep embeddings.
- Encode the profile as conditioning tokens or a conditioning vector.
- Use adaptive layer normalization for conditioning.
- Predict noise or velocity.

The DiT paper demonstrates strong scaling with model size and compute: [DiT paper](https://arxiv.org/abs/2212.09748). However, DiT is not automatically the best choice here. With limited domain-specific data, a convolutional ADM/EDM model may generalize better and be much cheaper to train. Treat DiT as a scaling experiment, not as the default SOTA model.

## Practical experiment order

1. Implement a conditional DDPM/ADM-style U-Net with the CNN profile encoder.
2. Establish deterministic sampling, fixed validation data, FID, and profile-consistency metrics.
3. Compare CNN and MLP profile encoders.
4. Add DDIM sampling for faster evaluation.
5. Add EDM training/preconditioning while keeping the same U-Net.
6. Test classifier-free guidance.
7. Only then evaluate a conditional DiT if the dataset size and compute justify it.

The implementation is provided as `train_diffusion.py` with reusable `diffusion_model.py` and `diffusion_process.py` modules. It reuses the existing dataset, profile encoders, fixed-seed utilities, validation FID, checkpoints, and sample logging.

Example runs:

```bash
# Phase 1: compact conditional DDPM with DDIM sampling
/home/satoshi/miniconda3/envs/fcmstylegan/bin/python train_diffusion.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/task1_processed/ \
    --backbone compact \
    --objective ddpm \
    --profile_encoder cnn \
    --sampler ddim \
    --sample_steps 50 \
    --seed 123 \
    --batch 32 \
    --num_workers 4 \
    --sample_every 1000 \
    --fid_every 5000 \
    --exp_dir experiments/diffusion/phase1_compact_ddpm_ddim

# Phase 2: ADM-style conditional DDPM
python train_diffusion.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/task1_processed/ \
    --backbone adm --objective ddpm --profile_encoder cnn

# Phase 3: EDM objective with the ADM-style backbone
python train_diffusion.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/task1_processed/ \
    --backbone adm --objective edm --profile_encoder cnn
```

For DDPM evaluation, `--sampler auto` now selects DDIM with 50 steps. Use `--sampler ddpm` for the slower full ancestral reference sampler, or explicitly use `--sampler ddim --sample_steps 50`. EDM uses Heun sampling by default; use `--sampler euler` or `--sampler heun`, and control the number of steps with `--sample_steps`.


## Training objective versus sampler

Do not confuse the training objective with the sampler:

- `--objective ddpm` controls how the denoiser is trained.
- `--sampler ddim` or `--sampler ddpm` controls how the trained denoiser generates images.
- DDIM is not a separate training objective in this implementation.

A model trained with the DDPM objective can be sampled with either DDIM or ancestral DDPM because both samplers use the same trained denoiser. For example, the same checkpoint can be used with:

```bash
# Fast validation and development sampling
--objective ddpm --sampler ddim --sample_steps 50

# Slower full DDPM evaluation
--objective ddpm --sampler ddpm
```

Recommended evaluation workflow:

1. Train using `--objective ddpm`.
2. Use DDIM with 50 steps for routine previews and validation FID.
3. Decide the final sampler and step count using the validation split.
4. Evaluate the selected protocol on the test split only after the model and sampler are fixed.

If the final reported metric is intended to represent full ancestral diffusion, use `--sampler ddpm` for the test evaluation. If the intended application prioritizes fast generation, use DDIM and report its step count explicitly. Do not choose between samplers based on test results.

## Which sampler should I use?

A sampler is the procedure used **after training** to turn random noise into an image. It does not change the model weights or training objective. The sampler runs the denoiser repeatedly, and `--sample_steps` controls how many denoising steps it takes.

| Objective | Sampler | Main tradeoff | Recommended use |
|---|---|---|---|
| DDPM | `ddpm` | Stochastic and usually slow; uses the full noise schedule | Explicit reference comparison |
| DDPM | `ddim` | Much faster and deterministic-style; can use fewer steps | Default routine sampler |
| EDM | `euler` | Fast first-order solver; lower cost per step | Fast baseline or quick previews |
| EDM | `heun` | More accurate second-order solver; roughly two denoiser evaluations per step | Recommended EDM sampler for quality |

### DDPM sampler

`ddpm` follows the original stochastic reverse diffusion process. With the default 1000 training steps, it can be expensive, but it is a useful reference. Use it as:

```bash
--objective ddpm --sampler ddpm
```

### DDIM sampler

`ddim` uses the same DDPM-trained denoiser but follows a faster, non-ancestral trajectory. It can usually generate an image with 20–100 steps instead of 1000. In this implementation, DDIM uses a fixed initial noise tensor, so generation is reproducible when `--seed` is fixed.

Use it as:

```bash
--objective ddpm --sampler ddim --sample_steps 50
```

### Euler and Heun samplers

EDM treats denoising as a continuous noise-level trajectory. Euler takes one numerical step per noise level. Heun evaluates the denoiser again to correct that step, so it is slower per step but generally more accurate.

Use them as:

```bash
# Fast EDM preview
--objective edm --sampler euler --sample_steps 30

# Recommended EDM evaluation
--objective edm --sampler heun --sample_steps 40
```

### Practical recommendation

There is no universally best sampler; the best choice depends on the trained objective and the number of steps. For the planned experiments:

1. Use **DDIM with 50 steps** by default for DDPM training during normal sampling and FID evaluation.
2. Use **DDPM with the full schedule** only as a slower reference experiment.
3. Use **Heun with 40 steps** for EDM evaluation.
4. Use **Euler with 20–30 steps** for quick previews.

Compare samplers using the same checkpoint, fixed profiles, and fixed initial noise. Select the final sampler using both FID and profile-consistency metrics, since a lower FID alone does not guarantee that the generated image follows the requested profile.

## `--backbone` versus `--objective`

These two arguments control different parts of the diffusion model.

### `--backbone`: the neural network architecture

The backbone is the image-processing network that receives a noisy image, a noise level, and a profile, then predicts the denoising signal.

```bash
--backbone compact
```

Uses a smaller U-Net:

- Fewer channels and attention layers.
- Faster and less memory-intensive.
- Good for the first working baseline.

```bash
--backbone adm
```

Uses a larger ADM-style U-Net:

- More residual capacity.
- More attention at lower resolutions.
- Adaptive conditioning through residual blocks.
- Intended as the stronger convolutional baseline.

The backbone answers: **“What network processes the noisy image?”**

### `--objective`: how the network is trained

The objective determines how noise is sampled, how the network output is parameterized, and how the training loss is weighted.

```bash
--objective ddpm
```

Uses the discrete DDPM formulation:

- Samples a discrete timestep from the DDPM schedule.
- Adds noise using that timestep.
- Trains the network to predict the added noise.
- Uses DDPM or DDIM samplers at inference time.

```bash
--objective edm
```

Uses the EDM formulation:

- Samples a continuous noise level `σ`.
- Uses EDM preconditioning and loss weighting.
- Trains the network to produce an EDM-preconditioned denoised image.
- Uses Euler or Heun samplers at inference time.

The objective answers: **“What diffusion process and loss train the network?”**

### Valid combinations

| Backbone | Objective | Meaning |
|---|---|---|
| `compact` | `ddpm` | Smallest and simplest baseline |
| `adm` | `ddpm` | Strong ADM-style DDPM baseline |
| `compact` | `edm` | EDM ablation with a small network |
| `adm` | `edm` | Recommended strong experiment: EDM with an ADM-style U-Net |

The recommended progression is:

```bash
# First working baseline
--backbone compact --objective ddpm

# Stronger convolutional baseline
--backbone adm --objective ddpm

# Recommended EDM experiment
--backbone adm --objective edm
```

Changing `--backbone` changes the network architecture. Changing `--objective` changes the diffusion training and noise parameterization. To compare them fairly, keep the dataset, profile encoder, seed, batch size, and evaluation procedure fixed whenever possible.

## Trainer options

The trainer is configured through `train_diffusion.py`:

- `--backbone compact|adm`: choose the Phase 1 or Phase 2 U-Net.
- `--objective ddpm|edm`: choose the training objective.
- `--profile_encoder cnn|mlp`: choose the profile encoder; CNN is the default.
- `--sampler auto|ddpm|ddim|euler|heun`: choose the sampling method; `auto` means DDIM for DDPM and Heun for EDM.
- `--sample_steps N`: override the number of sampling steps. If omitted, DDPM+DDIM uses **50** steps and EDM+Heun uses **40** steps.
- `--seed N`: enable reproducible model, sampler, DataLoader, and worker randomness.
- `--bf16`: enable bfloat16 autocasting on CUDA.
- `--compile_mode none|default|reduce-overhead|max-autotune`: enable `torch.compile`; `none` is the default.
- `--ckpt PATH`: resume from a diffusion checkpoint.
- `--sample_every N`: save generated sample grids every N iterations.
- `--checkpoint_every N`: save checkpoints every N iterations; the default is 10,000.
- `--fid_every N`: calculate validation FID every N iterations; use `0` to disable it.

The required dataset arguments are:

```bash
--datasplit ./data/task1_dataset_split.csv \
--preprocessed_root ./data/task1_processed/
```

A complete example is:

```bash
/home/satoshi/miniconda3/envs/fcmstylegan/bin/python train_diffusion.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/task1_processed/ \
    --backbone adm \
    --objective ddpm \
    --sampler ddim \
    --sample_steps 50 \
    --profile_encoder cnn \
    --seed 123 \
    --exp_dir experiments/diffusion/adm_ddim
```

`--sampler` affects generation during sample saving and FID evaluation; it does not change the training objective. For DDPM training, `auto` selects DDIM with 50 steps, `ddpm` uses the full ancestral schedule, and `ddim` uses 50 steps unless `--sample_steps` overrides it. For EDM training, `auto` selects Heun with 40 steps; use `euler` or `heun` to choose explicitly.

Each run writes `args.json`, `validation_fid.jsonl`, sample images, checkpoints, and Git provenance files (`git_status.txt`, `git_hash.txt`, and `git_diff.txt`) under `--exp_dir`. Checkpoints contain the online model, EMA model, optimizer state, iteration, and arguments. The EMA model is used for samples and FID.


# Command notes;

python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone compact     --objective ddpm     --profile_encoder cnn     --sampler ddim     --sample_steps 50     --seed 123     --batch 64     --num_workers 4     --sample_every 1000   --exp_dir experiments/diffusion/phase1_compact_ddpm_ddim --bf16 --compile default --fid_samples 1000 

python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone compact --objective ddpm --profile_encoder cnn --batch 128 --bf16 --compile default --fid_samples 1000

python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone compact --objective ddpm --sampler ddpm  --profile_encoder cnn --batch 128 --bf16 --compile default --fid_samples 1000

python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone adm --objective ddpm --profile_encoder cnn --batch 128 --bf16 --compile default --fid_samples 1000

python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/      --backbone adm --objective edm --profile_encoder cnn --batch 128 --bf16 --compile default --fid_samples 1000 


CUDA_VISIBLE_DEVICES=3 python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone compact --objective ddpm --profile_encoder mlp --batch 128 --bf16 --compile default --fid_samples 1000   --exp_dir experiments/diffusion/phase1_compact

CUDA_VISIBLE_DEVICES=2 python train_diffusion.py     --datasplit ./data/task1_dataset_split.csv     --preprocessed_root ./data/task1_processed/     --backbone compact --objective ddpm --sampler ddpm  --profile_encoder mlp --batch 128 --bf16 --compile default --fid_samples 1000   --exp_dir experiments/diffusion/phase1_compact