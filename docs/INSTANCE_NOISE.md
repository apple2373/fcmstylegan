# Instance Noise for Conditional StyleGAN2

## Purpose

`train.py` supports instance noise to reduce discriminator shortcutting and mode collapse when training segmented cells on 128x128 padded images. This is useful when padded backgrounds are mostly homogeneous and differ mainly because cells occupy different crop areas.

A discriminator can otherwise learn the fixed background or padding pattern instead of judging the cell structure and profile condition. Instance noise makes the discriminator less sensitive to exact pixel-level background values.

## How it works

During the discriminator update, Gaussian noise is added to both real and generated images:

```text
real image -> optional augmentation -> instance noise -> discriminator
fake image -> optional augmentation -> instance noise -> discriminator
```

The same treatment is used for the generator update, so gradients still pass from the discriminator through the noisy image to the generator. The generator output itself is not permanently modified.

Instance noise is not applied to:

- saved generated samples;
- validation FID images; or
- the clean images stored in the dataset.

The noise operation is differentiable with respect to the image. The sampled Gaussian tensor is treated as a constant for backpropagation.

## Command-line arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--instance_noise_sigma` | `0.0` | Initial Gaussian standard deviation. `0` disables instance noise. |
| `--instance_noise_sigma_final` | `0.0` | Standard deviation at the end of the schedule. |
| `--instance_noise_decay` | `cosine` | Schedule: `none`, `linear`, or `cosine`. |
| `--instance_noise_decay_iters` | `None` | Number of iterations over which noise decays. Defaults to `--iter`. |

The image range is `[-1, 1]`, so sigma values should be small. The implementation clips noisy discriminator inputs back to `[-1, 1]`.

## Recommended starting point

For the homogeneous-padding mode-collapse issue, start with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/task1_processed/ \
    --size 128 \
    --batch 32 \
    --iter 200000 \
    --channel_multiplier 1 \
    --lr 0.0002 \
    --d_reg_every 16 \
    --g_reg_every 4 \
    --augment \
    --bf16 \
    --compile_mode default \
    --instance_noise_sigma 0.03 \
    --instance_noise_sigma_final 0.0 \
    --instance_noise_decay cosine \
    --fid_every 5000
```

A practical sigma range is approximately `0.01` to `0.05`. If training remains unstable, try lowering the initial sigma. If the discriminator still relies heavily on the padding, try a slower decay or a slightly larger initial sigma.

## Monitoring

The active sigma is written to TensorBoard as:

```text
Stats/Instance Noise Sigma
```

Compare this curve with:

- `Loss/Generator`;
- `Loss/Discriminator`;
- real and fake discriminator scores; and
- `Validation/FID`.

The clean validation FID is the metric to use for model selection. Instance noise may improve training stability without being part of the desired final image distribution.

## Important limitation

The training code does not need to know which pixels are background in generated images. It applies noise to the complete generated image before the discriminator. This avoids requiring a generated segmentation mask while still preventing the discriminator from relying on exact homogeneous padding.

If the final output must have a strictly zero background, keep the clean image representation and use post-processing or a separately trained segmentation mask. Do not use a hard threshold inside the generator-to-discriminator path because that would block useful gradients.
