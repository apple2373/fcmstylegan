# FcmStyleGAN

Sanitychek Data: WBCAtt
```
mkdir -p data
cd data
wget https://huggingface.co/datasets/apple2373/wbcattplus/resolve/main/pbcseg_final_v1.tar?download=true
mv pbcseg_final_v1.tar\?download\=true pbcseg_final_v1.tar
tar xf pbcseg_final_v1.tar
cd pbcseg_final_v1
rm -rf *.png
cd ../../
```
```
# make ./data/pbcseg_final_v1_class/
# Resets the directory and sorts images like BA_01.jpg into a BA/ folder
i=0; total=$(find ./data/pbcseg_final_v1/ -maxdepth 1 -type f | wc -l)
echo "処理を開始します（総ファイル数: $total）"
for file in ./data/pbcseg_final_v1/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        classname=$(echo "$filename" | cut -d'_' -f1)
        
        mkdir -p "./data/pbcseg_final_v1_class/$classname"
        cp "$file" "./data/pbcseg_final_v1_class/$classname/"
        
        ((i++))
        # \r を使って常に同じ行の先頭に戻り、出力を上書きします
        printf "\r[進捗: %d/%d] クラス分類コピー中... 現在のファイル: %s\033[K" "$i" "$total" "$filename"
    fi
done
echo -e "\nすべて完了しました！"

python prepare_data.py --out ./data/pbcseg_final_v1.lmdb --n_worker 8 --size 64,128,256 ./data/pbcseg_final_v1_class/
```


```
conda create -n fcmstylegan python=3.12 -y
conda activate fcmstylegan
pip install uv
uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
#  uv pip install tqdm pillow lmdb click ninja binarized-atomic-gemm
uv pip install tqdm pillow lmdb click ninja tensorboard
# conda install -c nvidia cuda-toolkit -y
# module load CUDA/13.0.0

``` 

```
python train.py --size 128 --batch 16 --iter 800000 --channel_multiplier 1 ./data/pbcseg_final_v1.lmdb 

python train.py --size 128 --batch 32 --iter 800000 --channel_multiplier 1 ./data/pbcseg_final_v1.lmdb --bf16 --d_reg_every 64 --g_reg_every 32

python train.py --size 128 --batch 32 --iter 800000 --channel_multiplier 1 --bf16 --d_reg_every 64 --g_reg_every 32 ./data/pbcseg_final_v1.lmdb
python train.py --size 128 --batch 64 --iter 800000 --channel_multiplier 1 --bf16 --d_reg_every 32 --g_reg_every 16 --path_batch_shrink 4 ./data/pbcseg_final_v1.lmdb
python train.py --size 128 --batch 128 --iter 800000 --channel_multiplier 1 --bf16 --d_reg_every 16 --g_reg_every 8 --path_batch_shrink 8 ./data/pbcseg_final_v1.lmdb
python train.py --size 128 --batch 256 --iter 800000 --channel_multiplier 1 --bf16 --d_reg_every 8 --g_reg_every 4 --path_batch_shrink 16 ./data/pbcseg_final_v1.lmdb


OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py     --size 128     --batch 32     --iter 800000     --channel_multiplier 1     --bf16     --d_reg_every 64     --g_reg_every 32     ./data/pbcseg_final_v1.lmdb     --compile_mode default


module load CUDA/13.0.0
module load Miniconda3
conda activate fcmstylegan
cd /home/satoshi.tsutsui/satoshissd2/fcmstylegan
OMP_NUM_THREADS=8  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py     --size 128     --batch 32     --iter 800000     --channel_multiplier 1     --bf16     --d_reg_every 64     --g_reg_every 32     ./data/pbcseg_final_v1.lmdb     --compile_mode default

sbatch -J stgn2 --gpus pro6000:4 --time 2-00:00:00  --wrap="module load CUDA/13.0.0; module load Miniconda3; conda activate fcmstylegan; cd /projects/_ssd/satoshissd2/fcmstylegan/; OMP_NUM_THREADS=8  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py     --size 128     --batch 32     --iter 800000     --channel_multiplier 1     --bf16     --d_reg_every 64     --g_reg_every 32     ./data/pbcseg_final_v1.lmdb     --compile_mode default"

sbatch -J stgn2 --gpus a6000:4 --time 2-00:00:00  --wrap="module load CUDA/13.0.0;  module load Miniconda3; conda activate fcmstylegan; cd /projects/_ssd/satoshissd2/fcmstylegan/; OMP_NUM_THREADS=8  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py     --size 128     --batch 32     --iter 800000     --channel_multiplier 1     --bf16     --d_reg_every 64     --g_reg_every 32     ./data/pbcseg_final_v1.lmdb     --compile_mode default"


sbatch \
    --job-name=stgn2 \
    --gpus=pro6000:4 \
    --time=2-00:00:00 \
    --wrap='bash -lc "
        module load CUDA/13.0.0
        source ~/.bashrc
        conda activate fcmstylegan
        cd /projects/_ssd/satoshissd2/fcmstylegan
        export OMP_NUM_THREADS=8
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        torchrun --standalone --nproc_per_node=4 train.py \
            --size 128 \
            --batch 32 \
            --iter 800000 \
            --channel_multiplier 1 \
            --bf16 \
            --d_reg_every 64 \
            --g_reg_every 32 \
            ./data/pbcseg_final_v1.lmdb \
            --compile_mode default
    "'

```

ToDO
- [done] remove original op completely so that nvcc compile will not even run later 
- [done] make savedir configurable
- [done] save jpeg instead
- [done] make compile workable
- [probbaly not] make the training completely resumable
- [done] remove unnecessary augs?
- [done] checkFID periodically? 
- [done] replace dataset class free of lmdb

  CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    --size 128 \
    --batch 32 \
    --iter 800000 \
    --channel_multiplier 1 \
    --bf16 \
    --d_reg_every 64 \
    --g_reg_every 32 \
    ./data/pbcseg_final_v1.lmdb \
    --compile_mode default

## DiffAugment for DCGAN training

`train_dcgan.py` supports the differentiable augmentation method from the [MIT HAN Lab Data-Efficient GANs repository](https://github.com/mit-han-lab/data-efficient-gans) and the [DiffAugment paper](https://arxiv.org/abs/2006.10738). DiffAugment applies randomly sampled, differentiable image transformations to the inputs of the discriminator. This makes it harder for the discriminator to memorize a small training set and can improve GAN training when data is limited.

The augmentation is applied consistently to real images and generated images during the discriminator update. During the generator update, it is applied to generated images too, so gradients flow through the augmentation into the generator. Validation FID and saved samples are intentionally computed without DiffAugment.

The available policies are:

- `color`: random brightness, saturation, and contrast.
- `translation`: random spatial translation with padded borders.
- `cutout`: randomly masks a rectangular image region.

Pass a comma-separated policy to enable it; an empty policy (the default) disables augmentation:

```bash
python train_dcgan.py \
  --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --diff_aug_policy color,translation,cutout
```

`--diff_aug` is an alias for `--diff_aug_policy`. Start with `color,translation,cutout` for very small datasets, or use a subset such as `color,translation` for larger datasets. The selected policy is saved in each run's `args.json`.

## Optional EMA generator

Use `--ema` to maintain an exponential moving average of the generator. When enabled, EMA weights are used for validation FID and saved samples, and are stored in checkpoints as `generator_ema`. EMA is disabled by default. The decay can be configured with `--ema_decay` (default `0.999`):

```bash
python train_dcgan.py ... --ema --ema_decay 0.999
```

## Conditional brightfield training

`train.py` now trains a grayscale conditional StyleGAN2 using `BrightFieldProfileDataset`. The CSV supplies `cell_id` values, while the preprocessing directory supplies the brightfield PNGs and the `(SSC, CD45, mask)` profile archives:

```bash
python train.py --datasplit ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --id_column cell_id \
  --size 128 --batch 32 --iter 800000 \
  --channel_multiplier 1 \
  --bf16 \
  --d_reg_every 64 \
  --g_reg_every 32 \
  --compile_mode default \
  --mode pad --orientation horizontal --normalized
```

Generated images can be sampled from profiles in the same dataset with:

```bash
python generate.py --ckpt experiments/<run>/checkpoint/010000.pt \
  --csv ./data/task1_dataset_split.csv \
  --preprocessed_root ./data/task1_processed/ \
  --size 128 --pics 20
```

The conditional model is grayscale and expects profile tensors with shape `(batch, 3, 128)`. Existing unconditional/RGB checkpoints are not compatible with this model.

* Task1FCMPreprocessed is old one. task1_processed is correct one.

## for debugging the fid thing quickly (remove compile option for quicker test)
 CUDA_VISIBLE_DEVICES=0 python train.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/Task1FCMPreprocessed/ \
    --id_column cell_id \
    --size 128 \
    --batch 32 \
    --iter 800000 \
    --channel_multiplier 1 \
    --d_reg_every 64 \
    --g_reg_every 32 \
    --mode pad \
    --orientation horizontal \
    --normalized \
    --fid_every 1000 \
    --fid_batch 32 \
    --bf16 \
    --compile_mode default 
  
## history
  ```
python train.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/Task1FCMPreprocessed/ \
    --id_column cell_id \
    --size 128 \
    --batch 32 \
    --iter 800000 \
    --channel_multiplier 1 \
    --bf16 \
    --d_reg_every 64 \
    --g_reg_every 32 \
    --compile_mode default \
    --mode pad \
    --orientation horizontal \
    --normalized
 ```
-> mode collapse?

Then GPT suggested 
  ```
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --datasplit ./data/task1_dataset_split.csv \
    --preprocessed_root ./data/Task1FCMPreprocessed/ \
    --id_column cell_id \
    --size 128 \
    --batch 32 \
    --iter 800000 \
    --channel_multiplier 1 \
    --lr 0.001 \
    --d_reg_every 16 \
    --g_reg_every 4 \
    --augment \
    --bf16 \
    --compile_mode default \
    --mode pad \
    --orientation horizontal \
    --normalized
```

Why these changes?
--lr 0.001 (instead of the default 0.002)
Scientific imaging datasets are usually much smaller and less diverse than FFHQ, so a lower learning rate is often more stable.
--d_reg_every 16
This is the original StyleGAN2 default and gives the discriminator stronger, more frequent R1 regularization.
--g_reg_every 4
Also the original StyleGAN2 default. Path length regularization helps keep the generator well-behaved.
--augment
Enables Adaptive Discriminator Augmentation (ADA), which is particularly useful when the dataset isn't very large.


### If this run still collapses
Then my next experiment would be:
--lr 0.0005
--r1 20

new script
```
CUDA_VISIBLE_DEVICES=2 python3 train.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 32 \
      --iter 800000 \
      --channel_multiplier 1 \
      --lr 0.001 \
      --d_reg_every 16 \
      --g_reg_every 4 \
      --augment \
      --bf16 \
      --compile_mode default
```

Then mode cllapse. GPT says The main issue is likely the learning rate. For this custom conditional StyleGAN2, --lr 0.001 may be too aggressive and can cause collapse.

  Try:

```
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
      --fid_every 5000
```

  If collapse continues:

  1. Try --lr 0.0001.
  2. Temporarily remove --compile_mode default for easier debugging.
  3. Try --augment_p 0.1 instead of adaptive augmentation.
  4. Check whether discriminator scores rapidly become extreme while generator loss increases.

  I would first change only 0.001 → 0.0002 and reduce the initial run to 200000 iterations.


Well, these seems not work (20260815_081433 in olliemain). i think the background is the issue. so • You’re right—we do not know the generated foreground mask during training, so we cannot safely replace only the generated background.

The safest approach is:

- Train the generator to produce zero background.
- Add mild instance noise to the entire real and fake images only when passing them to the discriminator.
- Keep the generator’s actual output clean.
- Apply the same noise process to real and fake images.

This prevents the discriminator from relying on exact zero-padding without requiring a generated mask.

real_for_d = real_image + Gaussian noise
fake_for_d = fake_image + Gaussian noise

implemented : 
```
CUDA_VISIBLE_DEVICES=2 python3 train.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 32 \
      --iter 200000 \
      --channel_multiplier 1 \
      --lr 0.001 \
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
BTW  --iter 200000 looks enough 

DCGAN
```

CUDA_VISIBLE_DEVICES=2 python3 train_dcgan.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 128 \
      --iter 100000 \
      --lr 0.0002 \
      --beta1 0.5 \
      --bf16 \
      --compile_mode default \
      --base_channels 128 \ 


CUDA_VISIBLE_DEVICES=0 python3 train_dcgan.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 128 \
      --iter 100000 \
      --lr 0.0002 \
      --beta1 0.5 \
      --bf16 \
      --compile_mode default \
      --base_channels 256 \ 

CUDA_VISIBLE_DEVICES=0 python3 train_dcgan.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 128 \
      --iter 100000 \
      --lr 0.0002 \
      --beta1 0.5 \
      --bf16 \
      --compile_mode default \
      --base_channels 512 \ 


CUDA_VISIBLE_DEVICES=0 python3 train_dcgan.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 128 \
      --iter 100000 \
      --lr 0.0002 \
      --beta1 0.5 \
      --bf16 \
      --compile_mode default \
      --base_channels 1024 \ 
```

• For StyleGAN at channel_multiplier=1, size=128:

  - Generator: 512 → 512 → 512 → 512 → 256 → 128 → 1
  - Discriminator: 1 → 128 → 256 → 512 → 512 → 512 → 512

- after checking ['20260815_093638',
 '20260815_094038',
 '20260815_094723',
 '20260815_104948',
 '20260815_110256'], 
- 128 seems too little. 
- over 256 seems okay. maybe just 512 for now, because later we will move to color images. 


### next is instance noise
CUDA_VISIBLE_DEVICES=2 python3 train_dcgan.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 128 \
      --iter 100000 \
      --lr 0.0002 \
      --beta1 0.5 \
      --bf16 \
      --compile_mode default \
      --base_channels 512 \
      --instance_noise_sigma 0.03 \
      --instance_noise_sigma_final 0.0 \
      --instance_noise_decay cosine
-> not so effective, decided not to do. 


### try again

CUDA_VISIBLE_DEVICES=0 python3 train.py \
      --datasplit ./data/task1_dataset_split.csv \
      --preprocessed_root ./data/task1_processed/ \
      --size 128 \
      --batch 32 \
      --iter 800000 \
      --channel_multiplier 1 \
      --lr 0.001 \
      --d_reg_every 16 \
      --g_reg_every 4 \
      --augment \
      --bf16 \
      --compile_mode default \
      --fid_every 5000 \
      --brightfield_postfix _brightfield_crop_masked_normalized_randbg_pad128.png


### current default
CUDA_VISIBLE_DEVICES=0 python3 train.py       --datasplit ./data/task1_dataset_split.csv       --preprocessed_root ./data/task1_processed/       --size 128       --batch 32       --iter 800000       --channel_multiplier 1       --lr 0.001       --d_reg_every 16       --g_reg_every 4       --augment       --bf16       --compile_mode default       --brightfield_postfix _brightfield_crop_masked_normalized_avebg_pad128.png
