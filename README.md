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
- make the training completely resumable
- remove unnecessary augs?
- checkFID periodically? 
- replace dataset class free of lmdb

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
