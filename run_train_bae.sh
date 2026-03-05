#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

# 彻底清除 NFS 上所有之前安装的本地包，使用容器预装环境
rm -rf /home/runai-home/.local/lib/
# 只安装其他依赖，torch 使用容器预装版本
pip install -r requirements.txt -q

python tasks/train_bae_model.py \
    --dataset /nfs/home/ttao/Data/paired_oasis/pairwise_oasis/ \
    --cache_dir /nfs/home/ttao/cache/ \
    --output_dir /nfs/home/ttao/Projects/TADM-3D/checkpoints \
    --run_name bae_run1 \
    --n_epochs 100 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_workers 0
