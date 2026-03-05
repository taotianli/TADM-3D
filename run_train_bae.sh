#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

# 清除之前安装在 NFS 上的旧包
rm -rf /home/runai-home/.local/lib/python3.10/site-packages/torch*
rm -rf /home/runai-home/.local/lib/python3.10/site-packages/torchvision*
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
