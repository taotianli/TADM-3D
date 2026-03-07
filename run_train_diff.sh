#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

# RunAI/Kubernetes 容器中 UID 可能不存在于 /etc/passwd
export USER=${USER:-ttao}

# 清除旧的本地包
rm -rf /home/runai-home/.local/lib/
pip install -r requirements.txt -q
pip install monai-generative --no-deps -q

nvidia-smi

python -c "import torch; import monai; print('torch:', torch.__version__); print('monai:', monai.__version__)"

python tasks/train_diff_model.py \
    --dataset /nfs/home/ttao/Data/paired_oasis/pairwise_oasis/ \
    --cache_dir /nfs/home/ttao/cache/ \
    --output_dir /nfs/home/ttao/Projects/TADM-3D/checkpoints \
    --bae_ckpt /nfs/home/ttao/Projects/TADM-3D/checkpoints/bae-best.pth \
    --run_name diff_run1 \
    --n_epochs 500 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_workers 4
