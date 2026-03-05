#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

# RunAI/Kubernetes 容器中 UID 可能不存在于 /etc/passwd，
# 导致 getpass.getuser() 崩溃；提前设置 USER 让其走环境变量路径。
export USER=${USER:-ttao}

# 清除旧的本地包
rm -rf /home/runai-home/.local/lib/
# 安装依赖，monai-generative 用 --no-deps 避免拉取 torch
pip install -r requirements.txt -q
pip install monai-generative --no-deps -q

nvidia-smi

python -c "import torch; import monai; print('torch:', torch.__version__); print('monai:', monai.__version__)"

python tasks/train_bae_model.py \
    --dataset /nfs/home/ttao/Data/paired_oasis/pairwise_oasis/ \
    --cache_dir /nfs/home/ttao/cache/ \
    --output_dir /nfs/home/ttao/Projects/TADM-3D/checkpoints \
    --run_name bae_run1 \
    --n_epochs 1 \
    --batch_size 6 \
    --lr 1e-4 \
    --num_workers 0
