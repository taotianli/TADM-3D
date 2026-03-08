#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

export USER=${USER:-ttao}

rm -rf /home/runai-home/.local/lib/
pip install -r requirements.txt -q
pip install monai-generative --no-deps -q

nvidia-smi

python -c "import torch; import monai; print('torch:', torch.__version__); print('monai:', monai.__version__)"

python tasks/train_fm_model.py \
    --dataset     /nfs/home/ttao/Data/paired_oasis/pairwise_oasis/ \
    --cache_dir   /nfs/home/ttao/cache/ \
    --output_dir  /nfs/home/ttao/Projects/TADM-3D/checkpoints \
    --bae_ckpt    /nfs/home/ttao/Projects/TADM-3D/checkpoints/bae-best.pth \
    --run_name    fm_run1 \
    --n_epochs    500 \
    --batch_size  1 \
    --lr          1e-4 \
    --num_workers 4 \
    --interpolant stochastic \
    --solver      heun \
    --n_steps     20 \
    --lambda_cons 0.1
