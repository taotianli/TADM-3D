#!/bin/bash

cd /nfs/home/ttao/Projects/TADM-3D

export USER=${USER:-ttao}

rm -rf /home/runai-home/.local/lib/
pip install -r requirements.txt -q
pip install monai-generative --no-deps -q

python tasks/infer_fm_model.py \
    --dataset     /nfs/home/ttao/Data/paired_oasis/pairwise_oasis/ \
    --fm_ckpt     /nfs/home/ttao/Projects/TADM-3D/checkpoints/tafm-best.pth \
    --output_dir  /nfs/home/ttao/Projects/TADM-3D/outputs/infer_fm_5 \
    --n_samples   5 \
    --split       test \
    --num_workers 4 \
    --interpolant stochastic \
    --solver      heun \
    --n_steps     20
