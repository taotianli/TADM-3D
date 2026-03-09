#!/bin/bash
# Ablation study runner for TAFM-3D
# Usage: bash run_ablation.sh [a0|a1|a2|a3|a4|a5|a6|a7|a8|all]

cd /nfs/home/ttao/Projects/TADM-3D

export USER=${USER:-ttao}

# Environment setup
rm -rf /home/runai-home/.local/lib/
pip install -r requirements.txt -q
pip install monai-generative --no-deps -q

nvidia-smi
python -c "import torch; print('torch:', torch.__version__)"

DATASET=/nfs/home/ttao/Data/paired_oasis/pairwise_oasis/
CACHE=/nfs/home/ttao/cache/
CKPT_ROOT=/nfs/home/ttao/Projects/TADM-3D/checkpoints
BAE_CKPT=${CKPT_ROOT}/bae-best.pth
COMMON="--dataset ${DATASET} --cache_dir ${CACHE} --bae_ckpt ${BAE_CKPT} --n_epochs 50 --batch_size 1 --lr 1e-4 --num_workers 4"

run_a0() {
    echo "=== A0: Minimal CFM baseline (linear, no extras) ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a0 \
        --run_name ablation_a0_linear_baseline \
        --interpolant linear \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing \
        --no_ot_scaling \
        --no_tpg \
        --no_cross_attn
}

run_a1() {
    echo "=== A1: +Cosine interpolant ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a1 \
        --run_name ablation_a1_cosine \
        --interpolant cosine \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing \
        --no_ot_scaling \
        --no_tpg \
        --no_cross_attn
}

run_a2() {
    echo "=== A2: +Stochastic interpolant ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a2 \
        --run_name ablation_a2_stochastic \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing \
        --no_ot_scaling \
        --no_tpg \
        --no_cross_attn
}

run_a3() {
    echo "=== A3: +Temporal-Aware OT scaling ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a3 \
        --run_name ablation_a3_ot_scaling \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing \
        --no_tpg \
        --no_cross_attn
}

run_a4() {
    echo "=== A4: +Temporal Progression Gate (TPG) ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a4 \
        --run_name ablation_a4_tpg \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing \
        --no_cross_attn
}

run_a5() {
    echo "=== A5: +Cross-Attention Gate ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a5 \
        --run_name ablation_a5_cross_attn \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver euler \
        --n_steps 50 \
        --lambda_cons 0.0 \
        --no_time_annealing
}

run_a6() {
    echo "=== A6: +Heun solver ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a6 \
        --run_name ablation_a6_heun \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver heun \
        --n_steps 20 \
        --lambda_cons 0.0 \
        --no_time_annealing
}

run_a7() {
    echo "=== A7: +Adaptive time annealing ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a7 \
        --run_name ablation_a7_time_annealing \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver heun \
        --n_steps 20 \
        --lambda_cons 0.0
}

run_a8() {
    echo "=== A8: Full model (+Consistency loss) ==="
    python tasks/train_fm_model.py ${COMMON} \
        --output_dir ${CKPT_ROOT}/ablation_a8 \
        --run_name ablation_a8_full_model \
        --interpolant stochastic \
        --sigma_min 0.01 \
        --solver heun \
        --n_steps 20 \
        --lambda_cons 0.1
}

case "${1:-all}" in
    a0) run_a0 ;;
    a1) run_a1 ;;
    a2) run_a2 ;;
    a3) run_a3 ;;
    a4) run_a4 ;;
    a5) run_a5 ;;
    a6) run_a6 ;;
    a7) run_a7 ;;
    a8) run_a8 ;;
    all)
        run_a0
        run_a1
        run_a2
        run_a3
        run_a4
        run_a5
        run_a6
        run_a7
        run_a8
        ;;
    *)
        echo "Usage: $0 [a0|a1|a2|a3|a4|a5|a6|a7|a8|all]"
        exit 1
        ;;
esac
