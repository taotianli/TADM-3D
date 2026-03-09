# Flow Matching Ablation Study Plan

## Critical Bug Fixes Applied

### 1. StochasticInterpolant Noise Consistency Bug (FIXED)
**Problem**: `get_train_sample()` was calling `interpolate()` and `target_velocity()` with separate noise samples, making x_t and v_t inconsistent.

**Fix**: Generate noise once and pass to both functions.

**Expected Impact**: Major improvement in training stability and convergence.

---

## Ablation Experiments

### Baseline: A0 - Minimal CFM
- **Interpolant**: Linear (vanilla OT-CFM)
- **Time Sampling**: Uniform (no annealing)
- **Consistency Loss**: OFF
- **TPG**: OFF
- **CrossAttention**: OFF
- **Temporal-Aware OT**: OFF (no x0 scaling)
- **Solver**: Euler (50 steps)

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a0 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a0_linear_baseline \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant linear \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing \
  --no_ot_scaling \
  --no_tpg \
  --no_cross_attn
```

---

### A1 - Add Cosine Interpolant
- **Change**: Linear → Cosine interpolant
- **Hypothesis**: Smoother boundary behavior improves convergence

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a1 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a1_cosine \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant cosine \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing \
  --no_ot_scaling \
  --no_tpg \
  --no_cross_attn
```

---

### A2 - Add Stochastic Interpolant
- **Change**: Cosine → Stochastic (sigma_min=0.01)
- **Hypothesis**: Stochasticity improves robustness

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a2 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a2_stochastic \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing \
  --no_ot_scaling \
  --no_tpg \
  --no_cross_attn
```

---

### A3 - Add Temporal-Aware OT Scaling
- **Change**: Enable x0 scaling by age-gap
- **Hypothesis**: Scaling source noise by expected progression magnitude helps

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a3 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a3_ot_scaling \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing \
  --no_tpg \
  --no_cross_attn
```

---

### A4 - Add Temporal Progression Gate (TPG)
- **Change**: Enable TPG in decoder
- **Hypothesis**: Age-gap conditioned gating helps modulate features

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a4 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a4_tpg \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing \
  --no_cross_attn
```

---

### A5 - Add Cross-Attention Gate
- **Change**: Enable cross-attention at bottleneck
- **Hypothesis**: Explicit attention to context features improves spatial accuracy

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a5 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a5_cross_attn \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver euler \
  --n_steps 50 \
  --lambda_cons 0.0 \
  --no_time_annealing
```

---

### A6 - Add Heun Solver
- **Change**: Euler → Heun (2nd-order RK)
- **Hypothesis**: Better ODE accuracy with same NFE budget

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a6 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a6_heun \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver heun \
  --n_steps 20 \
  --lambda_cons 0.0 \
  --no_time_annealing
```

---

### A7 - Add Adaptive Time Annealing
- **Change**: Enable Beta distribution annealing
- **Hypothesis**: Emphasizing boundaries later in training helps

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a7 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a7_time_annealing \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver heun \
  --n_steps 20 \
  --lambda_cons 0.0
```

---

### A8 - Full Model (All Innovations)
- **Change**: Enable consistency loss (lambda=0.1)
- **Hypothesis**: Consistency regularization improves sample quality

**Command**:
```bash
python tasks/train_fm_model.py \
  --dataset /path/to/dataset/ \
  --cache_dir /path/to/cache/ \
  --output_dir ./checkpoints/ablation_a8 \
  --bae_ckpt ./checkpoints/bae-best.pth \
  --run_name ablation_a8_full_model \
  --n_epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --interpolant stochastic \
  --sigma_min 0.01 \
  --solver heun \
  --n_steps 20 \
  --lambda_cons 0.1
```

---

## Additional Experiments

### E1 - Consistency Loss Only (Linear)
- Test if consistency loss works with linear interpolant
- **Command**: A0 + `--lambda_cons 0.1`

### E2 - Higher Sigma Min
- Test stochastic interpolant with sigma_min=0.05
- **Command**: A2 + `--sigma_min 0.05`

### E3 - More ODE Steps
- Test if more steps help (50 steps with Heun)
- **Command**: A6 + `--n_steps 50`

### E4 - Reverse Annealing Direction
- Anneal Beta(1,1) → Beta(2,2) (bell-shaped, emphasize middle)
- Requires code change in `AdaptiveTimeSampler.anneal()`

---

## Evaluation Metrics

For each experiment, compute on validation set:
1. **MSE** (Mean Squared Error)
2. **PSNR** (Peak Signal-to-Noise Ratio)
3. **SSIM** (Structural Similarity Index)
4. **Training Loss Curve** (CFM loss, BAE loss, Consistency loss)
5. **Inference Time** (ODE integration time)

---

## Expected Outcomes

1. **A0 → A1**: Small improvement from smoother boundaries
2. **A1 → A2**: Moderate improvement from stochasticity
3. **A2 → A3**: Unclear (OT scaling may hurt or help)
4. **A3 → A4**: Small improvement from TPG
5. **A4 → A5**: Moderate improvement from cross-attention
6. **A5 → A6**: Moderate improvement from better ODE solver
7. **A6 → A7**: Unclear (annealing may hurt early training)
8. **A7 → A8**: Small improvement from consistency loss

**Critical Fix Impact**: The noise consistency bug fix should provide the largest single improvement across all experiments.

---

## Implementation Status

- [x] Fix StochasticInterpolant noise bug
- [ ] Add command-line flags: --no_time_annealing, --no_ot_scaling, --no_tpg, --no_cross_attn, --sigma_min
- [ ] Modify VelocityUNet to support --no_tpg and --no_cross_attn
- [ ] Create ablation run scripts
- [ ] Run all experiments
- [ ] Analyze results and create comparison table
