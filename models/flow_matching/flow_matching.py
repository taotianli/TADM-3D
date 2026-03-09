"""
Conditional Flow Matching for 3D Brain MRI Progression.

Implements Optimal Transport Conditional Flow Matching (OT-CFM) as described in:
  Lipman et al. "Flow Matching for Generative Modeling" (ICLR 2023)
  Tong et al. "Improving and Generalizing Flow-Matching" (NeurIPS 2023)

Key innovations over vanilla CFM:
  1. Stochastic Interpolant with learnable sigma_min (SigmaFM)
  2. Adaptive time sampling biased toward clinically-relevant intervals
  3. Temporal-aware OT path: interpolant conditioned on age-gap magnitude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Interpolant schedules
# ---------------------------------------------------------------------------

class LinearInterpolant:
    """
    Straight-line (OT) interpolant:
        x_t = (1 - t) * x_0 + t * x_1
        v_t = x_1 - x_0   (constant velocity)
    """
    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - t) * x0 + t * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        return x1 - x0


class CosineInterpolant:
    """
    Cosine interpolant (smoother boundary behaviour):
        x_t = cos(pi/2 * t)^2 * x_0 + sin(pi/2 * t)^2 * x_1
    """
    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.ndim - 1)))
        w0 = torch.cos(math.pi / 2 * t) ** 2
        w1 = torch.sin(math.pi / 2 * t) ** 2
        return w0 * x0 + w1 * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.ndim - 1)))
        # d/dt [cos^2(pi/2 t) * x0 + sin^2(pi/2 t) * x1]
        dw0 = -math.pi * torch.cos(math.pi / 2 * t) * torch.sin(math.pi / 2 * t)
        dw1 =  math.pi * torch.sin(math.pi / 2 * t) * torch.cos(math.pi / 2 * t)
        return dw0 * x0 + dw1 * x1


class MinibatchOTInterpolant:
    """
    Minibatch Optimal Transport CFM (OT-CFM) from Tong et al. NeurIPS 2023.

    Instead of pairing each x_0 with an independent x_1, we solve a
    minibatch OT problem to find the assignment that minimises total
    transport cost within the batch.  This reduces variance in the
    CFM training objective and produces straighter flow trajectories.

    For batch_size=1 (our 3D case) this degenerates to standard linear
    CFM, so the benefit only appears when batch_size > 1.  We include it
    here for the 2D ablation where larger batches are feasible.
    """
    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def _ot_permutation(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Greedy minibatch OT: find permutation of x0 that minimises
        sum of squared L2 distances to x1.
        Returns permuted x0.
        """
        B = x0.shape[0]
        if B == 1:
            return x0
        # Cost matrix: (B, B) pairwise squared L2
        x0_flat = x0.view(B, -1)
        x1_flat = x1.view(B, -1)
        cost = torch.cdist(x0_flat, x1_flat, p=2) ** 2  # (B, B)
        # Greedy assignment (Hungarian is O(B^3), greedy is O(B^2))
        used = torch.zeros(B, dtype=torch.bool, device=x0.device)
        perm = torch.zeros(B, dtype=torch.long, device=x0.device)
        for i in range(B):
            row = cost[i].clone()
            row[used] = float('inf')
            j = row.argmin()
            perm[i] = j
            used[j] = True
        return x0[perm]

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        x0 = self._ot_permutation(x0, x1)
        t_ = t.view(-1, *([1] * (x0.ndim - 1)))
        mu_t = (1.0 - t_) * x0 + t_ * x1
        if self.sigma_min > 0:
            return mu_t + self.sigma_min * torch.randn_like(x0)
        return mu_t

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        x0 = self._ot_permutation(x0, x1)
        return x1 - x0


class StochasticInterpolant:
    """
    Stochastic interpolant with learnable sigma_min (SigmaFM):
        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 + sigma_t * eps
        sigma_t = sigma_min * sqrt(t * (1 - t))   (bridge noise)

    This adds stochasticity along the path, improving sample diversity
    and robustness to out-of-distribution inputs.
    """
    def __init__(self, sigma_min: float = 0.01):
        self.sigma_min = sigma_min

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor,
                    noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_ = t.view(-1, *([1] * (x0.ndim - 1)))
        mu_t = (1.0 - (1.0 - self.sigma_min) * t_) * x0 + t_ * x1
        sigma_t = self.sigma_min * torch.sqrt(t_ * (1.0 - t_) + 1e-8)
        if noise is None:
            noise = torch.randn_like(x0)
        return mu_t + sigma_t * noise

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor,
                        t: torch.Tensor,
                        noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Conditional velocity field u_t(x | x_0, x_1)."""
        t_ = t.view(-1, *([1] * (x0.ndim - 1)))
        # d/dt mu_t
        dmu = x1 - (1.0 - self.sigma_min) * x0
        # d/dt sigma_t * eps
        dsigma = self.sigma_min * (1.0 - 2.0 * t_) / (
            2.0 * torch.sqrt(t_ * (1.0 - t_) + 1e-8))
        if noise is None:
            noise = torch.randn_like(x0)
        return dmu + dsigma * noise


# ---------------------------------------------------------------------------
# Adaptive time sampler
# ---------------------------------------------------------------------------

class AdaptiveTimeSampler:
    """
    Innovation 1 — Adaptive Time Sampling.

    Standard CFM uses Uniform[0,1] for t.  For brain aging, the model
    needs to be especially accurate near t=0 (early progression) and t=1
    (final state).  We use a Beta(alpha, beta) distribution that can be
    annealed during training:

        - Early training: Beta(1,1) = Uniform (broad exploration)
        - Late training:  Beta(0.5, 0.5) (U-shaped, emphasises boundaries)

    Additionally, we optionally bias sampling toward the normalised
    age-gap magnitude so that longer follow-up intervals receive more
    training signal.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta  = beta

    def sample(self, batch_size: int, device: torch.device,
               diff_ages: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = torch.distributions.Beta(
            torch.tensor(self.alpha, device=device),
            torch.tensor(self.beta,  device=device))
        t = dist.sample((batch_size,))

        if diff_ages is not None:
            # Soft bias: longer intervals → sample t closer to 1
            # diff_ages normalised to [0,1] by dividing by 10 years
            bias = (diff_ages.float().to(device) / 10.0).clamp(0, 1)
            t = t * (1.0 - 0.3 * bias) + 0.3 * bias * torch.rand_like(t)
            t = t.clamp(1e-4, 1.0 - 1e-4)

        return t

    def anneal(self, epoch: int, max_epochs: int):
        """Gradually shift from Uniform to U-shaped Beta."""
        progress = epoch / max_epochs
        self.alpha = 1.0 - 0.5 * progress   # 1.0 → 0.5
        self.beta  = 1.0 - 0.5 * progress   # 1.0 → 0.5


# ---------------------------------------------------------------------------
# ODE solver for inference
# ---------------------------------------------------------------------------

class EulerSolver:
    """Simple Euler ODE solver for inference."""
    def __init__(self, n_steps: int = 50):
        self.n_steps = n_steps

    @torch.no_grad()
    def solve(self, velocity_fn, x0: torch.Tensor,
              t_start: float = 0.0, t_end: float = 1.0) -> torch.Tensor:
        dt = (t_end - t_start) / self.n_steps
        x  = x0.clone()
        for i in range(self.n_steps):
            t_val = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t_val,
                                  device=x.device, dtype=x.dtype)
            v = velocity_fn(x, t_tensor)
            x = x + dt * v
        return x


class HeunSolver:
    """
    Heun (2nd-order Runge-Kutta) solver — better accuracy with same NFE budget.
    Uses predictor-corrector: one extra function evaluation per step.
    """
    def __init__(self, n_steps: int = 20):
        self.n_steps = n_steps

    @torch.no_grad()
    def solve(self, velocity_fn, x0: torch.Tensor,
              t_start: float = 0.0, t_end: float = 1.0) -> torch.Tensor:
        dt = (t_end - t_start) / self.n_steps
        x  = x0.clone()
        for i in range(self.n_steps):
            t_val = t_start + i * dt
            t_next = t_val + dt
            t_cur  = torch.full((x.shape[0],), t_val,  device=x.device, dtype=x.dtype)
            t_nxt  = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
            v1 = velocity_fn(x, t_cur)
            x_pred = x + dt * v1
            v2 = velocity_fn(x_pred, t_nxt)
            x = x + dt * 0.5 * (v1 + v2)
        return x


# ---------------------------------------------------------------------------
# Core Flow Matching module
# ---------------------------------------------------------------------------

class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching wrapper.

    Replaces the DDPM/DDIM diffusion process with a continuous normalising
    flow trained via the CFM objective:

        L_CFM = E_{t, x_0, x_1} [ || v_theta(x_t, t) - u_t(x_t|x_0,x_1) ||^2 ]

    where:
        x_0 ~ N(0, I)  (source noise)
        x_1 = target residual (img_hr - img_lr)
        x_t = interpolant at time t
        u_t = conditional velocity (closed-form)

    Innovation 2 — Temporal-Aware OT Path:
        The interpolant is conditioned on the normalised age-gap so that
        the straight-line path is "stretched" proportionally to the
        expected magnitude of brain change.  Concretely, we scale x_0 by
        a factor derived from diff_ages before interpolation, encouraging
        the model to learn larger residuals for longer follow-ups.
    """

    def __init__(self,
                 interpolant_type: str = "stochastic",
                 sigma_min: float = 0.01,
                 n_inference_steps: int = 20,
                 solver: str = "heun",
                 use_ot_scaling: bool = True):
        super().__init__()
        self.use_ot_scaling = use_ot_scaling

        if interpolant_type == "linear":
            self.interpolant = LinearInterpolant()
        elif interpolant_type == "cosine":
            self.interpolant = CosineInterpolant()
        elif interpolant_type == "ot":
            self.interpolant = MinibatchOTInterpolant(sigma_min=sigma_min)
        else:  # stochastic (default)
            self.interpolant = StochasticInterpolant(sigma_min=sigma_min)

        self.time_sampler = AdaptiveTimeSampler()

        if solver == "heun":
            self.solver = HeunSolver(n_steps=n_inference_steps)
        else:
            self.solver = EulerSolver(n_steps=n_inference_steps)

    def get_train_sample(self, x1: torch.Tensor,
                         diff_ages: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor]:
        """
        Sample (x_t, t, target_velocity, x0) for training.

        Innovation 2: scale source noise by age-gap magnitude so that
        the OT path length correlates with expected progression.
        """
        B = x1.shape[0]
        device = x1.device

        # Sample source: Gaussian noise, optionally scaled by age-gap
        x0 = torch.randn_like(x1)
        if self.use_ot_scaling and diff_ages is not None:
            # Normalise diff_ages (years) to [0.5, 1.5] scale factor
            scale = 1.0 + 0.5 * (diff_ages.float().to(device) / 10.0).clamp(0, 1)
            scale = scale.view(-1, *([1] * (x1.ndim - 1)))
            x0 = x0 * scale

        # Sample time
        t = self.time_sampler.sample(B, device, diff_ages)

        # Interpolate — share the same noise sample so x_t and v_t are consistent
        if isinstance(self.interpolant, StochasticInterpolant):
            noise = torch.randn_like(x1)
            x_t = self.interpolant.interpolate(x0, x1, t, noise)
            v_t = self.interpolant.target_velocity(x0, x1, t, noise)
        else:
            x_t = self.interpolant.interpolate(x0, x1, t)
            v_t = self.interpolant.target_velocity(x0, x1, t)

        return x_t, t, v_t, x0

    def sample(self, velocity_fn, shape: tuple,
               device: torch.device,
               diff_ages: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate a sample by integrating the learned velocity field."""
        x0 = torch.randn(shape, device=device)
        if diff_ages is not None:
            scale = 1.0 + 0.5 * (diff_ages.float().to(device) / 10.0).clamp(0, 1)
            scale = scale.view(-1, *([1] * (len(shape) - 1)))
            x0 = x0 * scale
        return self.solver.solve(velocity_fn, x0)
