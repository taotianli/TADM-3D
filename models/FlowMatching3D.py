"""
TAFM-3D: Temporally-Aware Flow Matching for 3D Brain MRI Progression.

Replaces the DDPM diffusion process in TADM-3D with Conditional Flow Matching.

Innovations over TADM-3D:
  1. Adaptive Time Sampling (Beta distribution, age-gap biased)
  2. Temporal-Aware OT Path (source noise scaled by age-gap)
  3. Cross-Attention Context Gate at bottleneck
  4. Temporal Progression Gate (TPG) in decoder
  5. Consistency Regularization Loss (Innovation 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.basic_unet import BasicUNetEncoder
from flow_matching.flow_matching import ConditionalFlowMatching
from flow_matching.velocity_unet import VelocityUNet


class FlowMatching3D(nn.Module):
    """
    Drop-in replacement for Diffusion3D using Conditional Flow Matching.

    API is intentionally compatible with Diffusion3D so that training/test
    scripts require minimal changes.

    pred_type options:
        "get_train_sample"  — sample (x_t, t, v_t) for training
        "predict_velocity"  — forward pass of velocity UNet
        "fm_sample"         — full ODE integration for inference
    """

    def __init__(self, channels: int = 1,
                 feature=(64, 64, 128, 256, 512, 64),
                 interpolant_type: str = "stochastic",
                 sigma_min: float = 0.01,
                 n_inference_steps: int = 20,
                 solver: str = "heun",
                 use_tpg: bool = True,
                 use_cross_attn: bool = True,
                 use_ot_scaling: bool = True):
        super().__init__()
        self.channels = channels
        self.use_ot_scaling = use_ot_scaling

        # Context encoder (same as TADM-3D)
        self.embed_model = BasicUNetEncoder(3, channels, channels, feature)

        # Velocity network (replaces denoising UNet)
        self.model = VelocityUNet(
            3, channels + channels, channels, feature,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
            use_tpg=use_tpg,
            use_cross_attn=use_cross_attn,
        )

        # Flow matching process
        self.fm = ConditionalFlowMatching(
            interpolant_type=interpolant_type,
            sigma_min=sigma_min,
            n_inference_steps=n_inference_steps,
            solver=solver,
            use_ot_scaling=use_ot_scaling,
        )

    def forward(self, image=None, x=None, metadata=None,
                pred_type=None, step=None, diff_ages=None):
        """
        Args:
            image:      baseline MRI context  (B, C, D, H, W)
            x:          target residual x_1   (B, C, D, H, W)  [training]
            metadata:   (B, 3) — [age, diff_age, condition]
            pred_type:  one of "get_train_sample", "predict_velocity", "fm_sample"
            step:       continuous time t     (B,)              [predict_velocity]
            diff_ages:  age gap in months     (B,)              [optional bias]
        """
        if pred_type == "get_train_sample":
            # Returns (x_t, t, v_t) for CFM training loss
            x_t, t, v_t, _ = self.fm.get_train_sample(x, diff_ages=diff_ages)
            return x_t, t, v_t

        elif pred_type == "predict_velocity":
            # Predict velocity field given x_t and t
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image,
                              embeddings=embeddings, metadata=metadata)

        elif pred_type == "fm_sample":
            # Full ODE integration: noise → residual
            embeddings = self.embed_model(image)

            def velocity_fn(x_t, t_tensor):
                return self.model(x_t, t=t_tensor, image=image,
                                  embeddings=embeddings, metadata=metadata)

            shape = (image.shape[0], self.channels, *image.shape[2:])
            return self.fm.sample(velocity_fn, shape, image.device,
                                  diff_ages=diff_ages)

        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")
