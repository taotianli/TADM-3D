"""2D Flow Matching model — spatial_dims=2 version of FlowMatching3D."""
import torch
import torch.nn as nn

from unet.basic_unet_2d import BasicUNetEncoder2D
from flow_matching.flow_matching import ConditionalFlowMatching
from flow_matching.velocity_unet_2d import VelocityUNet2D


class FlowMatching2D(nn.Module):
    """
    Drop-in 2-D replacement for FlowMatching3D.

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
                 solver: str = "heun"):
        super().__init__()
        self.channels = channels

        self.embed_model = BasicUNetEncoder2D(channels, channels, feature)

        self.model = VelocityUNet2D(
            spatial_dims=2,
            in_channels=channels + channels,
            out_channels=channels,
            features=feature,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        self.fm = ConditionalFlowMatching(
            interpolant_type=interpolant_type,
            sigma_min=sigma_min,
            n_inference_steps=n_inference_steps,
            solver=solver,
        )

    def forward(self, image=None, x=None, metadata=None,
                pred_type=None, step=None, diff_ages=None):
        if pred_type == "get_train_sample":
            x_t, t, v_t, _ = self.fm.get_train_sample(x, diff_ages=diff_ages)
            return x_t, t, v_t

        elif pred_type == "predict_velocity":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image,
                              embeddings=embeddings, metadata=metadata)

        elif pred_type == "fm_sample":
            embeddings = self.embed_model(image)

            def velocity_fn(x_t, t_tensor):
                return self.model(x_t, t=t_tensor, image=image,
                                  embeddings=embeddings, metadata=metadata)

            shape = (image.shape[0], self.channels, *image.shape[2:])
            return self.fm.sample(velocity_fn, shape, image.device,
                                  diff_ages=diff_ages)

        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")
