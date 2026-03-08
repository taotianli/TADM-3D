"""2D Diffusion model — spatial_dims=2 version of Diffusion3D."""
import torch
import torch.nn as nn

from unet.basic_unet_denose_2d import BasicUNetDe2D
from unet.basic_unet_2d import BasicUNetEncoder2D
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


class Diffusion2D(nn.Module):
    def __init__(self, channels: int = 1,
                 feature=(64, 64, 128, 256, 512, 64)) -> None:
        super().__init__()
        self.channels    = channels
        self.embed_model = BasicUNetEncoder2D(channels, channels, feature)
        self.model       = BasicUNetDe2D(
            in_channels=channels + channels,
            out_channels=channels,
            features=feature,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [50]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, metadata=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image,
                              embeddings=embeddings, metadata=metadata)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model,
                (1, self.channels, image.shape[2], image.shape[3]),
                model_kwargs={"image": image, "embeddings": embeddings, "metadata": metadata},
            )
            return sample_out["pred_xstart"]

        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")
