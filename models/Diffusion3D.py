from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from unet.SegResNet import SegResNet
from unet.SegResNet_denose import SegResNetDe
from unet.SwinUNETR import SwinUNETR
from unet.SwinUNETR_denose import SwinUNETRDe

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import torch
import torch.nn as nn 
from guided_diffusion import unet
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from guided_diffusion.utils import staple
import SimpleITK as sitk
import numpy as np

class Diffusion3D(nn.Module):
    def __init__(self, channels=1, feature=[64, 64, 128, 256, 512, 64]) -> None:
        super().__init__()
        self.channels = channels
        self.embed_model = BasicUNetEncoder(3, channels, channels, feature)
        self.model = BasicUNetDe(3, channels + channels, channels, feature, 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
    
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, metadata=None, pred_type=None, step=None):
        #ADD NOISE
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        #DENOISE
        elif pred_type == "denoise":
            embeddings = self.embed_model(image)         
            
            return self.model(x, t=step, image=image, embeddings=embeddings, metadata=metadata)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)  
                       
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.channels, 128, 128, 128), model_kwargs={"image": image, "embeddings": embeddings, "metadata": metadata})
            sample_out = sample_out["pred_xstart"]
            return sample_out