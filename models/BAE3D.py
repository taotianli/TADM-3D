from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from unet.basic_unet import BasicUNetEncoder
import pdb
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class BAE3D(nn.Module):
    def __init__(self, channels=1, feature=[64, 64, 128, 256, 512, 64]):
        super().__init__()
        
        self.channels = channels
        self.embed_model = BasicUNetEncoder(3, channels, channels, feature)
        self.projection = nn.Sequential(nn.Conv3d(64, 8, 1))
        self.pred_layer = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512,1))

        # age embedding
        self.aemb = nn.Module()
        self.aemb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        # condition embedding
        self.cemb = nn.Module()
        self.cemb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ]) 

    def forward(self, lr, hr, age, condition):
        embeddings_lr = self.embed_model(lr)
        embeddings_lr = self.projection(embeddings_lr[-1])

        embeddings_hr = self.embed_model(hr)
        embeddings_hr = self.projection(embeddings_hr[-1])
        
        aemb = get_timestep_embedding(age, 128)
        aemb = self.aemb.dense[0](aemb)
        aemb = nonlinearity(aemb)
        aemb = self.aemb.dense[1](aemb)
        
        cemb = get_timestep_embedding(condition, 128)
        cemb = self.cemb.dense[0](cemb)
        cemb = nonlinearity(cemb)
        cemb = self.cemb.dense[1](cemb)

        embeddings = torch.cat([embeddings_lr.flatten(1), embeddings_hr.flatten(1), aemb, cemb], dim=1)
        
        diff_pred = self.pred_layer(embeddings)

        return diff_pred
