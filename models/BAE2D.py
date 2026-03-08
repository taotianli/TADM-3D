"""2D Brain Age Estimator — spatial_dims=2 version of BAE3D."""
import math
import torch
import torch.nn as nn
from unet.basic_unet_2d import BasicUNetEncoder2D


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


class BAE2D(nn.Module):
    """2-D Brain Age Estimator.

    Inputs are 2-D slices (B, C, H, W).
    The projection reduces the bottleneck to 8 channels; the flattened
    spatial size depends on the input resolution.  For 128×128 inputs the
    bottleneck is 4×4 → 8*4*4 = 128 features per branch.
    """

    def __init__(self, channels: int = 1,
                 feature=(64, 64, 128, 256, 512, 64),
                 pred_input_dim: int = 1280):
        # For 128x128 input: bottleneck is 4x4, projection to 8ch → 8*4*4=128 per branch
        # Two branches (lr+hr) = 256, plus aemb(512) + cemb(512) = 1280
        super().__init__()
        self.channels    = channels
        self.embed_model = BasicUNetEncoder2D(channels, channels, feature)
        self.projection  = nn.Conv2d(64, 8, 1)
        self.pred_layer  = nn.Sequential(
            nn.Linear(pred_input_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        def make_emb():
            m = nn.Module()
            m.dense = nn.ModuleList([nn.Linear(128, 512), nn.Linear(512, 512)])
            return m

        self.aemb = make_emb()
        self.cemb = make_emb()

    def forward(self, lr, hr, age, condition):
        emb_lr = self.projection(self.embed_model(lr)[-1])
        emb_hr = self.projection(self.embed_model(hr)[-1])

        aemb = get_timestep_embedding(age, 128)
        aemb = nonlinearity(self.aemb.dense[0](aemb))
        aemb = self.aemb.dense[1](aemb)

        cemb = get_timestep_embedding(condition, 128)
        cemb = nonlinearity(self.cemb.dense[0](cemb))
        cemb = self.cemb.dense[1](cemb)

        embeddings = torch.cat(
            [emb_lr.flatten(1), emb_hr.flatten(1), aemb, cemb], dim=1
        )
        return self.pred_layer(embeddings)
