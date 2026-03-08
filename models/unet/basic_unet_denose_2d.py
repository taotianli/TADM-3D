"""2D denoising UNet — spatial_dims=2 version of BasicUNetDe."""
from typing import Optional, Sequence, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


class TwoConv(nn.Sequential):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias,
                 dropout=0.0, temb_dim=2048):
        super().__init__()
        self.temb_proj = nn.Linear(temb_dim, out_chns)
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm,
                             dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm,
                             dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv_1(x)
        return x


class Down(nn.Module):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias,
                 dropout=0.0, temb_dim=2048):
        super().__init__()
        self.max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        self.convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout, temb_dim)

    def forward(self, x, temb):
        return self.convs(self.max_pooling(x), temb)


class UpCat(nn.Module):
    def __init__(self, spatial_dims, in_chns, cat_chns, out_chns, act, norm, bias,
                 dropout=0.0, upsample="deconv", halves=True, temb_dim=2048):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(spatial_dims, in_chns, up_chns, 2,
                                 mode=upsample, pre_conv="default",
                                 interp_mode="linear", align_corners=True)
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns,
                             act, norm, bias, dropout, temb_dim)

    def forward(self, x, x_e, temb):
        x0 = self.upsample(x)
        if x_e is not None:
            dims = len(x.shape) - 2
            sp = [0] * (dims * 2)
            for i in range(dims):
                if x_e.shape[-i - 1] != x0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x0 = F.pad(x0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x0], dim=1), temb)
        else:
            x = self.convs(x0, temb)
        return x


class conv2d_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SqueezeAttentionBlock2D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.avg_pool   = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv       = conv2d_block(ch_in, ch_out)
        self.conv_atten = conv2d_block(ch_in, ch_out)
        self.upsample   = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        y = self.upsample(y)
        return (y * x_res) + y


class BasicUNetDe2D(nn.Module):
    """2-D denoising UNet for diffusion model."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Sequence[int] = (64, 64, 128, 256, 512, 64),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        spatial_dims = 2
        fea = ensure_tuple_rep(features, 6)
        TEMB = 2048

        def make_emb():
            m = nn.Module()
            m.dense = nn.ModuleList([nn.Linear(128, 512), nn.Linear(512, 512)])
            return m

        self.temb  = make_emb()
        self.aemb  = make_emb()
        self.daemb = make_emb()
        self.cemb  = make_emb()

        self.conv_0  = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout, TEMB)
        self.down_1  = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, TEMB)
        self.dsa1    = SqueezeAttentionBlock2D(fea[1], fea[1])
        self.down_2  = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, TEMB)
        self.dsa2    = SqueezeAttentionBlock2D(fea[2], fea[2])
        self.down_3  = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, TEMB)
        self.dsa3    = SqueezeAttentionBlock2D(fea[3], fea[3])
        self.down_4  = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, TEMB)
        self.dsa4    = SqueezeAttentionBlock2D(fea[4], fea[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.usa4    = SqueezeAttentionBlock2D(fea[3], fea[3])
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.usa3    = SqueezeAttentionBlock2D(fea[2], fea[2])
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.usa2    = SqueezeAttentionBlock2D(fea[1], fea[1])
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample,
                             halves=False, temb_dim=TEMB)
        self.usa1    = SqueezeAttentionBlock2D(fea[0], fea[0])

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def _embed(self, module, values):
        e = get_timestep_embedding(values, 128)
        e = module.dense[0](e)
        e = nonlinearity(e)
        e = module.dense[1](e)
        return e

    def forward(self, x, t, embeddings=None, image=None, metadata=None):
        temb  = self._embed(self.temb,  t)
        aemb  = self._embed(self.aemb,  metadata[:, 0])
        daemb = self._embed(self.daemb, metadata[:, 1])
        cemb  = self._embed(self.cemb,  metadata[:, 2])
        cond  = torch.cat([temb, aemb, daemb, cemb], dim=1)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, cond)
        if embeddings is not None:
            x0 = x0 + embeddings[0]

        x1 = self.down_1(x0, cond)
        if embeddings is not None:
            x1 = x1 + embeddings[1]
        x1 = self.dsa1(x1)

        x2 = self.down_2(x1, cond)
        if embeddings is not None:
            x2 = x2 + embeddings[2]
        x2 = self.dsa2(x2)

        x3 = self.down_3(x2, cond)
        if embeddings is not None:
            x3 = x3 + embeddings[3]
        x3 = self.dsa3(x3)

        x4 = self.down_4(x3, cond)
        if embeddings is not None:
            x4 = x4 + embeddings[4]
        x4 = self.dsa4(x4)

        u4 = self.upcat_4(x4, x3, cond)
        u4 = self.usa4(u4)
        u3 = self.upcat_3(u4, x2, cond)
        u3 = self.usa3(u3)
        u2 = self.upcat_2(u3, x1, cond)
        u2 = self.usa2(u2)
        u1 = self.upcat_1(u2, x0, cond)
        u1 = self.usa1(u1)

        return self.final_conv(u1)
