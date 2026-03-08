"""
2D Velocity UNet for Flow Matching — spatial_dims=2 version of VelocityUNet.
"""

from typing import Optional, Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep


def get_sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    assert values.ndim == 1
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=values.device) / (half - 1)
    )
    emb = values.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class TwoConvVel2D(nn.Sequential):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias,
                 dropout=0.0, temb_dim=2048):
        super().__init__()
        self.temb_proj = nn.Linear(temb_dim, out_chns)
        conv_0 = Convolution(spatial_dims, in_chns, out_chns,
                             act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns,
                             act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.conv_0(x)
        x = x + self.temb_proj(swish(temb))[:, :, None, None]
        x = self.conv_1(x)
        return x


class DownVel2D(nn.Module):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias,
                 dropout=0.0, temb_dim=2048):
        super().__init__()
        self.pool  = Pool["MAX", spatial_dims](kernel_size=2)
        self.convs = TwoConvVel2D(spatial_dims, in_chns, out_chns, act, norm, bias, dropout, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.convs(self.pool(x), temb)


class UpCatVel2D(nn.Module):
    def __init__(self, spatial_dims, in_chns, cat_chns, out_chns, act, norm, bias,
                 dropout=0.0, upsample="deconv", halves=True, temb_dim=2048):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(spatial_dims, in_chns, up_chns, 2,
                                 mode=upsample, pre_conv="default",
                                 interp_mode="linear", align_corners=True)
        self.convs = TwoConvVel2D(spatial_dims, cat_chns + up_chns, out_chns,
                                  act, norm, bias, dropout, temb_dim)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor],
                temb: torch.Tensor) -> torch.Tensor:
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


class CrossAttentionGate2D(nn.Module):
    """Cross-attention between noisy latent and context encoder at bottleneck (2D)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm_q  = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)
        self.attn    = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj    = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q  = self.norm_q(x).view(B, C, -1).permute(0, 2, 1)
        kv = self.norm_kv(ctx).view(B, C, -1).permute(0, 2, 1)
        out, _ = self.attn(q, kv, kv)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)


class TemporalProgressionGate2D(nn.Module):
    """Channel-wise gating conditioned on age-gap embedding (2D)."""

    def __init__(self, channels: int, daemb_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(daemb_dim, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, daemb: torch.Tensor) -> torch.Tensor:
        g = self.gate(daemb).view(daemb.shape[0], -1, 1, 1)
        return x * g + x


class VelocityUNet2D(nn.Module):
    """2-D velocity UNet for flow matching."""

    def __init__(
        self,
        spatial_dims: int = 2,
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
        fea  = ensure_tuple_rep(features, 6)
        TEMB = 2048

        def make_emb():
            m = nn.Module()
            m.dense = nn.ModuleList([nn.Linear(128, 512), nn.Linear(512, 512)])
            return m

        self.temb  = make_emb()
        self.aemb  = make_emb()
        self.daemb = make_emb()
        self.cemb  = make_emb()

        self.conv_0 = TwoConvVel2D(spatial_dims, in_channels, fea[0], act, norm, bias, dropout, TEMB)
        self.down_1 = DownVel2D(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, TEMB)
        self.dsa1   = SqueezeAttentionBlock2D(fea[1], fea[1])
        self.down_2 = DownVel2D(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, TEMB)
        self.dsa2   = SqueezeAttentionBlock2D(fea[2], fea[2])
        self.down_3 = DownVel2D(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, TEMB)
        self.dsa3   = SqueezeAttentionBlock2D(fea[3], fea[3])
        self.down_4 = DownVel2D(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, TEMB)
        self.dsa4   = SqueezeAttentionBlock2D(fea[4], fea[4])

        self.cross_attn = CrossAttentionGate2D(fea[4], num_heads=4)

        self.upcat_4 = UpCatVel2D(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg4    = TemporalProgressionGate2D(fea[3])
        self.usa4    = SqueezeAttentionBlock2D(fea[3], fea[3])

        self.upcat_3 = UpCatVel2D(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg3    = TemporalProgressionGate2D(fea[2])
        self.usa3    = SqueezeAttentionBlock2D(fea[2], fea[2])

        self.upcat_2 = UpCatVel2D(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg2    = TemporalProgressionGate2D(fea[1])
        self.usa2    = SqueezeAttentionBlock2D(fea[1], fea[1])

        self.upcat_1 = UpCatVel2D(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample,
                                   halves=False, temb_dim=TEMB)
        self.tpg1    = TemporalProgressionGate2D(fea[0])
        self.usa1    = SqueezeAttentionBlock2D(fea[0], fea[0])

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def _embed(self, module, values: torch.Tensor) -> torch.Tensor:
        e = get_sinusoidal_embedding(values, 128)
        e = module.dense[0](e)
        e = swish(e)
        e = module.dense[1](e)
        return e

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                embeddings=None, image=None, metadata=None) -> torch.Tensor:
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

        if embeddings is not None:
            x4 = self.cross_attn(x4, embeddings[4])

        u4 = self.upcat_4(x4, x3, cond)
        u4 = self.tpg4(u4, daemb)
        u4 = self.usa4(u4)

        u3 = self.upcat_3(u4, x2, cond)
        u3 = self.tpg3(u3, daemb)
        u3 = self.usa3(u3)

        u2 = self.upcat_2(u3, x1, cond)
        u2 = self.tpg2(u2, daemb)
        u2 = self.usa2(u2)

        u1 = self.upcat_1(u2, x0, cond)
        u1 = self.tpg1(u1, daemb)
        u1 = self.usa1(u1)

        return self.final_conv(u1)
