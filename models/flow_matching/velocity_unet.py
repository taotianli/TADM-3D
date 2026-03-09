"""
Velocity UNet for Flow Matching.

Adapts BasicUNetDe to predict velocity fields v_theta(x_t, t, context)
instead of denoised images.  Key changes:
  - Output is a velocity field (same shape as input residual)
  - Time embedding represents continuous t in [0,1] rather than discrete step
  - Added cross-attention gating at bottleneck (Innovation 3)
  - Added Temporal Progression Gate (Innovation 4)
"""

from typing import Optional, Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous scalars (time, age, diff_age)."""
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


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TwoConvVel(nn.Sequential):
    """Two convolutions with time-conditioned residual injection."""

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
        x = x + self.temb_proj(swish(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x


class DownVel(nn.Module):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias,
                 dropout=0.0, temb_dim=2048):
        super().__init__()
        self.pool = Pool["MAX", spatial_dims](kernel_size=2)
        self.convs = TwoConvVel(spatial_dims, in_chns, out_chns, act, norm, bias,
                                dropout, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.convs(self.pool(x), temb)


class UpCatVel(nn.Module):
    def __init__(self, spatial_dims, in_chns, cat_chns, out_chns, act, norm, bias,
                 dropout=0.0, upsample="deconv", halves=True, temb_dim=2048):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(spatial_dims, in_chns, up_chns, 2,
                                 mode=upsample, pre_conv="default",
                                 interp_mode="linear", align_corners=True)
        self.convs = TwoConvVel(spatial_dims, cat_chns + up_chns, out_chns,
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


class conv3d_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm3d(ch_out), nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm3d(ch_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.avg_pool   = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv       = conv3d_block(ch_in, ch_out)
        self.conv_atten = conv3d_block(ch_in, ch_out)
        self.upsample   = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        y = self.upsample(y)
        return (y * x_res) + y


# ---------------------------------------------------------------------------
# Innovation 3 — Cross-Attention Context Gate (bottleneck)
# ---------------------------------------------------------------------------

class CrossAttentionGate(nn.Module):
    """
    Lightweight cross-attention between the noisy latent (query) and the
    context encoder features (key/value) at the bottleneck.

    This allows the velocity network to explicitly attend to the most
    relevant spatial regions of the baseline MRI when predicting the
    progression velocity, rather than relying solely on additive skip
    connections.

    Complexity: O(N^2) where N = spatial tokens at bottleneck (8^3 = 512).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm_q  = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)
        self.attn    = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj    = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        q  = self.norm_q(x).view(B, C, -1).permute(0, 2, 1)   # (B, N, C)
        kv = self.norm_kv(ctx).view(B, C, -1).permute(0, 2, 1)
        out, _ = self.attn(q, kv, kv)
        out = out.permute(0, 2, 1).view(B, C, D, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Innovation 4 — Temporal Progression Gate (TPG)
# ---------------------------------------------------------------------------

class TemporalProgressionGate(nn.Module):
    """
    Temporal Progression Gate (TPG).

    Learns a channel-wise gating mask conditioned on the age-gap embedding.
    This modulates feature maps so that channels encoding fast-changing
    anatomy (e.g., hippocampus) are amplified for longer follow-ups, while
    stable regions are suppressed.

    Gate: g = sigmoid(W * daemb + b)
    Output: x * g + x  (residual gating)
    """

    def __init__(self, channels: int, daemb_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(daemb_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, daemb: torch.Tensor) -> torch.Tensor:
        g = self.gate(daemb).view(daemb.shape[0], -1, 1, 1, 1)
        return x * g + x


# ---------------------------------------------------------------------------
# Velocity UNet
# ---------------------------------------------------------------------------

class VelocityUNet(nn.Module):
    """
    UNet that predicts the velocity field v_theta(x_t, t, context, metadata).

    Differences from BasicUNetDe:
      - Predicts velocity (not denoised image)
      - Continuous time t in [0,1] (not discrete step)
      - CrossAttentionGate at bottleneck (Innovation 3)
      - TemporalProgressionGate at each decoder level (Innovation 4)
    """

    @deprecated_arg(name="dimensions", new_name="spatial_dims", since="0.6",
                    msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 64, 128, 256, 512, 64),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        use_tpg: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        TEMB = 2048  # total conditioning dim
        self.use_tpg = use_tpg
        self.use_cross_attn = use_cross_attn

        # ---- Conditioning embeddings ----
        def make_emb():
            m = nn.Module()
            m.dense = nn.ModuleList([nn.Linear(128, 512), nn.Linear(512, 512)])
            return m

        self.temb  = make_emb()   # flow time t
        self.aemb  = make_emb()   # age
        self.daemb = make_emb()   # diff_age
        self.cemb  = make_emb()   # condition

        # ---- Encoder ----
        self.conv_0 = TwoConvVel(spatial_dims, in_channels, fea[0], act, norm, bias, dropout, TEMB)
        self.down_1 = DownVel(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, TEMB)
        self.dsa1   = SqueezeAttentionBlock(fea[1], fea[1])
        self.down_2 = DownVel(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, TEMB)
        self.dsa2   = SqueezeAttentionBlock(fea[2], fea[2])
        self.down_3 = DownVel(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, TEMB)
        self.dsa3   = SqueezeAttentionBlock(fea[3], fea[3])
        self.down_4 = DownVel(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, TEMB)
        self.dsa4   = SqueezeAttentionBlock(fea[4], fea[4])

        # ---- Bottleneck cross-attention (Innovation 3) ----
        self.cross_attn = CrossAttentionGate(fea[4], num_heads=4) if use_cross_attn else None

        # ---- Decoder ----
        self.upcat_4 = UpCatVel(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg4    = TemporalProgressionGate(fea[3]) if use_tpg else None
        self.usa4    = SqueezeAttentionBlock(fea[3], fea[3])

        self.upcat_3 = UpCatVel(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg3    = TemporalProgressionGate(fea[2]) if use_tpg else None
        self.usa3    = SqueezeAttentionBlock(fea[2], fea[2])

        self.upcat_2 = UpCatVel(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, temb_dim=TEMB)
        self.tpg2    = TemporalProgressionGate(fea[1]) if use_tpg else None
        self.usa2    = SqueezeAttentionBlock(fea[1], fea[1])

        self.upcat_1 = UpCatVel(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample,
                                halves=False, temb_dim=TEMB)
        self.tpg1    = TemporalProgressionGate(fea[0]) if use_tpg else None
        self.usa1    = SqueezeAttentionBlock(fea[0], fea[0])

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def _embed(self, module, values: torch.Tensor) -> torch.Tensor:
        e = get_sinusoidal_embedding(values, 128)
        e = module.dense[0](e)
        e = swish(e)
        e = module.dense[1](e)
        return e

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                embeddings=None, image=None, metadata=None) -> torch.Tensor:
        """
        Args:
            x:          (B, C, D, H, W) — interpolated x_t
            t:          (B,)            — continuous time in [0, 1]
            embeddings: list of context encoder features
            image:      (B, C, D, H, W) — baseline MRI context
            metadata:   (B, 3)          — [age, diff_age, condition]
        """
        # Build conditioning vector
        temb  = self._embed(self.temb,  t)
        aemb  = self._embed(self.aemb,  metadata[:, 0])
        daemb = self._embed(self.daemb, metadata[:, 1])
        cemb  = self._embed(self.cemb,  metadata[:, 2])
        cond  = torch.cat([temb, aemb, daemb, cemb], dim=1)  # (B, 2048)

        # Concatenate context image channel-wise
        if image is not None:
            x = torch.cat([image, x], dim=1)

        # Encoder
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

        # Bottleneck cross-attention (Innovation 3)
        if self.use_cross_attn and self.cross_attn is not None and embeddings is not None:
            x4 = self.cross_attn(x4, embeddings[4])

        # Decoder with optional TPG (Innovation 4)
        u4 = self.upcat_4(x4, x3, cond)
        if self.use_tpg and self.tpg4 is not None:
            u4 = self.tpg4(u4, daemb)
        u4 = self.usa4(u4)

        u3 = self.upcat_3(u4, x2, cond)
        if self.use_tpg and self.tpg3 is not None:
            u3 = self.tpg3(u3, daemb)
        u3 = self.usa3(u3)

        u2 = self.upcat_2(u3, x1, cond)
        if self.use_tpg and self.tpg2 is not None:
            u2 = self.tpg2(u2, daemb)
        u2 = self.usa2(u2)

        u1 = self.upcat_1(u2, x0, cond)
        if self.use_tpg and self.tpg1 is not None:
            u1 = self.tpg1(u1, daemb)
        u1 = self.usa1(u1)

        return self.final_conv(u1)
