"""2D version of BasicUNetEncoder — spatial_dims=2."""
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep


class TwoConv(nn.Sequential):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias, dropout=0.0):
        super().__init__()
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm,
                             dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm,
                             dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    def __init__(self, spatial_dims, in_chns, out_chns, act, norm, bias, dropout=0.0):
        super().__init__()
        self.add_module("max_pooling", Pool["MAX", spatial_dims](kernel_size=2))
        self.add_module("convs", TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout))


class BasicUNetEncoder2D(nn.Module):
    """2-D context encoder (same architecture as 3-D, spatial_dims=2)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 64, 128, 256, 512, 64),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        spatial_dims = 2
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5 = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x5 = self.down_5(x4)
        return [x0, x1, x2, x3, x4, x5]
