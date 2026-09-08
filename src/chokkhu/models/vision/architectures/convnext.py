from __future__ import annotations

from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, LayerNorm, Linear, Module
from ...dl.activations import GELU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D


class ConvNeXtBlock(Module):
    """ConvNeXt Block (7x7 depthwise conv, LayerNorm, 1x1 conv inverted bottleneck, GELU)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = Conv2D(dim, dim, kernel_size=7, padding=3)  # Depthwise
        self.norm = LayerNorm(dim)
        self.pwconv1 = Linear(dim, 4 * dim)  # Inverted bottleneck
        self.act = GELU()
        self.pwconv2 = Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        identity = x
        out = self.dwconv(x)
        # Permute to (N, H, W, C) for LayerNorm & Linear
        N, C, H, W = out.shape
        out_perm = out.data.transpose(0, 2, 3, 1).reshape(-1, C)
        out_t = Tensor(out_perm, requires_grad=out.requires_grad)
        out_t = self.norm(out_t)
        out_t = self.pwconv1(out_t)
        out_t = self.act(out_t)
        out_t = self.pwconv2(out_t)
        out_res = out_t.data.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return identity + Tensor(out_res, requires_grad=x.requires_grad)


def ConvNeXtTiny(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """ConvNeXt-Tiny Architecture (Liu et al., 2022)."""
    model = Sequential(task="classification")
    # Patchify stem (4x4 conv, stride 4)
    model.add(Conv2D(in_channels, 96, kernel_size=4, stride=4, padding=0))
    model.add(ConvNeXtBlock(96))
    model.add(ConvNeXtBlock(96))
    # Downsample
    model.add(Conv2D(96, 192, kernel_size=2, stride=2, padding=0))
    model.add(ConvNeXtBlock(192))
    model.add(ConvNeXtBlock(192))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(192, num_classes))
    return model


ConvNeXt = ConvNeXtTiny
