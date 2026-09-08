from __future__ import annotations

from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU, SiLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, DepthwiseSeparableConv2D, GlobalAvgPool2D
from ..batchnorm2d import BatchNorm2D
from ..attention_blocks import SEBlock


def MobileNetV1(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """MobileNetV1 using Depthwise Separable Convolutions."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(32))
    model.add(ReLU())

    model.add(DepthwiseSeparableConv2D(32, 64, stride=1))
    model.add(BatchNorm2D(64))
    model.add(ReLU())

    model.add(DepthwiseSeparableConv2D(64, 128, stride=2))
    model.add(BatchNorm2D(128))
    model.add(ReLU())

    model.add(DepthwiseSeparableConv2D(128, 256, stride=2))
    model.add(BatchNorm2D(256))
    model.add(ReLU())

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(256, num_classes))
    return model


class InvertedResidual(Module):
    """MobileNetV2 Inverted Residual Block with Linear Bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        self.expand_conv = (
            Conv2D(in_channels, hidden_dim, kernel_size=1, bias=False)
            if expand_ratio != 1
            else None
        )
        self.expand_bn = BatchNorm2D(hidden_dim) if expand_ratio != 1 else None

        self.dw_conv = Conv2D(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.dw_bn = BatchNorm2D(hidden_dim)

        self.project_conv = Conv2D(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = BatchNorm2D(out_channels)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if self.expand_conv is not None and self.expand_bn is not None:
            out = self.relu(self.expand_bn(self.expand_conv(out)))
        out = self.relu(self.dw_bn(self.dw_conv(out)))
        out = self.project_bn(self.project_conv(out))  # Linear bottleneck
        if self.use_residual:
            return x + out
        return out


def MobileNetV2(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """MobileNetV2 Architecture (Sandler et al., 2018)."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(32))
    model.add(ReLU())

    model.add(InvertedResidual(32, 16, stride=1, expand_ratio=1))
    model.add(InvertedResidual(16, 24, stride=2, expand_ratio=6))
    model.add(InvertedResidual(24, 32, stride=2, expand_ratio=6))
    model.add(InvertedResidual(32, 64, stride=2, expand_ratio=6))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(64, num_classes))
    return model


def MobileNetV3(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """MobileNetV3 Architecture with SE Attention and Hard-Swish."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 16, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(16))
    model.add(SiLU())

    model.add(InvertedResidual(16, 24, stride=2, expand_ratio=4))
    model.add(SEBlock(24))
    model.add(InvertedResidual(24, 40, stride=2, expand_ratio=4))
    model.add(SEBlock(40))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(40, num_classes))
    return model
