from __future__ import annotations

from chokkhu.core.tensor import Tensor
from ...dl.layers import Dropout, Flatten, Linear, Module
from ...dl.activations import SiLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D
from ..batchnorm2d import BatchNorm2D
from ..attention_blocks import SEBlock


class MBConv(Module):
    """Mobile Inverted Bottleneck Conv (MBConv) Block for EfficientNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 6,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        self.expand = (
            Conv2D(in_channels, hidden_dim, kernel_size=1, bias=False)
            if expand_ratio != 1
            else None
        )
        self.expand_bn = BatchNorm2D(hidden_dim) if expand_ratio != 1 else None

        self.dw = Conv2D(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.dw_bn = BatchNorm2D(hidden_dim)

        self.se = SEBlock(hidden_dim, reduction=4)

        self.project = Conv2D(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = BatchNorm2D(out_channels)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if self.expand is not None and self.expand_bn is not None:
            out = self.silu(self.expand_bn(self.expand(out)))
        out = self.silu(self.dw_bn(self.dw(out)))
        out = self.se(out)
        out = self.project_bn(self.project(out))
        if self.use_residual:
            return x + out
        return out


def EfficientNetB0(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """EfficientNet-B0 Architecture from Scratch."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False))
    model.add(BatchNorm2D(32))
    model.add(SiLU())

    model.add(MBConv(32, 16, expand_ratio=1, stride=1, kernel_size=3))
    model.add(MBConv(16, 24, expand_ratio=6, stride=2, kernel_size=3))
    model.add(MBConv(24, 40, expand_ratio=6, stride=2, kernel_size=5))
    model.add(MBConv(40, 80, expand_ratio=6, stride=2, kernel_size=3))

    model.add(GlobalAvgPool2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Linear(80, num_classes))
    return model
