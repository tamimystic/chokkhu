import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool2D,
    BatchNorm2D,
    DepthwiseSeparableConv2D,
    ConvTranspose2D,
)


def test_conv2d_forward_backward():
    np.random.seed(42)
    N, C, H, W = 2, 3, 8, 8
    x = Tensor(np.random.randn(N, C, H, W), requires_grad=True)

    conv = Conv2D(
        in_channels=3, out_channels=4, kernel_size=3, padding="same", stride=1
    )
    out = conv(x)

    assert out.shape == (2, 4, 8, 8)
    loss = out.sum()
    loss.backward()

    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 8, 8)


def test_pooling_and_batchnorm2d():
    N, C, H, W = 2, 4, 8, 8
    x = Tensor(np.random.randn(N, C, H, W), requires_grad=True)

    maxpool = MaxPool2D(pool_size=2, stride=2)
    out_max = maxpool(x)
    assert out_max.shape == (2, 4, 4, 4)
    out_max.sum().backward()
    assert x.grad is not None

    x.zero_grad()
    avgpool = AvgPool2D(pool_size=2, stride=2)
    out_avg = avgpool(x)
    assert out_avg.shape == (2, 4, 4, 4)
    out_avg.sum().backward()
    assert x.grad is not None

    x.zero_grad()
    gap = GlobalAvgPool2D()
    out_gap = gap(x)
    assert out_gap.shape == (2, 4)

    bn = BatchNorm2D(num_features=4)
    out_bn = bn(x)
    assert out_bn.shape == (2, 4, 8, 8)


def test_depthwise_and_transpose_conv():
    x = Tensor(np.random.randn(2, 4, 8, 8), requires_grad=True)

    dw = DepthwiseSeparableConv2D(
        in_channels=4, out_channels=8, kernel_size=3, padding=1
    )
    out_dw = dw(x)
    assert out_dw.shape == (2, 8, 8, 8)

    upconv = ConvTranspose2D(in_channels=4, out_channels=2, kernel_size=2, stride=2)
    out_up = upconv(x)
    assert out_up.shape == (2, 2, 16, 16)
