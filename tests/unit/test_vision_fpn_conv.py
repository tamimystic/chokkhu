"""Unit tests for Feature Pyramid Networks (FPN, PANet) and Dilated Convolution."""

import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    AtrousConv2D,
    DilatedConv2D,
    FeaturePyramidNetwork,
    PANet,
)


def test_feature_pyramid_network():
    # Simulate multi-scale backbone feature maps: C2, C3, C4, C5
    N = 2
    c2 = Tensor(np.random.randn(N, 64, 32, 32))
    c3 = Tensor(np.random.randn(N, 128, 16, 16))
    c4 = Tensor(np.random.randn(N, 256, 8, 8))
    c5 = Tensor(np.random.randn(N, 512, 4, 4))

    fpn = FeaturePyramidNetwork(in_channels_list=[64, 128, 256, 512], out_channels=128)
    pyramids = fpn([c2, c3, c4, c5])

    assert len(pyramids) == 4
    assert pyramids[0].shape == (N, 128, 32, 32)
    assert pyramids[1].shape == (N, 128, 16, 16)
    assert pyramids[2].shape == (N, 128, 8, 8)
    assert pyramids[3].shape == (N, 128, 4, 4)


def test_panet_bottom_up_augmentation():
    N = 2
    c2 = Tensor(np.random.randn(N, 32, 16, 16))
    c3 = Tensor(np.random.randn(N, 64, 8, 8))
    c4 = Tensor(np.random.randn(N, 128, 4, 4))

    pan = PANet(in_channels_list=[32, 64, 128], out_channels=64)
    pan_features = pan([c2, c3, c4])

    assert len(pan_features) == 3
    assert pan_features[0].shape == (N, 64, 16, 16)
    assert pan_features[1].shape == (N, 64, 8, 8)
    assert pan_features[2].shape == (N, 64, 4, 4)


def test_dilated_convolution():
    # Dilated 3x3 with dilation=2 has effective receptive field 5x5
    dilated_conv = DilatedConv2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
    )
    x = Tensor(np.random.randn(2, 3, 16, 16))
    out = dilated_conv(x)
    # (16 + 2*2 - 5)/1 + 1 = 16
    assert out.shape == (2, 8, 16, 16)

    # Test alias AtrousConv2D
    assert AtrousConv2D is DilatedConv2D
