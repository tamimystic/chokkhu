from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Module
from ...dl.activations import ReLU
from ..conv_layers import Conv2D, ConvTranspose2D, MaxPool2D


class UNet(Module, ChokkhuModel):
    """U-Net Architecture for Image Segmentation from Scratch."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        # Encoder
        self.enc1_1 = Conv2D(in_channels, 32, kernel_size=3, padding=1)
        self.enc1_2 = Conv2D(32, 32, kernel_size=3, padding=1)
        self.pool1 = MaxPool2D(2, 2)

        self.enc2_1 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.enc2_2 = Conv2D(64, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2D(2, 2)

        # Bottleneck
        self.bottle1 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.bottle2 = Conv2D(128, 128, kernel_size=3, padding=1)

        # Decoder
        self.upconv2 = ConvTranspose2D(128, 64, kernel_size=2, stride=2)
        self.dec2_1 = Conv2D(128, 64, kernel_size=3, padding=1)
        self.dec2_2 = Conv2D(64, 64, kernel_size=3, padding=1)

        self.upconv1 = ConvTranspose2D(64, 32, kernel_size=2, stride=2)
        self.dec1_1 = Conv2D(64, 32, kernel_size=3, padding=1)
        self.dec1_2 = Conv2D(32, 32, kernel_size=3, padding=1)

        self.final_conv = Conv2D(32, out_channels, kernel_size=1, padding=0)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Encoder 1
        e1 = self.relu(self.enc1_1(x))
        e1 = self.relu(self.enc1_2(e1))
        p1 = self.pool1(e1)

        # Encoder 2
        e2 = self.relu(self.enc2_1(p1))
        e2 = self.relu(self.enc2_2(e2))
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.relu(self.bottle1(p2))
        b = self.relu(self.bottle2(b))

        # Decoder 2
        u2 = self.upconv2(b)
        cat2 = Tensor(
            np.concatenate([u2.data, e2.data], axis=1), requires_grad=u2.requires_grad
        )
        d2 = self.relu(self.dec2_1(cat2))
        d2 = self.relu(self.dec2_2(d2))

        # Decoder 1
        u1 = self.upconv1(d2)
        cat1 = Tensor(
            np.concatenate([u1.data, e1.data], axis=1), requires_grad=u1.requires_grad
        )
        d1 = self.relu(self.dec1_1(cat1))
        d1 = self.relu(self.dec1_2(d1))

        return self.final_conv(d1)
