"""Feature Pyramid Networks (FPN) and Path Aggregation Network (PANet) from Scratch (Lin et al., 2017; Liu et al., 2018)."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Module
from .conv_layers import Conv2D


class FeaturePyramidNetwork(Module):
    """Feature Pyramid Network (FPN) for Multi-Scale Feature Extraction."""

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.lateral_convs = []
        self.fpn_convs = []

        for i, in_c in enumerate(in_channels_list):
            lat = Conv2D(in_c, out_channels, kernel_size=1, stride=1, padding=0)
            fpn = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(lat)
            self.fpn_convs.append(fpn)
            setattr(self, f"lateral_{i}", lat)
            setattr(self, f"fpn_{i}", fpn)

    def _upsample_2x(self, x: Tensor, target_shape: Tuple[int, int]) -> Tensor:
        """Vectorized nearest-neighbor 2x upsampling."""
        N, C, H, W = x.shape
        target_h, target_w = target_shape
        # Repeat along spatial dimensions
        data = x.data
        up = np.repeat(np.repeat(data, target_h // H, axis=2), target_w // W, axis=3)
        return Tensor(up, requires_grad=x.requires_grad)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward pass over backbone multi-scale feature maps [C2, C3, C4, C5].

        Returns:
            List of multi-scale pyramid features [P2, P3, P4, P5]
        """
        # Start top-down pathway from highest-level feature map
        laterals = [
            lat_conv(feat) for lat_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down merging
        for i in range(len(laterals) - 1, 0, -1):
            prev_lat = laterals[i]
            curr_lat = laterals[i - 1]
            up = self._upsample_2x(prev_lat, (curr_lat.shape[2], curr_lat.shape[3]))
            laterals[i - 1] = curr_lat + up

        # Apply 3x3 FPN smoothing convolutions to eliminate aliasing
        outputs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]
        return outputs


class PANet(Module):
    """Path Aggregation Network (PANet) with Bottom-Up Path Augmentation."""

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels=out_channels)
        self.downsample_convs = []

        for i in range(len(in_channels_list) - 1):
            down_conv = Conv2D(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.downsample_convs.append(down_conv)
            setattr(self, f"pan_down_{i}", down_conv)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward pass through FPN + Bottom-Up Path Augmentation."""
        fpn_features = self.fpn(features)
        pan_features = [fpn_features[0]]

        for i in range(len(fpn_features) - 1):
            prev_n = pan_features[-1]
            down_sampled = self.downsample_convs[i](prev_n)
            curr_p = fpn_features[i + 1]
            n_curr = curr_p + down_sampled
            pan_features.append(n_curr)

        return pan_features
