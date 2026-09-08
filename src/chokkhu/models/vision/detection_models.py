"""Object Detection Heads and Decoding Frameworks (YOLO, SSD, RetinaNet) from First Principles."""

from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Module
from .conv_layers import Conv2D


class YOLOHead(Module):
    """YOLO Multi-Scale Detection Head for Bounding Box & Class Prediction (Redmon et al., 2016/2018)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Per anchor: (x, y, w, h, obj_conf) + num_classes
        self.out_channels_per_anchor = 5 + num_classes
        total_out_c = num_anchors * self.out_channels_per_anchor
        self.conv = Conv2D(in_channels, total_out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass converting feature map to grid cell predictions.

        Returns:
            Tensor of shape (N, num_anchors, H, W, 5 + num_classes)
        """
        raw = self.conv(x)
        N, _, H, W = raw.shape
        data = raw.data.reshape(N, self.num_anchors, self.out_channels_per_anchor, H, W)
        data = data.transpose(0, 1, 3, 4, 2)  # (N, num_anchors, H, W, 5 + num_classes)
        return Tensor(data, requires_grad=x.requires_grad)


class SSDHead(Module):
    """Single Shot MultiBox Detector (SSD) Multi-Scale Classification and Box Regression Head (Liu et al., 2016)."""

    def __init__(
        self,
        in_channels_list: List[int],
        num_anchors_list: List[int],
        num_classes: int = 21,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = []
        self.reg_convs = []

        for i, (in_c, num_a) in enumerate(zip(in_channels_list, num_anchors_list)):
            cls_c = Conv2D(
                in_c, num_a * num_classes, kernel_size=3, stride=1, padding=1
            )
            reg_c = Conv2D(in_c, num_a * 4, kernel_size=3, stride=1, padding=1)
            self.cls_convs.append(cls_c)
            self.reg_convs.append(reg_c)
            setattr(self, f"cls_conv_{i}", cls_c)
            setattr(self, f"reg_conv_{i}", reg_c)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward pass over multi-scale feature maps.

        Returns:
            (all_class_logits, all_box_deltas) with shapes:
            - all_class_logits: (N, total_anchors, num_classes)
            - all_box_deltas: (N, total_anchors, 4)
        """
        all_cls = []
        all_reg = []
        N = features[0].shape[0]

        for cls_c, reg_c, feat in zip(self.cls_convs, self.reg_convs, features):
            cls_out = cls_c(feat)
            reg_out = reg_c(feat)
            H, W = cls_out.shape[2], cls_out.shape[3]

            cls_flat = (
                cls_out.data.reshape(N, -1, self.num_classes, H, W)
                .transpose(0, 1, 3, 4, 2)
                .reshape(N, -1, self.num_classes)
            )
            reg_flat = (
                reg_out.data.reshape(N, -1, 4, H, W)
                .transpose(0, 1, 3, 4, 2)
                .reshape(N, -1, 4)
            )

            all_cls.append(cls_flat)
            all_reg.append(reg_flat)

        cat_cls = np.concatenate(all_cls, axis=1)
        cat_reg = np.concatenate(all_reg, axis=1)

        requires_grad = any(f.requires_grad for f in features)
        return Tensor(cat_cls, requires_grad=requires_grad), Tensor(
            cat_reg, requires_grad=requires_grad
        )


class RetinaNetHead(Module):
    """RetinaNet Subnets for Dense Classification (Focal Loss) and Box Regression (Lin et al., 2017)."""

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,
        num_classes: int = 80,
        num_convs: int = 4,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.cls_tower = []
        self.reg_tower = []
        for i in range(num_convs):
            c_cls = Conv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            c_reg = Conv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.cls_tower.append(c_cls)
            self.reg_tower.append(c_reg)
            setattr(self, f"cls_tower_{i}", c_cls)
            setattr(self, f"reg_tower_{i}", c_reg)

        self.cls_pred = Conv2D(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.reg_pred = Conv2D(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward pass over FPN feature levels P3 to P7.

        Returns:
            (cls_outputs, reg_outputs) per pyramid level
        """
        cls_outs = []
        reg_outs = []

        for feat in features:
            cls_feat = feat
            reg_feat = feat
            for c_cls, c_reg in zip(self.cls_tower, self.reg_tower):
                cls_feat = c_cls(cls_feat).relu()
                reg_feat = c_reg(reg_feat).relu()

            cls_outs.append(self.cls_pred(cls_feat))
            reg_outs.append(self.reg_pred(reg_feat))

        return cls_outs, reg_outs


def generate_anchor_grid(
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    anchor_scales: List[float],
    aspect_ratios: List[float],
) -> np.ndarray:
    """Generate dense anchor bounding boxes across grid cells [x1, y1, x2, y2]."""
    img_h, img_w = image_size
    feat_h, feat_w = feature_size
    stride_y = img_h / feat_h
    stride_x = img_w / feat_w

    coords: Any = np.meshgrid(
        np.arange(feat_h), np.arange(feat_w), indexing="ij"
    )
    grid_y: np.ndarray = np.asarray(coords[0], dtype=np.float64)
    grid_x: np.ndarray = np.asarray(coords[1], dtype=np.float64)
    centers_x = (grid_x + 0.5) * stride_x
    centers_y = (grid_y + 0.5) * stride_y

    anchors = []
    for scale in anchor_scales:
        for ar in aspect_ratios:
            w = scale * np.sqrt(ar)
            h = scale / np.sqrt(ar)
            x1 = centers_x - w / 2.0
            y1 = centers_y - h / 2.0
            x2 = centers_x + w / 2.0
            y2 = centers_y + h / 2.0
            anchor = np.stack([x1, y1, x2, y2], axis=-1)  # (feat_h, feat_w, 4)
            anchors.append(anchor)

    return np.stack(anchors, axis=2).reshape(
        -1, 4
    )  # (feat_h * feat_w * num_anchors, 4)
