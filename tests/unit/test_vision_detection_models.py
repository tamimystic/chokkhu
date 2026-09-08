"""Unit tests for Object Detection Heads (YOLO, SSD, RetinaNet) and Anchor Grid Generator."""

from __future__ import annotations
import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    RetinaNetHead,
    SSDHead,
    YOLOHead,
    generate_anchor_grid,
)


def test_yolo_head():
    N = 2
    in_c = 128
    num_classes = 20
    num_anchors = 3
    H, W = 13, 13

    head = YOLOHead(in_channels=in_c, num_classes=num_classes, num_anchors=num_anchors)
    x = Tensor(np.random.randn(N, in_c, H, W))
    out = head(x)

    assert out.shape == (N, num_anchors, H, W, 5 + num_classes)


def test_ssd_head():
    N = 2
    # Simulate multi-scale feature maps from backbone
    feat1 = Tensor(np.random.randn(N, 64, 16, 16))
    feat2 = Tensor(np.random.randn(N, 128, 8, 8))

    # feat1 has 4 anchors per location, feat2 has 6 anchors per location
    head = SSDHead(
        in_channels_list=[64, 128],
        num_anchors_list=[4, 6],
        num_classes=10,
    )
    cls_preds, reg_preds = head([feat1, feat2])

    expected_anchors = 16 * 16 * 4 + 8 * 8 * 6  # 1024 + 384 = 1408
    assert cls_preds.shape == (N, expected_anchors, 10)
    assert reg_preds.shape == (N, expected_anchors, 4)


def test_retinanet_head():
    N = 2
    in_c = 64
    # Simulate FPN P3, P4, P5
    p3 = Tensor(np.random.randn(N, in_c, 16, 16))
    p4 = Tensor(np.random.randn(N, in_c, 8, 8))
    p5 = Tensor(np.random.randn(N, in_c, 4, 4))

    head = RetinaNetHead(in_channels=in_c, num_anchors=9, num_classes=5, num_convs=2)
    cls_outs, reg_outs = head([p3, p4, p5])

    assert len(cls_outs) == 3
    assert len(reg_outs) == 3
    assert cls_outs[0].shape == (N, 9 * 5, 16, 16)
    assert reg_outs[0].shape == (N, 9 * 4, 16, 16)


def test_anchor_grid_generation():
    image_size = (128, 128)
    feature_size = (8, 8)
    scales = [16.0, 32.0]
    aspect_ratios = [0.5, 1.0, 2.0]

    anchors = generate_anchor_grid(
        image_size=image_size,
        feature_size=feature_size,
        anchor_scales=scales,
        aspect_ratios=aspect_ratios,
    )
    # Total anchors = 8 * 8 * (2 * 3) = 64 * 6 = 384
    assert anchors.shape == (384, 4)
    # Ensure x2 >= x1 and y2 >= y1
    assert np.all(anchors[:, 2] >= anchors[:, 0])
    assert np.all(anchors[:, 3] >= anchors[:, 1])
