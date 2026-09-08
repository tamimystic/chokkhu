"""Unit tests for Computer Vision specialized loss functions."""

from __future__ import annotations
import numpy as np

from chokkhu.models.vision import (
    CIoULoss,
    DIoULoss,
    DiceLoss,
    FocalLoss,
    GIoULoss,
    TverskyLoss,
)


def test_focal_loss_binary():
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    # Well-classified easy positive
    y_pred_easy = np.array([[0.99], [0.95]])
    y_true = np.array([[1.0], [1.0]])
    loss_easy = focal(y_pred_easy, y_true).data

    # Hard positive
    y_pred_hard = np.array([[0.20], [0.30]])
    loss_hard = focal(y_pred_hard, y_true).data

    assert loss_easy < loss_hard
    assert float(loss_easy) > 0.0


def test_focal_loss_multiclass():
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.85, 0.05]])
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = focal(y_pred, y_true)
    assert float(loss.data) > 0.0


def test_dice_loss_properties():
    dice_loss_fn = DiceLoss(smooth=1.0)

    # Perfect overlap
    pred_perfect = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    true_perfect = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    loss_perfect = float(dice_loss_fn(pred_perfect, true_perfect).data)
    assert np.isclose(loss_perfect, 0.0, atol=1e-3)

    # Zero overlap
    pred_zero = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    true_zero = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    loss_zero = float(dice_loss_fn(pred_zero, true_zero).data)
    assert loss_zero > 0.6


def test_tversky_loss():
    tversky_fn = TverskyLoss(alpha=0.7, beta=0.3)
    pred = np.array([[[0.8, 0.2], [0.1, 0.9]]])
    true = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    loss = float(tversky_fn(pred, true).data)
    assert 0.0 <= loss <= 1.0


def test_giou_diou_ciou_losses():
    # Identical boxes in [x1, y1, x2, y2]
    boxes_pred = np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]])
    boxes_true = np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]])

    giou_fn = GIoULoss()
    diou_fn = DIoULoss()
    ciou_fn = CIoULoss()

    giou_loss = float(giou_fn(boxes_pred, boxes_true).data)
    diou_loss = float(diou_fn(boxes_pred, boxes_true).data)
    ciou_loss = float(ciou_fn(boxes_pred, boxes_true).data)

    assert np.isclose(giou_loss, 0.0, atol=1e-4)
    assert np.isclose(diou_loss, 0.0, atol=1e-4)
    assert np.isclose(ciou_loss, 0.0, atol=1e-4)

    # Disjoint boxes
    boxes_disjoint = np.array(
        [[100.0, 100.0, 150.0, 150.0], [200.0, 200.0, 250.0, 250.0]]
    )
    giou_disjoint = float(giou_fn(boxes_disjoint, boxes_true).data)
    diou_disjoint = float(diou_fn(boxes_disjoint, boxes_true).data)
    ciou_disjoint = float(ciou_fn(boxes_disjoint, boxes_true).data)

    assert giou_disjoint > 1.0  # GIoU < 0 for distant boxes, so 1 - GIoU > 1.0
    assert diou_disjoint > 1.0
    assert ciou_disjoint > 1.0
