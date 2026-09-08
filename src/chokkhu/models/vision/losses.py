"""Specialized Loss Functions for Computer Vision (Object Detection & Segmentation) from First Principles."""

from __future__ import annotations

from typing import Dict, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.losses import Loss


class FocalLoss(Loss):
    """Focal Loss for Dense Object Detection & Class Imbalance (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t + eps)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-12,
    ) -> None:
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        y_pred: Union[np.ndarray, Tensor],
        y_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        pred_data = (
            y_pred.data
            if isinstance(y_pred, Tensor)
            else np.asarray(y_pred, dtype=np.float64)
        )
        true_data = (
            y_true.data
            if isinstance(y_true, Tensor)
            else np.asarray(y_true, dtype=np.float64)
        )

        probs = np.clip(pred_data, self.eps, 1.0 - self.eps)

        if true_data.ndim == 1 or (true_data.ndim == 2 and true_data.shape[1] == 1):
            p_t = true_data * probs + (1.0 - true_data) * (1.0 - probs)
            alpha_t = true_data * self.alpha + (1.0 - true_data) * (1.0 - self.alpha)
            loss_arr = -alpha_t * ((1.0 - p_t) ** self.gamma) * np.log(p_t)
        else:
            p_t = np.sum(true_data * probs, axis=-1, keepdims=True)
            alpha_t = self.alpha
            loss_arr = -alpha_t * ((1.0 - p_t) ** self.gamma) * np.log(p_t)

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(y_pred, Tensor) and y_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)


class DiceLoss(Loss):
    """Dice Loss for Semantic and Medical Image Segmentation (Milletari et al., 2016).

    Dice = 2 * |X cap Y| / (|X| + |Y| + eps)
    DiceLoss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        y_pred: Union[np.ndarray, Tensor],
        y_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        pred_data = (
            y_pred.data
            if isinstance(y_pred, Tensor)
            else np.asarray(y_pred, dtype=np.float64)
        )
        true_data = (
            y_true.data
            if isinstance(y_true, Tensor)
            else np.asarray(y_true, dtype=np.float64)
        )

        pred_flat = pred_data.reshape(pred_data.shape[0], -1)
        true_flat = true_data.reshape(true_data.shape[0], -1)

        intersection = np.sum(pred_flat * true_flat, axis=-1)
        cardinality = np.sum(pred_flat, axis=-1) + np.sum(true_flat, axis=-1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_arr = 1.0 - dice

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(y_pred, Tensor) and y_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)


class TverskyLoss(Loss):
    """Tversky Loss with Asymmetric False Positive / False Negative Penalties (Salehi et al., 2017).

    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    TverskyLoss = 1 - TI
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        y_pred: Union[np.ndarray, Tensor],
        y_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        pred_data = (
            y_pred.data
            if isinstance(y_pred, Tensor)
            else np.asarray(y_pred, dtype=np.float64)
        )
        true_data = (
            y_true.data
            if isinstance(y_true, Tensor)
            else np.asarray(y_true, dtype=np.float64)
        )

        pred_flat = pred_data.reshape(pred_data.shape[0], -1)
        true_flat = true_data.reshape(true_data.shape[0], -1)

        tp = np.sum(pred_flat * true_flat, axis=-1)
        fp = np.sum(pred_flat * (1.0 - true_flat), axis=-1)
        fn = np.sum((1.0 - pred_flat) * true_flat, axis=-1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss_arr = 1.0 - tversky

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(y_pred, Tensor) and y_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)


def _compute_box_metrics(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Helper to compute Intersection, Union, Enclosing Box, Distance, and Aspect Ratios."""
    b1_x1, b1_y1, b1_x2, b1_y2 = (
        boxes1[:, 0],
        boxes1[:, 1],
        boxes1[:, 2],
        boxes1[:, 3],
    )
    b2_x1, b2_y1, b2_x2, b2_y2 = (
        boxes2[:, 0],
        boxes2[:, 1],
        boxes2[:, 2],
        boxes2[:, 3],
    )

    w1, h1 = np.maximum(0.0, b1_x2 - b1_x1), np.maximum(0.0, b1_y2 - b1_y1)
    w2, h2 = np.maximum(0.0, b2_x2 - b2_x1), np.maximum(0.0, b2_y2 - b2_y1)

    area1 = w1 * h1
    area2 = w2 * h2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + eps)

    enc_x1 = np.minimum(b1_x1, b2_x1)
    enc_y1 = np.minimum(b1_y1, b2_y1)
    enc_x2 = np.maximum(b1_x2, b2_x2)
    enc_y2 = np.maximum(b1_y2, b2_y2)

    enc_w = np.maximum(0.0, enc_x2 - enc_x1)
    enc_h = np.maximum(0.0, enc_y2 - enc_y1)
    enc_area = enc_w * enc_h
    enc_diag_sq = enc_w**2 + enc_h**2 + eps

    c1_x, c1_y = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    c2_x, c2_y = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
    center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    v = (4.0 / (np.pi**2)) * (
        (np.arctan(w2 / (h2 + eps)) - np.arctan(w1 / (h1 + eps))) ** 2
    )
    alpha = v / ((1.0 - iou) + v + eps)

    return {
        "iou": iou,
        "union_area": union_area,
        "enc_area": enc_area,
        "enc_diag_sq": enc_diag_sq,
        "center_dist_sq": center_dist_sq,
        "v": v,
        "alpha": alpha,
    }


class GIoULoss(Loss):
    """Generalized Intersection over Union (GIoU) Loss (Rezatofighi et al., 2019).

    GIoU = IoU - (Area(C) - Area(Union)) / Area(C)
    GIoULoss = 1 - GIoU
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        boxes_pred: Union[np.ndarray, Tensor],
        boxes_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        p_data = (
            boxes_pred.data
            if isinstance(boxes_pred, Tensor)
            else np.asarray(boxes_pred, dtype=np.float64)
        )
        t_data = (
            boxes_true.data
            if isinstance(boxes_true, Tensor)
            else np.asarray(boxes_true, dtype=np.float64)
        )

        metrics = _compute_box_metrics(p_data, t_data, eps=self.eps)
        giou = metrics["iou"] - (metrics["enc_area"] - metrics["union_area"]) / (
            metrics["enc_area"] + self.eps
        )
        loss_arr = 1.0 - giou

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(boxes_pred, Tensor) and boxes_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)


class DIoULoss(Loss):
    """Distance-IoU (DIoU) Loss (Zheng et al., 2020).

    DIoU = IoU - (rho^2(b, b_gt) / c^2)
    DIoULoss = 1 - DIoU
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        boxes_pred: Union[np.ndarray, Tensor],
        boxes_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        p_data = (
            boxes_pred.data
            if isinstance(boxes_pred, Tensor)
            else np.asarray(boxes_pred, dtype=np.float64)
        )
        t_data = (
            boxes_true.data
            if isinstance(boxes_true, Tensor)
            else np.asarray(boxes_true, dtype=np.float64)
        )

        metrics = _compute_box_metrics(p_data, t_data, eps=self.eps)
        diou = metrics["iou"] - metrics["center_dist_sq"] / metrics["enc_diag_sq"]
        loss_arr = 1.0 - diou

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(boxes_pred, Tensor) and boxes_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)


class CIoULoss(Loss):
    """Complete-IoU (CIoU) Loss (Zheng et al., 2020).

    CIoU = IoU - (rho^2(b, b_gt) / c^2 + alpha * v)
    CIoULoss = 1 - CIoU
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        boxes_pred: Union[np.ndarray, Tensor],
        boxes_true: Union[np.ndarray, Tensor],
    ) -> Tensor:
        p_data = (
            boxes_pred.data
            if isinstance(boxes_pred, Tensor)
            else np.asarray(boxes_pred, dtype=np.float64)
        )
        t_data = (
            boxes_true.data
            if isinstance(boxes_true, Tensor)
            else np.asarray(boxes_true, dtype=np.float64)
        )

        metrics = _compute_box_metrics(p_data, t_data, eps=self.eps)
        ciou = metrics["iou"] - (
            metrics["center_dist_sq"] / metrics["enc_diag_sq"]
            + metrics["alpha"] * metrics["v"]
        )
        loss_arr = 1.0 - ciou

        loss_val: Union[float, np.ndarray]
        if self.reduction == "mean":
            loss_val = float(np.mean(loss_arr))
        elif self.reduction == "sum":
            loss_val = float(np.sum(loss_arr))
        else:
            loss_val = loss_arr

        requires_grad = isinstance(boxes_pred, Tensor) and boxes_pred.requires_grad
        return Tensor(loss_val, requires_grad=requires_grad)
