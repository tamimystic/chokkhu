from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
from chokkhu.core.tensor import Function, Tensor


class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss: L = mean((y_pred - y_true)^2)."""

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff**2).mean()


class MAEFunction(Function):
    def forward(self, y_pred: Any, y_true: Any = None) -> Any:  # type: ignore[override]
        return np.array(np.mean(np.abs(y_pred - y_true)), dtype=np.float64)

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, y_true = self.inputs[0].data, self.inputs[1].data
        diff = y_pred - y_true
        grad = (np.sign(diff) / float(max(1, diff.size))) * gy
        return grad, np.zeros_like(y_true)


class MAELoss(Loss):
    """Mean Absolute Error Loss."""

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return MAEFunction()(y_pred, y_true)


class CrossEntropyFunction(Function):
    def __init__(self) -> None:
        super().__init__()
        self.probs: Optional[np.ndarray] = None

    def forward(self, logits: Any, targets: Any = None) -> Any:  # type: ignore[override]
        N = logits.shape[0]
        z = logits
        max_z = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - max_z)
        self.probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        eps = 1e-12

        if targets.ndim == 1 or targets.shape[-1] == 1:
            indices: np.ndarray = targets.astype(int).flatten()
            log_likelihood = -np.log(self.probs[np.arange(N), indices] + eps)
            return np.array(np.mean(log_likelihood), dtype=np.float64)
        else:
            loss_data = -np.mean(np.sum(targets * np.log(self.probs + eps), axis=-1))
            return np.array(loss_data, dtype=np.float64)

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits, targets = self.inputs[0].data, self.inputs[1].data
        N = logits.shape[0]
        if targets.ndim == 1 or targets.shape[-1] == 1:
            indices: np.ndarray = targets.astype(int).flatten()
            grad = self.probs.copy()
            grad[np.arange(N), indices] -= 1.0
            grad = (grad / float(N)) * gy
        else:
            grad = ((self.probs - targets) / float(N)) * gy
        return grad, np.zeros_like(targets)


class CrossEntropyLoss(Loss):
    """Numerically stabilized Softmax Cross Entropy Loss."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return CrossEntropyFunction()(logits, targets)


class BinaryCrossEntropyFunction(Function):
    def __init__(self, from_logits: bool = True) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.p: Optional[np.ndarray] = None

    def forward(self, y_pred: Any, y_true: Any = None) -> Any:  # type: ignore[override]
        eps = 1e-12
        if self.from_logits:
            self.p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -500.0, 500.0)))
        else:
            self.p = np.clip(y_pred, eps, 1.0 - eps)

        loss_val = -np.mean(
            y_true * np.log(self.p + eps) + (1.0 - y_true) * np.log(1.0 - self.p + eps)
        )
        return np.array(loss_val, dtype=np.float64)

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_true = self.inputs[1].data
        eps = 1e-12
        if self.from_logits:
            grad = ((self.p - y_true) / float(y_true.size)) * gy
        else:
            grad = (
                ((self.p - y_true) / (self.p * (1.0 - self.p) + eps))
                / float(y_true.size)
            ) * gy
        return grad, np.zeros_like(y_true)


class BinaryCrossEntropyLoss(Loss):
    """Binary Cross Entropy Loss."""

    def __init__(self, from_logits: bool = True) -> None:
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return BinaryCrossEntropyFunction(self.from_logits)(y_pred, y_true)


class HuberFunction(Function):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: Any, y_true: Any = None) -> Any:  # type: ignore[override]
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        is_small = abs_diff <= self.delta
        loss_val = np.mean(
            np.where(
                is_small, 0.5 * (diff**2), self.delta * (abs_diff - 0.5 * self.delta)
            )
        )
        return np.array(loss_val, dtype=np.float64)

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, y_true = self.inputs[0].data, self.inputs[1].data
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        is_small = abs_diff <= self.delta
        grad = (
            np.where(is_small, diff, self.delta * np.sign(diff))
            / float(max(1, diff.size))
        ) * gy
        return grad, np.zeros_like(y_true)


class HuberLoss(Loss):
    """Huber Loss (Smooth L1 Loss)."""

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return HuberFunction(self.delta)(y_pred, y_true)
