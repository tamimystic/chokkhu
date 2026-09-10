"""Uplift Modeling, Campaign Targeting & Qini Evaluation.

Pure NumPy implementations of:
- TwoModelUplift: Separate treated/control classifiers for differential conversion
- ClassTransformationUplift: Single-model target transformation (Lai / Jaskowski)
- Uplift evaluation metrics: qini_curve, qini_score, cumulative_gain_curve, uplift_at_k
"""

from typing import List, Optional, Tuple
import numpy as np


class TwoModelUplift:
    r"""Two-Model Uplift Estimator for Incremental Conversion Modeling.

    Trains two independent classifiers :math:`P(Y=1|X, T=1)` and :math:`P(Y=1|X, T=0)`:

    .. math::
        \text{Uplift}(X) = P(Y=1|X, T=1) - P(Y=1|X, T=0)

    Parameters
    ----------
    lr : float, default=0.05
        Learning rate.
    n_epochs : int, default=100
        Number of epochs.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        lr: float = 0.05,
        n_epochs: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed

        self.w1: Optional[np.ndarray] = None
        self.b1: float = 0.0
        self.w0: Optional[np.ndarray] = None
        self.b0: float = 0.0
        self.is_fitted: bool = False

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.seed)
        w = (rng.randn(n_features) * 0.01).astype(np.float32)
        bias = 0.0

        for _ in range(self.n_epochs):
            logits = np.dot(X, w) + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            err = probs - y
            w -= self.lr * (np.dot(X.T, err) / n_samples + 0.01 * w)
            bias -= float(self.lr * np.mean(err))

        return w, bias

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TwoModelUplift":
        """Fit uplift models on treated and control groups."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.int32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        treated_idx = np.where(T_arr == 1)[0]
        control_idx = np.where(T_arr == 0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            raise ValueError(
                "Both treated (T=1) and control (T=0) observations are required."
            )

        self.w1, self.b1 = self._fit_logistic(X_arr[treated_idx], Y_arr[treated_idx])
        self.w0, self.b0 = self._fit_logistic(X_arr[control_idx], Y_arr[control_idx])
        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Predict individual incremental uplift :math:`P(Y=1|X, T=1) - P(Y=1|X, T=0)`."""
        if not self.is_fitted or self.w1 is None or self.w0 is None:
            raise RuntimeError("Model must be fitted before predict_uplift.")

        X_arr = np.asarray(X, dtype=np.float32)
        p1 = 1.0 / (
            1.0 + np.exp(-np.clip(np.dot(X_arr, self.w1) + self.b1, -30.0, 30.0))
        )
        p0 = 1.0 / (
            1.0 + np.exp(-np.clip(np.dot(X_arr, self.w0) + self.b0, -30.0, 30.0))
        )
        return p1 - p0


class ClassTransformationUplift:
    r"""Class Transformation Uplift Estimator (Lai's Generalized Method).

    Maps binary outcome :math:`Y \in \{0, 1\}` and treatment :math:`T \in \{0, 1\}`
    into transformed target variable :math:`Z`:

    .. math::
        Z_i = Y_i T_i + (1 - Y_i)(1 - T_i)

    For randomized trials with :math:`P(T=1) = 0.5`, :math:`2 P(Z=1|X) - 1 = 	ext{Uplift}(X)`.

    Parameters
    ----------
    lr : float, default=0.05
        Learning rate.
    n_epochs : int, default=100
        Number of epochs.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        lr: float = 0.05,
        n_epochs: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> "ClassTransformationUplift":
        """Fit single classifier on transformed target Z."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.int32).ravel()
        Y_arr = np.asarray(Y, dtype=np.int32).ravel()

        # Z = 1 if (Y=1 & T=1) or (Y=0 & T=0), else 0
        Z_arr = (Y_arr == T_arr).astype(np.float32)

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.seed)
        self.weights = (rng.randn(n_features) * 0.01).astype(np.float32)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            logits = np.dot(X_arr, self.weights) + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            errors = probs - Z_arr
            self.weights -= self.lr * (
                np.dot(X_arr.T, errors) / n_samples + 0.01 * self.weights
            )
            self.bias -= float(self.lr * np.mean(errors))

        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Predict individual treatment uplift."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before calling predict_uplift.")

        X_arr = np.asarray(X, dtype=np.float32)
        logits = np.dot(X_arr, self.weights) + self.bias
        p_z = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return 2.0 * p_z - 1.0


def qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate cumulative Qini curve coordinates (x_fractions, qini_values)."""
    y_arr = np.asarray(y_true, dtype=np.float32).ravel()
    t_arr = np.asarray(treatment, dtype=np.int32).ravel()
    u_arr = np.asarray(uplift_preds, dtype=np.float32).ravel()

    n = len(y_arr)
    sorted_idx = np.argsort(-u_arr)
    y_sorted = y_arr[sorted_idx]
    t_sorted = t_arr[sorted_idx]

    total_treated: float = float(np.sum(t_arr == 1))
    total_control: float = float(np.sum(t_arr == 0))

    if total_control == 0 or total_treated == 0:
        return np.linspace(0, 1, n_bins + 1, dtype=np.float32), np.zeros(
            n_bins + 1, dtype=np.float32
        )

    bin_sizes = np.linspace(0, n, n_bins + 1, dtype=int)
    qini_vals: List[float] = [0.0]
    fractions: List[float] = [0.0]

    for b in bin_sizes[1:]:
        sub_y = y_sorted[:b]
        sub_t = t_sorted[:b]

        n_t: float = float(np.sum(sub_t == 1))
        n_c: float = float(np.sum(sub_t == 0))

        y_t = float(np.sum(sub_y[sub_t == 1])) if n_t > 0 else 0.0
        y_c = float(np.sum(sub_y[sub_t == 0])) if n_c > 0 else 0.0

        val = float(y_t - y_c * (total_treated / total_control))
        qini_vals.append(val)
        fractions.append(float(b / n))

    return np.array(fractions, dtype=np.float32), np.array(qini_vals, dtype=np.float32)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible across NumPy 1.x and 2.x."""
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * (x[1:] - x[:-1])))


def qini_score(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
) -> float:
    """Calculate normalized Qini score / Area Under Qini Curve above random targeting."""
    x, y = qini_curve(y_true, treatment, uplift_preds, n_bins=20)
    # Area under model curve
    auc_model = _trapezoid(y, x)
    # Area under random line (connects (0, 0) to (1, y[-1]))
    auc_random = float(0.5 * y[-1])
    return float(auc_model - auc_random)


def cumulative_gain_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative gain curve representing incremental outcome percentage."""
    x, q = qini_curve(y_true, treatment, uplift_preds, n_bins=n_bins)
    max_val = q[-1] if abs(q[-1]) > 1e-9 else 1.0
    gains = q / max_val
    return x, gains


def uplift_at_k(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    k: float = 0.2,
) -> float:
    """Compute incremental uplift within the top-k percentile of predicted scores."""
    y_arr = np.asarray(y_true, dtype=np.float32).ravel()
    t_arr = np.asarray(treatment, dtype=np.int32).ravel()
    u_arr = np.asarray(uplift_preds, dtype=np.float32).ravel()

    n = len(y_arr)
    top_n = max(1, int(n * k))
    sorted_idx = np.argsort(-u_arr)[:top_n]

    sub_y = y_arr[sorted_idx]
    sub_t = t_arr[sorted_idx]

    treated_mask = sub_t == 1
    control_mask = sub_t == 0

    if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
        return 0.0

    mean_treated = float(np.mean(sub_y[treated_mask]))
    mean_control = float(np.mean(sub_y[control_mask]))
    return float(mean_treated - mean_control)
