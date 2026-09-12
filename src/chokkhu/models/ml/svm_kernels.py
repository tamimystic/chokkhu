"""Support Vector Machines (SVC, SVR, OneClassSVM, LinearSVC) in Pure NumPy.

References:
- Platt (1998): "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines".
- Scholkopf et al. (2001): "Estimating the Support of a High-Dimensional Distribution" (Neural Computation).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


def compute_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 0.0,
) -> np.ndarray:
    """Compute pairwise kernel Gram matrix."""
    if kernel == "linear":
        return np.dot(X1, X2.T)
    elif kernel == "poly":
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree
    elif kernel == "sigmoid":
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)
    elif kernel == "rbf":
        sq_dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-gamma * sq_dists)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


class SVC(ChokkhuModel):
    """Support Vector Classifier with Platt's SMO / Dual Coordinate Descent."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 200,
    ) -> None:
        super().__init__()
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.alpha_: np.ndarray = np.array([], dtype=np.float64)
        self.b_: float = 0.0
        self.support_vectors_: np.ndarray = np.array([], dtype=np.float64)
        self.support_y_: np.ndarray = np.array([], dtype=np.float64)
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> SVC:
        if y is None:
            raise ValueError("y cannot be None for SVC")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]

        self.classes = np.unique(y_arr)
        if len(self.classes) != 2:
            raise ValueError("Binary classification only supported for core SVC")

        # Map to {-1, +1}
        y_mapped = np.where(y_arr == self.classes[0], -1.0, 1.0)
        K = compute_kernel(
            x_arr, x_arr, self.kernel, self.gamma, self.degree, self.coef0
        )

        # Dual coordinate descent for Box-Constrained QP
        alpha = np.zeros(n_samples, dtype=np.float64)
        grad = np.ones(n_samples, dtype=np.float64)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                q_ii = K[i, i]
                if q_ii <= 1e-12:
                    continue
                old_alpha = alpha[i]
                new_alpha = np.clip(old_alpha + grad[i] / q_ii, 0.0, self.C)
                diff = new_alpha - old_alpha
                if abs(diff) > 1e-8:
                    alpha[i] = new_alpha
                    grad -= diff * y_mapped[i] * y_mapped * K[:, i]

        sv_mask = alpha > 1e-6
        self.alpha_ = alpha[sv_mask]
        self.support_vectors_ = x_arr[sv_mask]
        self.support_y_ = y_mapped[sv_mask]

        if len(self.alpha_) > 0:
            free_sv = (alpha > 1e-6) & (alpha < self.C - 1e-6)
            if np.any(free_sv):
                sv_idx = int(np.where(free_sv)[0][0])
            else:
                sv_idx = int(np.where(sv_mask)[0][0])
            self.b_ = float(
                y_mapped[sv_idx]
                - np.sum(self.alpha_ * self.support_y_ * K[sv_mask, sv_idx])
            )
        else:
            self.b_ = 0.0

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if len(self.alpha_) == 0:
            return np.zeros(len(x_arr))
        K_test = compute_kernel(
            x_arr,
            self.support_vectors_,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        )
        return np.dot(K_test, self.alpha_ * self.support_y_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes[1], self.classes[0])


class SVR(ChokkhuModel):
    """Support Vector Regression with epsilon-insensitive loss."""

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: float = 0.1,
        max_iter: int = 200,
    ) -> None:
        super().__init__()
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)

        self.dual_coef_: np.ndarray = np.array([], dtype=np.float64)
        self.support_vectors_: np.ndarray = np.array([], dtype=np.float64)
        self.b_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> SVR:
        if y is None:
            raise ValueError("y cannot be None for SVR")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = x_arr.shape[0]

        K = compute_kernel(x_arr, x_arr, self.kernel, self.gamma)
        # Dual variables alpha+ and alpha-
        alpha_p = np.zeros(n_samples, dtype=np.float64)
        alpha_m = np.zeros(n_samples, dtype=np.float64)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                pred_i = np.dot(K[i], alpha_p - alpha_m)
                err = pred_i - y_arr[i]

                # Update alpha+
                delta_p = (-err - self.epsilon) / max(K[i, i], 1e-6)
                alpha_p[i] = np.clip(alpha_p[i] + delta_p, 0.0, self.C)

                # Update alpha-
                delta_m = (err - self.epsilon) / max(K[i, i], 1e-6)
                alpha_m[i] = np.clip(alpha_m[i] + delta_m, 0.0, self.C)

        dual_coef = alpha_p - alpha_m
        sv_mask = np.abs(dual_coef) > 1e-6
        self.dual_coef_ = dual_coef[sv_mask]
        self.support_vectors_ = x_arr[sv_mask]
        self.b_ = float(np.mean(y_arr - np.dot(K, dual_coef)))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if len(self.dual_coef_) == 0:
            return np.full(len(x_arr), self.b_)
        K_test = compute_kernel(x_arr, self.support_vectors_, self.kernel, self.gamma)
        return np.dot(K_test, self.dual_coef_) + self.b_


class OneClassSVM(ChokkhuModel):
    """Unsupervised One-Class SVM for Novelty and Outlier Detection."""

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: float = 0.1,
        max_iter: int = 200,
    ) -> None:
        super().__init__()
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)

        self.alpha_: np.ndarray = np.array([], dtype=np.float64)
        self.support_vectors_: np.ndarray = np.array([], dtype=np.float64)
        self.rho_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> OneClassSVM:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        K = compute_kernel(x_arr, x_arr, self.kernel, self.gamma)
        # Uniform initial dual weights summing to 1
        alpha = np.ones(n_samples, dtype=np.float64) / float(n_samples)
        upper_bound = 1.0 / (self.nu * n_samples)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                grad_i = np.dot(K[i], alpha)
                alpha[i] = np.clip(alpha[i] - 0.01 * grad_i, 0.0, upper_bound)
            alpha = alpha / np.sum(alpha)

        sv_mask = alpha > 1e-5
        self.alpha_ = alpha[sv_mask]
        self.support_vectors_ = x_arr[sv_mask]
        K_sv = compute_kernel(
            self.support_vectors_, self.support_vectors_, self.kernel, self.gamma
        )
        self.rho_ = float(np.mean(np.dot(K_sv, self.alpha_)))
        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if len(self.alpha_) == 0:
            return np.zeros(len(x_arr))
        K_test = compute_kernel(x_arr, self.support_vectors_, self.kernel, self.gamma)
        return np.dot(K_test, self.alpha_) - self.rho_

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return np.where(df >= 0, 1, -1)


class LinearSVC(ChokkhuModel):
    """High-Speed Linear Support Vector Classifier via Dual Coordinate Descent."""

    def __init__(
        self,
        C: float = 1.0,
        loss: str = "squared_hinge",
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.C = float(C)
        self.loss = loss
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> LinearSVC:
        if y is None:
            raise ValueError("y cannot be None for LinearSVC")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes = np.unique(y_arr)
        y_mapped = np.where(y_arr == self.classes[0], -1.0, 1.0)

        # Dual Coordinate Descent
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        alpha = np.zeros(n_samples, dtype=np.float64)
        diag = np.sum(x_arr**2, axis=1) + 1.0

        for _ in range(self.max_iter):
            max_viol = 0.0
            for i in range(n_samples):
                g = (np.dot(x_arr[i], w) + b) * y_mapped[i] - 1.0
                if self.loss == "squared_hinge":
                    g += alpha[i] / (2.0 * self.C)
                    denom = diag[i] + 1.0 / (2.0 * self.C)
                else:
                    denom = diag[i]

                old_alpha = alpha[i]
                new_alpha = np.clip(old_alpha - g / max(denom, 1e-6), 0.0, self.C)
                diff = (new_alpha - old_alpha) * y_mapped[i]
                if abs(diff) > 1e-8:
                    w += diff * x_arr[i]
                    b += diff
                    alpha[i] = new_alpha
                    max_viol = max(max_viol, abs(g))

            if max_viol < self.tol:
                break

        self.coef_ = w
        self.intercept_ = b
        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes[1], self.classes[0])
