"""Support Vector Data Description (SVDD) and Minimum Enclosing Ball Anomaly Detection.

Formulated from first principles using kernelized Minimum Enclosing Ball (MEB) quadratic programming,
support vector boundary extraction, and dual sphere center projection in pure NumPy.
"""

from typing import Optional

import numpy as np


class SupportVectorDataDescription:
    r"""Support Vector Data Description (SVDD) One-Class Anomaly Detector.

    Fits the minimal enclosing hypersphere in kernel Hilbert space:
    \min_{R, a, \xi} R^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad \|\phi(x_i) - a\|^2 \le R^2 + \xi_i, \quad \xi_i \ge 0

    Dual QP Formulation:
    \max_\alpha \sum_i \alpha_i K(x_i, x_i) - \sum_{i, j} \alpha_i \alpha_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum_i \alpha_i = 1

    Parameters
    ----------
    C : float, default=0.1
        Outlier penalty slack trade-off parameter (C in (0, 1]).
    kernel : str, default="rbf"
        Kernel type: "rbf", "poly", or "linear".
    gamma : Optional[float], default=None
        RBF kernel scale parameter gamma = 1 / (2 * sigma^2). If None, defaults to 1 / d.
    degree : int, default=3
        Polynomial kernel degree.
    max_iter : int, default=200
        Maximum SMO optimization iterations.
    tol : float, default=1e-5
        KKT optimality tolerance.
    """

    def __init__(
        self,
        C: float = 0.1,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        max_iter: int = 200,
        tol: float = 1e-5,
    ) -> None:
        self.C = max(1e-4, min(1.0, float(C)))
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.max_iter = max(10, int(max_iter))
        self.tol = float(tol)

        self.support_vectors_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.radius_sq_: float = 0.0
        self.center_norm_sq_: float = 0.0
        self.gamma_: float = 1.0

    def _kernel_func(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Gram matrix between X1 and X2."""
        if self.kernel == "rbf":
            # K(x, y) = exp(-gamma * ||x - y||^2)
            diff = X1[:, None, :] - X2[None, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            return np.exp(-self.gamma_ * dist_sq)
        elif self.kernel == "poly":
            # K(x, y) = (gamma * <x, y> + 1)^degree
            return (self.gamma_ * np.dot(X1, X2.T) + 1.0) ** self.degree
        else:  # linear
            return np.dot(X1, X2.T)

    def fit(self, X: np.ndarray) -> "SupportVectorDataDescription":
        """Fit SVDD minimum enclosing sphere on training observations X."""
        X_mat = np.asarray(X, dtype=float)
        N, d = X_mat.shape

        self.gamma_ = 1.0 / d if self.gamma is None else float(self.gamma)

        # 1. Compute Kernel Gram Matrix K
        K = self._kernel_func(X_mat, X_mat)
        diag_K = np.diag(K)

        # 2. Sequential Minimal Optimization (SMO) for SVDD
        alpha = np.ones(N, dtype=float) / N  # Initial uniform feasible distribution
        # Gradient: g_i = diag(K)_i - 2 * sum_j alpha_j K_ij
        g = diag_K - 2.0 * np.dot(K, alpha)

        for _ in range(self.max_iter):
            # Find violating pair (i, j)
            i = int(np.argmax(g))
            j = int(np.argmin(g))

            if g[i] - g[j] < self.tol:
                break

            sum_ij = alpha[i] + alpha[j]
            eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
            if abs(eta) < 1e-12:
                continue

            delta_j = (g[i] - g[j]) / (-eta)
            new_alpha_j = alpha[j] + delta_j

            # Clip within [0, C] and maintain sum_ij
            low = max(0.0, sum_ij - self.C)
            high = min(self.C, sum_ij)
            new_alpha_j = float(np.clip(new_alpha_j, low, high))

            new_alpha_i = sum_ij - new_alpha_j

            diff_i = new_alpha_i - alpha[i]
            diff_j = new_alpha_j - alpha[j]

            alpha[i] = new_alpha_i
            alpha[j] = new_alpha_j

            # Update gradient vector
            g -= 2.0 * (diff_i * K[:, i] + diff_j * K[:, j])

        # 3. Extract support vectors (alpha > 1e-6)
        sv_mask = alpha > 1e-6
        self.support_vectors_ = X_mat[sv_mask]
        self.alpha_ = alpha[sv_mask]

        # Normalize alpha to exactly sum to 1
        self.alpha_ = self.alpha_ / np.sum(self.alpha_)

        # 4. Center norm: ||a||^2 = sum_{i, j} alpha_i alpha_j K(x_i, x_j)
        K_sv = self._kernel_func(self.support_vectors_, self.support_vectors_)
        self.center_norm_sq_ = float(np.dot(self.alpha_, np.dot(K_sv, self.alpha_)))

        # 5. Radius calculation from boundary support vectors (0 < alpha < C)
        boundary_mask = (self.alpha_ < (self.C - 1e-4)) & (self.alpha_ > 1e-4)
        if np.any(boundary_mask):
            sv_boundary = self.support_vectors_[boundary_mask]
        else:
            sv_boundary = self.support_vectors_

        K_b_sv = self._kernel_func(sv_boundary, self.support_vectors_)
        diag_b = np.diag(self._kernel_func(sv_boundary, sv_boundary))
        dists_sq = diag_b - 2.0 * np.dot(K_b_sv, self.alpha_) + self.center_norm_sq_
        self.radius_sq_ = float(max(1e-12, float(np.mean(dists_sq))))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance from hypersphere boundary.

        Negative values denote inliers (inside sphere), positive values denote anomalies (outside).
        """
        if self.support_vectors_ is None or self.alpha_ is None:
            raise ValueError("SVDD has not been fitted yet.")

        X_mat = np.asarray(X, dtype=float)
        K_x_sv = self._kernel_func(X_mat, self.support_vectors_)
        diag_x = np.diag(self._kernel_func(X_mat, X_mat))

        dist_sq = diag_x - 2.0 * np.dot(K_x_sv, self.alpha_) + self.center_norm_sq_
        # Distance relative to boundary: dist^2 - R^2
        return dist_sq - self.radius_sq_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outlier status: 1 for anomaly (outside), 0 for normal inlier."""
        scores = self.decision_function(X)
        return (scores > 0.0).astype(int)
