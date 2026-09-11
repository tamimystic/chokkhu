r"""Entropic Regularized Optimal Transport and Sinkhorn Algorithm in pure NumPy."""

from __future__ import annotations

from typing import Optional
import numpy as np


class SinkhornOptimalTransport:
    r"""Entropic Regularized Optimal Transport via Sinkhorn-Knopp Matrix Scaling (Cuturi 2013).

    Computes the regularized Wasserstein distance and optimal coupling matrix
    between discrete probability measures $a$ and $b$.

    Parameters
    ----------
    reg : float, default=0.05
        Entropic regularization coefficient (epsilon).
    max_iter : int, default=100
        Maximum number of Sinkhorn-Knopp scaling iterations.
    tol : float, default=1e-6
        Stopping criterion threshold for marginal error convergence.
    """

    def __init__(
        self,
        reg: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        if reg <= 0:
            raise ValueError("Regularization parameter reg must be > 0.")
        self.reg = float(reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.plan_: Optional[np.ndarray] = None
        self.distance_: float = 0.0

    def _compute_cost(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
        metric: str = "sqeuclidean",
    ) -> np.ndarray:
        """Computes pairwise ground metric cost matrix between source and target points."""
        if metric == "cityblock" or metric == "manhattan":
            diff = np.abs(X_s[:, np.newaxis, :] - X_t[np.newaxis, :, :])
            return np.sum(diff, axis=-1)
        elif metric == "euclidean":
            diff = X_s[:, np.newaxis, :] - X_t[np.newaxis, :, :]
            return np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
        else:  # sqeuclidean
            diff = X_s[:, np.newaxis, :] - X_t[np.newaxis, :, :]
            return np.sum(diff**2, axis=-1)

    def fit(
        self,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        X_source: Optional[np.ndarray] = None,
        X_target: Optional[np.ndarray] = None,
        metric: str = "sqeuclidean",
    ) -> SinkhornOptimalTransport:
        """Solves the Sinkhorn optimal transport problem.

        Parameters
        ----------
        a : Optional[np.ndarray]
            Source distribution histogram of shape (N,). Defaults to uniform distribution.
        b : Optional[np.ndarray]
            Target distribution histogram of shape (M,). Defaults to uniform distribution.
        cost_matrix : Optional[np.ndarray]
            Precomputed pairwise ground cost matrix of shape (N, M).
        X_source : Optional[np.ndarray]
            Source feature coordinates (N, D).
        X_target : Optional[np.ndarray]
            Target feature coordinates (M, D).
        metric : str, default="sqeuclidean"
            Ground distance metric if cost_matrix is not supplied.
        """
        if cost_matrix is None:
            if X_source is None or X_target is None:
                raise ValueError(
                    "Either cost_matrix or (X_source, X_target) must be provided."
                )
            C = self._compute_cost(
                np.asarray(X_source), np.asarray(X_target), metric=metric
            )
        else:
            C = np.asarray(cost_matrix, dtype=np.float64)

        N, M = C.shape

        # Default to uniform marginal distributions if not provided
        a_vec = (
            np.ones(N, dtype=np.float64) / N
            if a is None
            else np.asarray(a, dtype=np.float64)
        )
        b_vec = (
            np.ones(M, dtype=np.float64) / M
            if b is None
            else np.asarray(b, dtype=np.float64)
        )

        # Normalize marginals
        a_vec = a_vec / np.sum(a_vec)
        b_vec = b_vec / np.sum(b_vec)

        # Gibbs Kernel K = exp(-C / eps)
        K = np.exp(-C / self.reg)

        # Initialize scaling vectors u and v
        u = np.ones(N, dtype=np.float64) / N
        v = np.ones(M, dtype=np.float64) / M

        for _ in range(self.max_iter):
            u_prev = u.copy()

            # Sinkhorn updates: u = a / (K @ v), v = b / (K^T @ u)
            Kv = np.matmul(K, v) + 1e-12
            u = a_vec / Kv

            KTu = np.matmul(K.T, u) + 1e-12
            v = b_vec / KTu

            # Convergence check
            err: float = float(np.max(np.abs(u - u_prev)))
            if err < self.tol:
                break

        # Optimal transport plan: P = diag(u) @ K @ diag(v)
        P = u[:, np.newaxis] * K * v[np.newaxis, :]
        self.plan_ = P

        # Regularized Wasserstein distance
        self.distance_ = float(np.sum(P * C))
        return self


def sinkhorn_distance(
    X_source: np.ndarray,
    X_target: np.ndarray,
    reg: float = 0.05,
    metric: str = "sqeuclidean",
) -> float:
    """Convenience function calculating regularized Wasserstein distance between two point clouds."""
    ot = SinkhornOptimalTransport(reg=reg)
    ot.fit(X_source=X_source, X_target=X_target, metric=metric)
    return ot.distance_
