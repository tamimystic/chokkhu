"""Wasserstein Barycenter Computation via Iterative Sinkhorn in pure NumPy."""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


class WassersteinBarycenter:
    r"""Wasserstein Geometric Barycenter (Frechet Mean) across Multiple Distributions.

    Computes the geometric mean distribution $\mu^*$ minimizing weighted
    Wasserstein distances to a collection of distributions (Agueh & Carlier 2011).

    Parameters
    ----------
    reg : float, default=0.01
        Entropic regularization strength.
    max_iter : int, default=50
        Maximum number of iterative barycentric update steps.
    tol : float, default=1e-5
        Tolerance for convergence.
    """

    def __init__(
        self,
        reg: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> None:
        self.reg = float(reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.barycenter_: Optional[np.ndarray] = None

    def compute(
        self,
        distributions: Sequence[np.ndarray],
        cost_matrix: np.ndarray,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Computes the discrete Wasserstein barycenter histogram.

        Parameters
        ----------
        distributions : Sequence[np.ndarray]
            List of K 1D probability mass functions, each of shape (N,).
        cost_matrix : np.ndarray
            Pairwise distance cost matrix of shape (N, N).
        weights : Optional[Sequence[float]]
            Weights lambda_k for each distribution (must sum to 1.0).

        Returns
        -------
        np.ndarray
            Optimal barycenter histogram of shape (N,).
        """
        K_dists = len(distributions)
        N = len(cost_matrix)

        if weights is None:
            w = np.ones(K_dists, dtype=np.float64) / K_dists
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / np.sum(w)

        # Normalize input distributions
        dists = [np.asarray(d, dtype=np.float64) / np.sum(d) for d in distributions]

        # Gibbs kernel
        K = np.exp(-cost_matrix / self.reg)

        # Initialize scaling matrices (K_dists, N)
        v: np.ndarray = np.ones((K_dists, N), dtype=np.float64)
        u: np.ndarray = np.ones((K_dists, N), dtype=np.float64)

        barycenter = np.ones(N, dtype=np.float64) / N

        for _ in range(self.max_iter):
            prev_bary = barycenter.copy()

            # 1. Update u_k = p_k / (K @ v_k)
            for k in range(K_dists):
                Kv = np.matmul(K, v[k]) + 1e-12
                u[k] = dists[k] / Kv

            # 2. Geometric barycenter update: mu = prod_k (K^T @ u_k)^w_k
            log_bary: np.ndarray = np.zeros(N, dtype=np.float64)
            for k in range(K_dists):
                KTu = np.matmul(K.T, u[k]) + 1e-12
                log_bary += w[k] * np.log(KTu)

            barycenter = np.exp(log_bary)
            barycenter = barycenter / (np.sum(barycenter) + 1e-12)

            # 3. Update v_k = mu / (K^T @ u_k)
            for k in range(K_dists):
                KTu = np.matmul(K.T, u[k]) + 1e-12
                v[k] = barycenter / KTu

            if np.max(np.abs(barycenter - prev_bary)) < self.tol:
                break

        self.barycenter_ = barycenter
        return barycenter


def wasserstein_barycenter(
    distributions: Sequence[np.ndarray],
    cost_matrix: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    reg: float = 0.01,
) -> np.ndarray:
    """Convenience function computing Wasserstein barycenter across histograms."""
    wb = WassersteinBarycenter(reg=reg)
    return wb.compute(distributions, cost_matrix, weights=weights)
