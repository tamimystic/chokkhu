"""Debiased Sinkhorn Divergences and Entropic Optimal Transport.

Formulated from first principles using log-domain stabilized dual potential iterations
and positive-definite debiased Sinkhorn metrics in pure NumPy.
"""

from typing import Optional, Tuple

import numpy as np


class SinkhornDivergence:
    r"""Debiased Sinkhorn Divergence S_eps(alpha, beta).

        Eliminates entropic bias of regularized Optimal Transport, guaranteeing positive definiteness
        and metric faithfulness S_eps(alpha, alpha) = 0:
        S_\epsilon(lpha, eta) = 	ext{OT}_\epsilon(lpha, eta) -
    rac{1}{2} 	ext{OT}_\epsilon(lpha, lpha) -
    rac{1}{2} 	ext{OT}_\epsilon(eta, eta)

        Parameters
        ----------
        epsilon : float, default=0.1
            Entropic regularization coefficient (temperature).
        max_iter : int, default=100
            Maximum Sinkhorn fixed-point iterations.
        tol : float, default=1e-6
            Dual potential convergence tolerance.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        self.epsilon = max(1e-5, float(epsilon))
        self.max_iter = max(10, int(max_iter))
        self.tol = float(tol)

    def _sinkhorn_loss(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute regularized OT cost OT_eps(a, b) and optimal dual potentials (f, g)."""
        eps = self.epsilon
        n, m = C.shape

        # Initialize dual potentials in log domain: f in R^n, g in R^m
        f = np.zeros(n, dtype=float)
        g = np.zeros(m, dtype=float)

        log_a = np.log(np.maximum(a, 1e-12))
        log_b = np.log(np.maximum(b, 1e-12))

        for _ in range(self.max_iter):
            f_prev = f.copy()

            # Softmin updates in log-domain:
            # f_i = -eps * logsumexp((g_j - C_ij)/eps + log b_j)
            M_f = (g[None, :] - C) / eps + log_b[None, :]
            max_Mf = np.max(M_f, axis=-1, keepdims=True)
            f = -eps * (
                np.log(np.sum(np.exp(M_f - max_Mf), axis=-1)) + max_Mf.squeeze(-1)
            )

            M_g = (f[:, None] - C) / eps + log_a[:, None]
            max_Mg = np.max(M_g, axis=0, keepdims=True)
            g = -eps * (
                np.log(np.sum(np.exp(M_g - max_Mg), axis=0)) + max_Mg.squeeze(0)
            )

            if np.max(np.abs(f - f_prev)) < self.tol:
                break

        # Dual value: <f, a> + <g, b>
        ot_cost = float(np.dot(f, a) + np.dot(g, b))
        return ot_cost, f, g

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> float:
        """Compute debiased Sinkhorn divergence S_eps(X, Y) between two point clouds.

        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            First point cloud.
        Y : np.ndarray of shape (M, d)
            Second point cloud.
        a : Optional[np.ndarray] of shape (N,), default=None
            Probability weights for X (uniform if None).
        b : Optional[np.ndarray] of shape (M,), default=None
            Probability weights for Y (uniform if None).

        Returns
        -------
        divergence : float
            Unbiased Sinkhorn divergence S_eps(X, Y) >= 0.
        """
        X_mat = np.atleast_2d(np.asarray(X, dtype=float))
        Y_mat = np.atleast_2d(np.asarray(Y, dtype=float))

        N = X_mat.shape[0]
        M = Y_mat.shape[0]

        a_vec = (
            np.ones(N, dtype=float) / N
            if a is None
            else np.asarray(a, dtype=float) / np.sum(a)
        )
        b_vec = (
            np.ones(M, dtype=float) / M
            if b is None
            else np.asarray(b, dtype=float) / np.sum(b)
        )

        # 1. Cross cost matrix C_XY
        diff_XY = X_mat[:, None, :] - Y_mat[None, :, :]
        C_XY = np.sum(diff_XY**2, axis=-1)

        # 2. Self cost matrices C_XX and C_YY
        diff_XX = X_mat[:, None, :] - X_mat[None, :, :]
        C_XX = np.sum(diff_XX**2, axis=-1)

        diff_YY = Y_mat[:, None, :] - Y_mat[None, :, :]
        C_YY = np.sum(diff_YY**2, axis=-1)

        # 3. Compute 3 OT terms
        ot_XY, _, _ = self._sinkhorn_loss(a_vec, b_vec, C_XY)
        ot_XX, _, _ = self._sinkhorn_loss(a_vec, a_vec, C_XX)
        ot_YY, _, _ = self._sinkhorn_loss(b_vec, b_vec, C_YY)

        # 4. Debiased divergence: OT(X, Y) - 0.5 * OT(X, X) - 0.5 * OT(Y, Y)
        div = ot_XY - 0.5 * ot_XX - 0.5 * ot_YY
        return float(max(0.0, div))

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> float:
        """Alias for compute(X, Y, a, b)."""
        return self.compute(X, Y, a, b)
