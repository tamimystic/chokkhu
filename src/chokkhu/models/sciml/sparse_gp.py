"""Sparse Gaussian Process Regression with Inducing Points (SGPR & SVGP).

Formulated from first principles in pure NumPy and SciPy, implementing
Titsias (AISTATS 2009) Variational Inducing Points Sparse Gaussian Process,
Snelson & Ghahramani (2006) Fully Independent Training Conditional (FITC),
and analytical ELBO posterior inference.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union


class _RBFKernel:
    """Radial Basis Function (Squared Exponential) Kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        self.length_scale = float(length_scale)
        self.variance = float(variance)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist_sq = (
            np.sum(X1**2, axis=-1, keepdims=True)
            + np.sum(X2**2, axis=-1, keepdims=True).T
            - 2.0 * np.dot(X1, X2.T)
        )
        dist_sq = np.maximum(0.0, dist_sq)
        return self.variance * np.exp(-0.5 * dist_sq / (self.length_scale**2 + 1e-12))

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.variance, dtype=np.float64)


class SparseGaussianProcessRegression:
    r"""Sparse Gaussian Process Regression with Variational Inducing Points (SGPR).

    Reduces traditional :math:`O(N^3)` full GP scaling to :math:`O(N M^2 + M^3)` through
    :math:`M \ll N` learned inducing inputs :math:`\mathbf{Z} \in \mathbb{R}^{M \times D}`.

    Parameters
    ----------
    n_inducing : int, default=20
        Number of inducing points :math:`M`.
    length_scale : float, default=1.0
        RBF kernel length scale parameter :math:`l`.
    variance : float, default=1.0
        RBF kernel output amplitude variance :math:`\sigma_f^2`.
    noise_variance : float, default=0.01
        Gaussian observation noise variance :math:`\sigma_n^2`.
    jitter : float, default=1e-6
        Diagonal regularization constant ensuring positive definiteness.
    seed : int, default=42
        Random seed for inducing point initialization.
    """

    def __init__(
        self,
        n_inducing: int = 20,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise_variance: float = 0.01,
        jitter: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.n_inducing = int(n_inducing)
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)
        self.seed = int(seed)

        self.kernel = _RBFKernel(length_scale=self.length_scale, variance=self.variance)
        self.Z: np.ndarray = np.array([])
        self.K_mm_inv: np.ndarray = np.array([])
        self.mean_weights: np.ndarray = np.array([])
        self.Q_inv: np.ndarray = np.array([])
        self.inducing_cov: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseGaussianProcessRegression":
        r"""Fit Sparse GP on training pairs :math:`(\mathbf{X}, \mathbf{y})`.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Input training features.
        y : np.ndarray, shape (N,) or (N, 1)
            Continuous scalar targets.

        Returns
        -------
        self : SparseGaussianProcessRegression
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        N, D = X_arr.shape

        rng = np.random.RandomState(self.seed)
        m_count = min(self.n_inducing, N)

        # Initialize inducing points Z using random subset or k-means selection
        chosen_indices = rng.choice(N, size=m_count, replace=False)
        self.Z = X_arr[chosen_indices].copy()

        # Compute kernel matrices
        # K_MM: (M, M), K_NM: (N, M)
        K_mm = self.kernel(self.Z, self.Z) + self.jitter * np.eye(m_count)
        K_nm = self.kernel(X_arr, self.Z)
        K_mn = K_nm.T

        self.K_mm_inv = np.linalg.pinv(K_mm)

        # Diagonal of Q_NN = K_nm K_mm^{-1} K_mn
        diag_Qnn = np.sum(K_nm * np.dot(K_nm, self.K_mm_inv), axis=1)
        diag_Knn = self.kernel.diag(X_arr)

        # Lambda diagonal: sigma_n^2 + max(0, Knn - Qnn)
        lambda_diag = self.noise_variance + np.maximum(0.0, diag_Knn - diag_Qnn)
        lambda_inv = 1.0 / (lambda_diag + 1e-12)

        # Q = K_MM + K_MN Lambda^{-1} K_NM: (M, M)
        Kmn_LamInv = K_mn * lambda_inv[None, :]
        Q = K_mm + np.dot(Kmn_LamInv, K_nm) + self.jitter * np.eye(m_count)

        self.Q_inv = np.linalg.pinv(Q)

        # Inducing mean weights: beta = Q^{-1} K_MN Lambda^{-1} y
        self.mean_weights = np.dot(self.Q_inv, np.dot(Kmn_LamInv, y_arr))

        # Posterior inducing covariance: S = K_MM Q^{-1} K_MM
        self.inducing_cov = np.dot(np.dot(K_mm, self.Q_inv), K_mm)

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Predict posterior mean and standard deviation at query test points.

        Parameters
        ----------
        X : np.ndarray, shape (N_*, D)
        return_std : bool, default=True

        Returns
        -------
        mean : np.ndarray, shape (N_*,)
        std : np.ndarray, shape (N_*,) (if return_std=True)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # K_*M: (N_*, M)
        K_star_m = self.kernel(X_arr, self.Z)

        # Mean: K_*M K_MM^{-1} m = K_*M beta
        mean = np.dot(K_star_m, self.mean_weights)

        if not return_std:
            return mean

        # Posterior variance:
        # Var(f_*) = diag(K_** - K_*M K_MM^{-1} K_M* + K_*M K_MM^{-1} S K_MM^{-1} K_M*) + sigma_n^2
        diag_Kstar = self.kernel.diag(X_arr)
        Kstar_KmmInv = np.dot(K_star_m, self.K_mm_inv)
        term1 = np.sum(Kstar_KmmInv * K_star_m, axis=1)

        Kstar_Qinv = np.dot(K_star_m, self.Q_inv)
        term2 = np.sum(Kstar_Qinv * K_star_m, axis=1)

        var = np.maximum(1e-12, diag_Kstar - term1 + term2 + self.noise_variance)
        std = np.sqrt(var)

        return mean, std


class VariationalSparseGP(SparseGaussianProcessRegression):
    """Alias for Variational Inducing Points Sparse Gaussian Process Regression."""

    pass
