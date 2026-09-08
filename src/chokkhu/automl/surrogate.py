"""Gaussian Process Regression Surrogate and Acquisition Functions for Bayesian Optimization."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.stats import norm


class GaussianProcessSurrogate:
    """Gaussian Process Regression with RBF Kernel from First Principles."""

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise_level: float = 1e-4,
    ) -> None:
        self.length_scale = length_scale
        self.variance = variance
        self.noise_level = noise_level
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.K_inv: np.ndarray = np.array([])

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF (Squared Exponential) kernel matrix."""
        # Pairwise squared Euclidean distances
        dists = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1, keepdims=True).T
            - 2.0 * np.matmul(X1, X2.T)
        )
        dists = np.maximum(0.0, dists)
        return self.variance * np.exp(-0.5 * dists / (self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessSurrogate:
        """Fit GP on observed parameter points and metric values."""
        self.X_train = np.asarray(X, dtype=np.float64)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = np.asarray(y, dtype=np.float64).flatten()

        N = self.X_train.shape[0]
        K = self._rbf_kernel(self.X_train, self.X_train) + self.noise_level * np.eye(N)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)

        return self

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior mean mu and variance sigma^2 at query points X_star."""
        X_s = np.asarray(X_star, dtype=np.float64)
        if X_s.ndim == 1:
            X_s = X_s.reshape(-1, 1)

        if len(self.X_train) == 0:
            return np.zeros(len(X_s)), np.ones(len(X_s)) * self.variance

        K_s = self._rbf_kernel(self.X_train, X_s)  # (N, N_star)
        K_ss = self._rbf_kernel(X_s, X_s)  # (N_star, N_star)

        # Posterior mean: mu = K_s.T @ K_inv @ y
        mu = np.matmul(np.matmul(K_s.T, self.K_inv), self.y_train)

        # Posterior covariance: sigma^2 = diag(K_ss - K_s.T @ K_inv @ K_s)
        cov = K_ss - np.matmul(np.matmul(K_s.T, self.K_inv), K_s)
        sigma2 = np.maximum(1e-8, np.diag(cov))

        return mu, sigma2


def expected_improvement(
    mu: np.ndarray,
    sigma2: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement (EI) Acquisition Function for Maximization."""
    sigma = np.sqrt(np.maximum(1e-9, sigma2))
    improvement = mu - best_y - xi

    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-8] = 0.0
    return ei


def upper_confidence_bound(
    mu: np.ndarray,
    sigma2: np.ndarray,
    kappa: float = 2.576,
) -> np.ndarray:
    """Upper Confidence Bound (UCB) Acquisition Function for Maximization."""
    sigma = np.sqrt(np.maximum(1e-9, sigma2))
    return mu + kappa * sigma
