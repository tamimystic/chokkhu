"""Information Theory, KSG Continuous Mutual Information, and Multivariate KDE.

Formulated from first principles using Kraskov-Stögbauer-Grassberger (KSG) k-NN non-parametric
digamma statistics and multivariate kernel density estimation in pure NumPy and SciPy.
"""

from typing import Optional, Union

import numpy as np
from scipy.special import digamma


class KraskovMutualInformation:
    r"""Kraskov-Stögbauer-Grassberger (KSG) Non-Parametric Mutual Information Estimator.

        Computes continuous mutual information I(X; Y) between multidimensional variables:
        I(X; Y) = \psi(k) -
    rac{1}{N} \sum_{i=1}^N \left[ \psi(n_x(i) + 1) + \psi(n_y(i) + 1)
    ight] + \psi(N)

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors in joint space (Z = (X, Y)).
        noise_level : float, default=1e-10
            Minimal jitter added to eliminate degenerate zero-distance ties.
    """

    def __init__(self, k: int = 3, noise_level: float = 1e-10) -> None:
        self.k = max(1, int(k))
        self.noise_level = float(noise_level)

    def estimate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        seed: int = 42,
    ) -> float:
        """Estimate mutual information I(X; Y) in nats.

        Parameters
        ----------
        X : np.ndarray of shape (N, d_x) or (N,)
            Continuous random variable samples X.
        Y : np.ndarray of shape (N, d_y) or (N,)
            Continuous random variable samples Y.
        seed : int, default=42
            Random seed for noise jittering.

        Returns
        -------
        mutual_info : float
            Estimated mutual information I(X; Y) >= 0.
        """
        x_mat = np.atleast_2d(np.asarray(X, dtype=float))
        if x_mat.shape[0] == 1 and x_mat.shape[1] > 1 and np.asarray(X).ndim == 1:
            x_mat = x_mat.T

        y_mat = np.atleast_2d(np.asarray(Y, dtype=float))
        if y_mat.shape[0] == 1 and y_mat.shape[1] > 1 and np.asarray(Y).ndim == 1:
            y_mat = y_mat.T

        N = x_mat.shape[0]
        if y_mat.shape[0] != N:
            raise ValueError(
                f"Sample size mismatch: X has {N} samples, Y has {y_mat.shape[0]} samples"
            )

        if N <= self.k:
            raise ValueError(f"Number of samples N={N} must be greater than k={self.k}")

        # Add tiny jitter to avoid distance collisions
        rng = np.random.RandomState(seed)
        if self.noise_level > 0.0:
            x_mat = x_mat + self.noise_level * rng.randn(*x_mat.shape)
            y_mat = y_mat + self.noise_level * rng.randn(*y_mat.shape)

        # 1. Compute pairwise Chebyshev (L_infinity) distance matrices
        # dx_mat[i, j] = max_d |x_i,d - x_j,d|
        dx_mat = np.max(np.abs(x_mat[:, None, :] - x_mat[None, :, :]), axis=-1)
        dy_mat = np.max(np.abs(y_mat[:, None, :] - y_mat[None, :, :]), axis=-1)

        # Joint distance dz = max(dx, dy)
        dz_mat = np.maximum(dx_mat, dy_mat)

        # 2. For each point i, find distance to k-th nearest neighbor in joint space
        # Sort along rows and extract k-th neighbor distance (excluding self at index 0)
        sorted_dz = np.sort(dz_mat, axis=1)
        eps_vec = sorted_dz[:, self.k]  # eps(i) is strictly k-th neighbor distance

        # 3. Count points in marginal spaces strictly within eps(i)
        # n_x(i) = sum_{j != i} I(dx(i, j) < eps(i))
        # n_y(i) = sum_{j != i} I(dy(i, j) < eps(i))
        nx_counts = np.sum(dx_mat < eps_vec[:, None], axis=1) - 1
        ny_counts = np.sum(dy_mat < eps_vec[:, None], axis=1) - 1

        # 4. KSG digamma formula (estimator 1)
        psi_k = digamma(self.k)
        psi_N = digamma(N)
        mean_psi_marginals = np.mean(digamma(nx_counts + 1) + digamma(ny_counts + 1))

        mi = psi_k - mean_psi_marginals + psi_N
        return float(max(0.0, mi))


class MultivariateKDE:
    r"""Non-Parametric Multivariate Continuous Kernel Density Estimator.

        p(x) =
    rac{1}{N h^d} \sum_{i=1}^N K\left(
    rac{x - x_i}{h}
    ight)

        Parameters
        ----------
        bandwidth : Union[str, float], default="silverman"
            Bandwidth parameter h or rule-of-thumb heuristic ("silverman" or "scott").
        kernel : str, default="gaussian"
            Kernel function: "gaussian", "epanechnikov", or "box".
    """

    def __init__(
        self,
        bandwidth: Union[str, float] = "silverman",
        kernel: str = "gaussian",
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data_: Optional[np.ndarray] = None
        self.h_: float = 1.0
        self.dim_: int = 1

    def fit(self, X: np.ndarray) -> "MultivariateKDE":
        """Fit kernel density estimator from training data X."""
        X_mat = np.atleast_2d(np.asarray(X, dtype=float))
        if X_mat.shape[0] == 1 and X_mat.shape[1] > 1 and np.asarray(X).ndim == 1:
            X_mat = X_mat.T

        self.data_ = X_mat
        N, d = X_mat.shape
        self.dim_ = d

        # Bandwidth selection
        if isinstance(self.bandwidth, str):
            std_dev = float(np.mean(np.std(X_mat, axis=0)) + 1e-6)
            if self.bandwidth == "silverman":
                # Silverman's rule: sigma * (4 / (d + 2))^(1 / (d + 4)) * N^(-1 / (d + 4))
                factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
                self.h_ = float(std_dev * factor * (N ** (-1.0 / (d + 4.0))))
            elif self.bandwidth == "scott":
                # Scott's rule: sigma * N^(-1 / (d + 4))
                self.h_ = float(std_dev * (N ** (-1.0 / (d + 4.0))))
            else:
                self.h_ = 1.0
        else:
            self.h_ = max(1e-6, float(self.bandwidth))

        return self

    def _eval_kernel(self, u_sq: np.ndarray) -> np.ndarray:
        """Evaluate kernel on squared Mahalanobis / Euclidean distances u^2 = ||x - x_i||^2 / h^2."""
        d = self.dim_
        if self.kernel == "gaussian":
            norm_const = (2.0 * np.pi) ** (-d / 2.0)
            return norm_const * np.exp(-0.5 * u_sq)
        elif self.kernel == "epanechnikov":
            # Parabolic kernel: (d + 2) / (2 * V_d) * (1 - u^2) for u^2 <= 1
            # For simplicity, normalized on volume
            u = np.sqrt(u_sq)
            val = np.maximum(0.0, 1.0 - u_sq)
            return val * (0.75 / (self.h_**d))
        else:  # box / uniform
            u = np.sqrt(u_sq)
            return np.where(u <= 1.0, 1.0, 0.0)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute log-density log p(x) for test points."""
        if self.data_ is None:
            raise ValueError("KDE model has not been fitted yet.")

        X_eval = np.atleast_2d(np.asarray(X, dtype=float))
        if X_eval.shape[0] == 1 and X_eval.shape[1] > 1 and np.asarray(X).ndim == 1:
            X_eval = X_eval.T

        N, d = self.data_.shape
        h = self.h_

        # Pairwise squared normalized distance: (M, N)
        diff = X_eval[:, None, :] - self.data_[None, :, :]
        u_sq = np.sum((diff / h) ** 2, axis=-1)

        k_vals = self._eval_kernel(u_sq)  # (M, N)
        densities = np.sum(k_vals, axis=1) / (N * (h**d) + 1e-12)
        densities = np.maximum(densities, 1e-300)
        return np.log(densities)

    def sample(self, n_samples: int = 100, seed: int = 42) -> np.ndarray:
        """Draw random samples from the fitted multivariate KDE distribution."""
        if self.data_ is None:
            raise ValueError("KDE model has not been fitted yet.")

        N, d = self.data_.shape
        rng = np.random.RandomState(seed)

        # 1. Randomly select data point centers
        chosen_indices = rng.choice(N, size=n_samples, replace=True)
        centers = self.data_[chosen_indices]

        # 2. Add kernel perturbation noise
        if self.kernel == "gaussian":
            noise = rng.randn(n_samples, d) * self.h_
        else:
            noise = (rng.rand(n_samples, d) * 2.0 - 1.0) * self.h_

        return centers + noise
