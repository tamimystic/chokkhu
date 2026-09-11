"""Uniform Manifold Approximation and Projection (UMAP) in pure NumPy."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class UMAP:
    """Uniform Manifold Approximation and Projection (UMAP) in pure NumPy.

    Constructs a fuzzy topological representation of high-dimensional data
    and optimizes a low-dimensional layout using fuzzy set cross-entropy SGD.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        learning_rate: float = 1.0,
        n_epochs: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.rng = np.random.default_rng(random_state)

        # Smooth curve approximation parameters a and b
        self.a, self.b = self._find_ab_params(min_dist)
        self.embedding_: Optional[np.ndarray] = None

    def _find_ab_params(self, min_dist: float) -> Tuple[float, float]:
        """Approximates curve parameters a and b for 1 / (1 + a * d^(2b))."""
        # Closed-form power approximation matching McInnes et al.
        a = 1.5769 / (min_dist + 0.1) ** 0.5
        b = 0.8951 + 0.1 * min_dist
        return float(a), float(b)

    def _compute_fuzzy_simplicial_set(self, X: np.ndarray) -> np.ndarray:
        """Computes symmetric fuzzy neighborhood graph adjacency matrix P."""
        n_samples = len(X)
        k = min(self.n_neighbors, n_samples - 1)
        if k < 1:
            return np.zeros((n_samples, n_samples), dtype=np.float64)

        # Pairwise Euclidean distance matrix
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)

        P: np.ndarray = np.zeros((n_samples, n_samples), dtype=np.float64)
        target_sum = np.log2(k)

        for i in range(n_samples):
            row_dists = dists[i]
            sorted_indices = np.argsort(row_dists)
            knn_idx = sorted_indices[1 : k + 1]
            knn_dists = row_dists[knn_idx]

            rho_i = knn_dists[0]  # Distance to nearest neighbor

            # Binary search for bandwidth sigma_i
            sigma_low, sigma_high = 1e-4, 100.0
            sigma_i = 1.0

            for _ in range(20):
                sigma_mid = (sigma_low + sigma_high) / 2.0
                val: float = float(
                    np.sum(np.exp(-np.maximum(0.0, knn_dists - rho_i) / sigma_mid))
                )
                if val > target_sum:
                    sigma_high = sigma_mid
                else:
                    sigma_low = sigma_mid
                sigma_i = sigma_mid

            # Compute directed probabilities
            p_directed = np.exp(-np.maximum(0.0, knn_dists - rho_i) / (sigma_i + 1e-12))
            P[i, knn_idx] = p_directed

        # Symmetrize via fuzzy union: P = P + P^T - P * P^T
        P_sym = P + P.T - (P * P.T)
        return P_sym

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Projects high-dimensional matrix X to low-dimensional embedding."""
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)

        if n_samples <= self.n_components:
            return X_arr[:, : self.n_components]

        # 1. High-dimensional fuzzy graph
        P = self._compute_fuzzy_simplicial_set(X_arr)

        # 2. Spectral / Random initialization for low-dimensional layout
        # PCA-like SVD initialization
        X_centered = X_arr - np.mean(X_arr, axis=0)
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        Y = np.matmul(X_centered, vt[: self.n_components].T).astype(np.float64)
        Y = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-5)

        # 3. Layout Optimization via Cross-Entropy SGD
        lr = self.learning_rate
        a, b = self.a, self.b

        for epoch in range(self.n_epochs):
            alpha = lr * (1.0 - epoch / self.n_epochs)

            # Sample non-zero positive edges
            for i in range(n_samples):
                pos_neighbors = np.where(P[i] > 0.05)[0]
                if len(pos_neighbors) == 0:
                    continue

                for j in pos_neighbors:
                    p_ij = P[i, j]
                    diff = Y[i] - Y[j]
                    dist_sq: float = float(np.sum(diff**2))

                    # Attractive force: -2 b a d^(2(b-1)) / (1 + a d^2b)
                    q_ij = 1.0 / (1.0 + a * (dist_sq**b))
                    grad_coeff = (-2.0 * b * a * (dist_sq ** max(0.0, b - 1.0))) * q_ij
                    grad = p_ij * grad_coeff * diff

                    # Clip gradient for numerical stability
                    grad = np.clip(grad, -4.0, 4.0)

                    Y[i] += alpha * grad
                    Y[j] -= alpha * grad

                # Repulsive force from random negative samples
                neg_samples = self.rng.choice(
                    n_samples, size=min(4, n_samples), replace=False
                )
                for k_neg in neg_samples:
                    if k_neg == i:
                        continue
                    diff_neg = Y[i] - Y[k_neg]
                    dist_sq_neg = max(1e-4, float(np.sum(diff_neg**2)))

                    # Repulsive gradient: 2 b / ((0.001 + d^2) * (1 + a d^2b))
                    grad_neg_coeff = (2.0 * b) / (
                        (0.001 + dist_sq_neg) * (1.0 + a * (dist_sq_neg**b))
                    )
                    grad_neg = (1.0 - P[i, k_neg]) * grad_neg_coeff * diff_neg
                    grad_neg = np.clip(grad_neg, -4.0, 4.0)

                    Y[i] += alpha * grad_neg

        self.embedding_ = Y
        return Y

    def fit(self, X: np.ndarray) -> UMAP:
        """Fits UMAP layout to data."""
        self.fit_transform(X)
        return self
