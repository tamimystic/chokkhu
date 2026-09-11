"""Spectral Graph Partitioning, Normalized Graph Laplacians, and Fiedler Clustering.

Formulated from first principles using symmetric/random-walk graph Laplacians, Fiedler vector bisection,
and Ng-Jordan-Weiss multi-way eigenvector embedding in pure NumPy and SciPy.
"""

from typing import Optional

import numpy as np


class SpectralGraphClusterer:
    r"""Spectral Graph Partitioning and Multi-Way Graph Clustering.

    Minimizes Normalized Cut (NCut) via graph Laplacian spectral embedding:
    L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}, \quad L_{\text{sym}} v_k = \lambda_k v_k

    Parameters
    ----------
    n_clusters : int, default=2
        Target number of clusters K.
    laplacian_type : str, default="symmetric"
        Graph Laplacian formulation: "symmetric" (L_sym), "random_walk" (L_rw), or "unnormalized" (L).
    affinity : str, default="precomputed"
        Affinity type: "precomputed" (adjacency matrix) or "rbf" (Gaussian kernel on features).
    gamma : float, default=1.0
        RBF affinity scale parameter.
    max_iter : int, default=100
        Maximum iterations for final K-means embedding assignment.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        laplacian_type: str = "symmetric",
        affinity: str = "precomputed",
        gamma: float = 1.0,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_clusters = max(2, int(n_clusters))
        self.laplacian_type = laplacian_type
        self.affinity = affinity
        self.gamma = float(gamma)
        self.max_iter = max(10, int(max_iter))
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.labels_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None

    def _compute_affinity(self, X: np.ndarray) -> np.ndarray:
        """Compute affinity / adjacency matrix."""
        if self.affinity == "precomputed":
            A = np.asarray(X, dtype=float)
            # Symmetrize
            return 0.5 * (A + A.T)
        else:
            diff = X[:, None, :] - X[None, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            A = np.exp(-self.gamma * dist_sq)
            np.fill_diagonal(A, 0.0)
            return A

    def _kmeans_cluster(self, U: np.ndarray) -> np.ndarray:
        """Simple Lloyd's K-means on spectral coordinates U."""
        N, K = U.shape
        # Initialize centers randomly from points
        init_idx = self.rng.choice(N, size=self.n_clusters, replace=False)
        centers = U[init_idx].copy()

        labels = np.zeros(N, dtype=int)

        for _ in range(self.max_iter):
            # Assign points to closest center
            dists = np.sum((U[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
            new_labels = np.argmin(dists, axis=-1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Recompute centers
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    centers[k] = np.mean(U[mask], axis=0)

        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit spectral graph clusterer and return discrete cluster labels."""
        A = self._compute_affinity(X)
        N = A.shape[0]

        # 1. Degree vector and matrix D
        d = np.sum(A, axis=-1)
        d_safe = np.maximum(d, 1e-12)

        # 2. Construct Graph Laplacian
        if self.laplacian_type == "symmetric":
            # L_sym = I - D^{-1/2} A D^{-1/2}
            d_inv_sqrt = 1.0 / np.sqrt(d_safe)
            L = np.eye(N) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
        elif self.laplacian_type == "random_walk":
            # L_rw = I - D^{-1} A
            L = np.eye(N) - (A / d_safe[:, None])
        else:  # unnormalized
            L = np.diag(d) - A

        # 3. Eigen-decomposition (Hermitian / Symmetric)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (L + L.T))
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        self.eigenvalues_ = eigvals[: self.n_clusters]
        self.eigenvectors_ = eigvecs[:, : self.n_clusters]

        # 4. For K=2, Fiedler vector bisection can directly partition
        if self.n_clusters == 2:
            fiedler_vec = eigvecs[:, 1]  # 2nd smallest eigenvector
            threshold = float(np.median(fiedler_vec))
            labels = (fiedler_vec > threshold).astype(int)
        else:
            # Ng-Jordan-Weiss row-normalized spectral coordinates
            U = eigvecs[:, : self.n_clusters]
            row_norms = np.linalg.norm(U, axis=-1, keepdims=True) + 1e-12
            T_mat = U / row_norms
            labels = self._kmeans_cluster(T_mat)

        self.labels_ = labels
        return labels

    def normalized_cut(self, A: np.ndarray) -> float:
        """Compute Normalized Cut (NCut) value for current clustering."""
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")

        A_mat = self._compute_affinity(A)
        ncut_val: float = 0.0

        for k in range(self.n_clusters):
            mask_k = self.labels_ == k
            mask_not_k = ~mask_k

            if not np.any(mask_k) or not np.any(mask_not_k):
                continue

            cut_k = float(np.sum(A_mat[mask_k][:, mask_not_k]))
            vol_k = float(np.sum(A_mat[mask_k])) + 1e-12
            ncut_val += cut_k / vol_k

        return ncut_val
