"""K-Medoids Clustering & Partitioning Around Medoids (PAM) in Pure NumPy.

References:
- Kaufman & Rousseeuw (1990): "Partitioning Around Medoids (Program PAM)" (Wiley).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


class KMedoids(ChokkhuModel):
    """Partitioning Around Medoids (PAM) K-Medoids Clustering with arbitrary distance metrics."""

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.medoid_indices_: np.ndarray = np.array([], dtype=np.int64)
        self.medoids_: np.ndarray = np.array([], dtype=np.float64)
        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.inertia_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KMedoids:
        """Fit K-Medoids by finding exemplar points minimizing sum of dissimilarities."""
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        # Compute full pairwise distance matrix
        dist_matrix = cdist(x_arr, x_arr, metric=self.metric)

        # 1. BUILD step: greedy initial medoid selection
        medoid_indices: List[int] = []
        first_medoid = int(np.argmin(np.sum(dist_matrix, axis=1)))
        medoid_indices.append(first_medoid)

        for _ in range(1, self.n_clusters):
            curr_min_dists = np.min(dist_matrix[:, medoid_indices], axis=1)
            best_gain: float = -1.0
            best_candidate: int = -1

            for candidate in range(n_samples):
                if candidate in medoid_indices:
                    continue
                gain: float = float(
                    np.sum(np.maximum(0.0, curr_min_dists - dist_matrix[:, candidate]))
                )
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            if best_candidate != -1:
                medoid_indices.append(best_candidate)
            else:
                medoid_indices.append(int(self.rng.choice(n_samples)))

        medoids_arr = np.array(medoid_indices, dtype=np.int64)

        # 2. SWAP step: iteratively swap medoid with non-medoid if cost decreases
        for _ in range(self.max_iter):
            improved = False
            best_swap_cost: float = float(
                np.sum(np.min(dist_matrix[:, medoids_arr], axis=1))
            )
            best_swap = None

            for i_idx, _ in enumerate(medoids_arr):
                for candidate in range(n_samples):
                    if candidate in medoids_arr:
                        continue

                    # Try swap
                    test_medoids = np.copy(medoids_arr)
                    test_medoids[i_idx] = candidate
                    cost: float = float(
                        np.sum(np.min(dist_matrix[:, test_medoids], axis=1))
                    )

                    if cost < best_swap_cost:
                        best_swap_cost = cost
                        best_swap = (i_idx, candidate)

            if best_swap is not None:
                i_idx_swap, candidate_swap = best_swap
                medoids_arr[i_idx_swap] = candidate_swap
                improved = True

            if not improved:
                break

        self.medoid_indices_ = medoids_arr
        self.medoids_ = x_arr[medoids_arr]
        self.labels_ = np.argmin(dist_matrix[:, medoids_arr], axis=1)
        self.inertia_ = float(np.sum(np.min(dist_matrix[:, medoids_arr], axis=1)))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to nearest medoid."""
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.medoids_, metric=self.metric)
        return np.argmin(dists, axis=1)
