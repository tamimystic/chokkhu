from __future__ import annotations

from typing import Any, Optional

import numpy as np

from chokkhu.core.math import cdist
from chokkhu.models.base import ChokkhuModel


class KMedoids(ChokkhuModel):
    def __init__(
        self,
        n_clusters: int = 8,
        metric: str = "euclidean",
        method: str = "pam",
        init: str = "build",
        max_iter: int = 300,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.medoid_indices_: np.ndarray = np.array([], dtype=np.int64)
        self.medoids_: np.ndarray = np.array([], dtype=np.float64)
        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.inertia_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KMedoids:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        k = min(self.n_clusters, n_samples)
        dist_matrix = cdist(x_arr, x_arr, metric=self.metric)

        if self.init == "random":
            medoid_indices = self.rng.choice(n_samples, size=k, replace=False).tolist()
        else:
            medoid_indices = [int(np.argmin(np.sum(dist_matrix, axis=1)))]
            for _ in range(1, k):
                curr_min = np.min(dist_matrix[:, medoid_indices], axis=1)
                best_gain, best_c = -1.0, -1
                for c in range(n_samples):
                    if c in medoid_indices:
                        continue
                    gain = float(np.sum(np.maximum(0.0, curr_min - dist_matrix[:, c])))
                    if gain > best_gain:
                        best_gain, best_c = gain, c
                medoid_indices.append(
                    best_c if best_c != -1 else int(self.rng.choice(n_samples))
                )

        medoids_arr = np.array(medoid_indices, dtype=np.int64)

        for _ in range(self.max_iter):
            improved = False
            best_cost = float(np.sum(np.min(dist_matrix[:, medoids_arr], axis=1)))
            for m_idx in range(k):
                old_medoid = medoids_arr[m_idx]
                for candidate in range(n_samples):
                    if candidate in medoids_arr:
                        continue
                    medoids_arr[m_idx] = candidate
                    cost = float(np.sum(np.min(dist_matrix[:, medoids_arr], axis=1)))
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        improved = True
                        break
                    else:
                        medoids_arr[m_idx] = old_medoid
                if improved:
                    break
            if not improved:
                break

        self.medoid_indices_ = medoids_arr
        self.medoids_ = x_arr[medoids_arr]
        dists_to_medoids = dist_matrix[:, medoids_arr]
        self.labels_ = np.argmin(dists_to_medoids, axis=1)
        self.inertia_ = float(np.sum(np.min(dists_to_medoids, axis=1)))
        self.is_fitted = True
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("KMedoids is not fitted yet.")
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.medoids_, metric=self.metric)
        return np.argmin(dists, axis=1)
