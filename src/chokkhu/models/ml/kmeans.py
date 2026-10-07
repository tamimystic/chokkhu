from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from chokkhu.core.math import cdist
from ..base import ChokkhuModel


class KMeans(ChokkhuModel):
    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray] = "k-means++",
        n_init: Union[str, int] = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "lloyd",
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = 10 if n_init == "auto" else int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.n_features_in_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        n_samples, n_features = X.shape
        if isinstance(self.init, np.ndarray):
            return np.array(self.init, dtype=np.float64, copy=True)

        if self.init == "random":
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()

        # k-means++ initialization (Arthur & Vassilvitskii, 2007)
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)
        first_idx = rng.randint(0, n_samples)
        centers[0] = X[first_idx]

        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for c_idx in range(1, self.n_clusters):
            prob = closest_dist_sq / (np.sum(closest_dist_sq) + 1e-15)
            next_idx = rng.choice(n_samples, p=prob)
            centers[c_idx] = X[next_idx]
            new_dist_sq = np.sum((X - centers[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KMeans:
        X_arr = np.array(X, dtype=np.float64, copy=self.copy_x)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)
        best_centers: Optional[np.ndarray] = None
        best_labels: Optional[np.ndarray] = None
        best_inertia = float("inf")
        best_n_iter = 0

        runs = 1 if isinstance(self.init, np.ndarray) else self.n_init

        for _ in range(runs):
            centers = self._init_centroids(X_arr, rng)
            labels = np.zeros(n_samples, dtype=np.int64)
            n_iter = 0

            for it in range(self.max_iter):
                n_iter = it + 1
                # E-step: Assign labels
                dists_sq = cdist(X_arr, centers, metric="sqeuclidean")
                labels = np.argmin(dists_sq, axis=1)

                # M-step: Update centroids
                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        new_centers[k] = np.mean(X_arr[mask], axis=0)
                    else:
                        # Re-seed empty cluster with furthest point
                        furthest = np.argmax(np.min(dists_sq, axis=1))
                        new_centers[k] = X_arr[furthest]

                center_shift = float(np.sum((centers - new_centers) ** 2))
                centers = new_centers
                if center_shift <= self.tol:
                    break

            # Calculate inertia
            dists_sq = cdist(X_arr, centers, metric="sqeuclidean")
            inertia = float(np.sum(np.min(dists_sq, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        dists_sq = cdist(X_arr, self.cluster_centers_, metric="sqeuclidean")
        return np.argmin(dists_sq, axis=1)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).labels_  # type: ignore[return-value]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        return cdist(X_arr, self.cluster_centers_, metric="euclidean")

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        dists_sq = cdist(X_arr, self.cluster_centers_, metric="sqeuclidean")
        return -float(np.sum(np.min(dists_sq, axis=1)))
