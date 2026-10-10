from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..base import ChokkhuModel


class DBSCAN(ChokkhuModel):
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        metric_params: Optional[dict] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = 2.0,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.labels_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DBSCAN:
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=np.int64)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X_arr, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self._expand_cluster(X_arr, i, neighbors, cluster_id, visited)
                cluster_id += 1

        self.is_fitted = True
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_

    def _expand_cluster(
        self,
        X: np.ndarray,
        point_idx: int,
        neighbors: List[int],
        cluster_id: int,
        visited: np.ndarray,
    ) -> None:
        if self.labels_ is None:
            return
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            i += 1

    def _region_query(self, X: np.ndarray, point_idx: int) -> List[int]:
        diff = X - X[point_idx]
        if self.metric == "manhattan" or self.metric == "cityblock":
            distances = np.sum(np.abs(diff), axis=1)
        elif self.metric == "cosine":
            norm1 = np.linalg.norm(X, axis=1)
            norm2 = np.linalg.norm(X[point_idx])
            denom = np.maximum(norm1 * norm2, 1e-12)
            distances = 1.0 - (X @ X[point_idx]) / denom
        else:
            distances = np.linalg.norm(diff, axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DBSCAN does not support prediction for new samples. Use .labels_ instead."
        )
