from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ..base import ChokkhuModel


class HierarchicalClustering(ChokkhuModel):
    def __init__(
        self,
        n_clusters: Optional[int] = 2,
        metric: str = "euclidean",
        affinity: str = "deprecated",
        memory: Any = None,
        connectivity: Any = None,
        compute_full_tree: Union[str, bool] = "auto",
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.metric = affinity if affinity != "deprecated" else metric
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.labels_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> HierarchicalClustering:
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        self.labels_ = np.arange(n_samples)

        distances = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                diff = X_arr[i] - X_arr[j]
                if self.metric in ("manhattan", "cityblock"):
                    d = float(np.sum(np.abs(diff)))
                elif self.metric == "cosine":
                    d = float(
                        1.0
                        - np.dot(X_arr[i], X_arr[j])
                        / max(
                            np.linalg.norm(X_arr[i]) * np.linalg.norm(X_arr[j]), 1e-12
                        )
                    )
                else:
                    d = float(np.linalg.norm(diff))
                distances[i, j] = distances[j, i] = d

        np.fill_diagonal(distances, np.inf)
        current_clusters = n_samples
        target_clusters = self.n_clusters if self.n_clusters is not None else 1

        while current_clusters > target_clusters:
            min_idx = np.argmin(distances)
            unraveled = np.unravel_index(min_idx, distances.shape)
            i, j = int(unraveled[0]), int(unraveled[1])
            min_dist = distances[i, j]

            if (
                self.distance_threshold is not None
                and min_dist > self.distance_threshold
            ):
                break

            cluster_i = self.labels_[i]
            cluster_j = self.labels_[j]
            self.labels_[self.labels_ == cluster_j] = cluster_i

            for k in range(n_samples):
                if k != i and k != j:
                    if self.linkage == "complete":
                        new_dist = max(distances[i, k], distances[j, k])
                    elif self.linkage == "average":
                        new_dist = 0.5 * (distances[i, k] + distances[j, k])
                    else:
                        new_dist = min(distances[i, k], distances[j, k])
                    distances[i, k] = distances[k, i] = new_dist

            distances[j, :] = np.inf
            distances[:, j] = np.inf
            current_clusters -= 1

        unique_labels = np.unique(self.labels_)
        new_labels = np.zeros_like(self.labels_)
        for idx, label in enumerate(unique_labels):
            new_labels[self.labels_ == label] = idx
        self.labels_ = new_labels
        self.is_fitted = True
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Hierarchical clustering does not support prediction for new samples. Use .labels_ instead."
        )


AgglomerativeClustering = HierarchicalClustering
