from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from chokkhu.core.math import cdist
from chokkhu.models.base import ChokkhuModel


class OPTICS(ChokkhuModel):
    def __init__(
        self,
        min_samples: int = 5,
        max_eps: float = float("inf"),
        metric: str = "euclidean",
        p: int = 2,
        metric_params: Optional[dict] = None,
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Optional[Union[int, float]] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.min_samples = int(min_samples)
        self.max_eps = float(max_eps)
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = float(xi)
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.ordering_: np.ndarray = np.array([], dtype=np.int64)
        self.reachability_: np.ndarray = np.array([], dtype=np.float64)
        self.core_distances_: np.ndarray = np.array([], dtype=np.float64)
        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> OPTICS:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        dist_matrix = cdist(x_arr, x_arr, metric=self.metric)

        core_distances = np.full(n_samples, np.inf, dtype=np.float64)
        for i in range(n_samples):
            sorted_dists = np.sort(dist_matrix[i])
            if len(sorted_dists) >= self.min_samples:
                d_val = sorted_dists[self.min_samples - 1]
                if d_val <= self.max_eps:
                    core_distances[i] = d_val

        processed = np.zeros(n_samples, dtype=bool)
        reachability = np.full(n_samples, np.inf, dtype=np.float64)
        ordering: List[int] = []

        for start_idx in range(n_samples):
            if processed[start_idx]:
                continue
            order_queue = [start_idx]
            while order_queue:
                p_idx = order_queue.pop(0)
                if processed[p_idx]:
                    continue
                processed[p_idx] = True
                ordering.append(p_idx)
                if core_distances[p_idx] < np.inf:
                    unprocessed = np.where(~processed)[0]
                    for up_idx in unprocessed:
                        new_r = max(core_distances[p_idx], dist_matrix[p_idx, up_idx])
                        if new_r < reachability[up_idx]:
                            reachability[up_idx] = new_r
                    unprocessed_orders = sorted(
                        unprocessed, key=lambda idx: reachability[idx]
                    )
                    order_queue = [
                        idx for idx in unprocessed_orders if reachability[idx] < np.inf
                    ]

        self.ordering_ = np.array(ordering, dtype=np.int64)
        self.reachability_ = reachability[self.ordering_]
        self.core_distances_ = core_distances[self.ordering_]

        eps_threshold = self.eps if self.eps is not None else 0.5
        labels = np.full(n_samples, -1, dtype=np.int64)
        cluster_id = 0
        for idx in self.ordering_:
            if reachability[idx] <= eps_threshold:
                labels[idx] = cluster_id
            else:
                if core_distances[idx] <= eps_threshold:
                    cluster_id += 1
                    labels[idx] = cluster_id

        self.labels_ = labels
        self.is_fitted = True
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "OPTICS does not support prediction for new samples. Use .labels_ instead."
        )
