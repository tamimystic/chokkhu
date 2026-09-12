"""OPTICS (Ordering Points To Identify the Clustering Structure) in Pure NumPy/SciPy.

References:
- Ankerst, Breunig, Kriegel, Sander (1999): "OPTICS: Ordering Points To Identify the Clustering Structure" (ACM SIGMOD).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


class OPTICS(ChokkhuModel):
    """OPTICS Density-Based Clustering Algorithm with Reachability Analysis."""

    def __init__(
        self,
        min_samples: int = 5,
        max_eps: float = float("inf"),
        metric: str = "euclidean",
    ) -> None:
        super().__init__()
        self.min_samples = int(min_samples)
        self.max_eps = float(max_eps)
        self.metric = metric

        self.ordering_: np.ndarray = np.array([], dtype=np.int64)
        self.reachability_: np.ndarray = np.array([], dtype=np.float64)
        self.core_distances_: np.ndarray = np.array([], dtype=np.float64)
        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> OPTICS:
        """Compute OPTICS reachability ordering and cluster labels."""
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        dist_matrix = cdist(x_arr, x_arr, metric=self.metric)

        # Core distances: distance to min_samples-th nearest neighbor
        core_distances: np.ndarray = np.full(n_samples, np.inf, dtype=np.float64)
        for i in range(n_samples):
            sorted_dists = np.sort(dist_matrix[i])
            if len(sorted_dists) >= self.min_samples:
                d_val = sorted_dists[self.min_samples - 1]
                if d_val <= self.max_eps:
                    core_distances[i] = d_val

        processed = np.zeros(n_samples, dtype=bool)
        reachability: np.ndarray = np.full(n_samples, np.inf, dtype=np.float64)
        ordering: List[int] = []

        for start_idx in range(n_samples):
            if processed[start_idx]:
                continue

            order_queue = [start_idx]

            while order_queue:
                q_dists = [reachability[idx] for idx in order_queue]
                best_idx_in_queue = int(np.argmin(q_dists))
                curr_point = order_queue.pop(best_idx_in_queue)

                if processed[curr_point]:
                    continue

                processed[curr_point] = True
                ordering.append(curr_point)

                if core_distances[curr_point] < np.inf:
                    neighbors = np.where(~processed)[0]
                    for neighbor in neighbors:
                        direct_dist = dist_matrix[curr_point, neighbor]
                        if direct_dist <= self.max_eps:
                            new_reach = max(core_distances[curr_point], direct_dist)
                            if new_reach < reachability[neighbor]:
                                reachability[neighbor] = new_reach
                                if neighbor not in order_queue:
                                    order_queue.append(neighbor)

        self.ordering_ = np.array(ordering, dtype=np.int64)
        self.reachability_ = reachability[self.ordering_]
        self.core_distances_ = core_distances[self.ordering_]

        cluster_id = 0
        labels: np.ndarray = np.full(n_samples, -1, dtype=np.int64)
        threshold: float = (
            float(np.percentile(self.reachability_[self.reachability_ < np.inf], 75))
            if np.any(self.reachability_ < np.inf)
            else 1.0
        )

        for i, point_idx in enumerate(self.ordering_):
            if self.reachability_[i] <= threshold:
                labels[point_idx] = cluster_id
            else:
                if self.core_distances_[i] <= threshold:
                    cluster_id += 1
                    labels[point_idx] = cluster_id
                else:
                    labels[point_idx] = -1

        self.labels_ = labels
        self.is_fitted = True
        return self
