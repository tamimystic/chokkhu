from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from chokkhu.core.math import cdist

from ..base import ChokkhuModel


class KNN(ChokkhuModel):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Union[str, Callable[[np.ndarray], np.ndarray]] = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: float = 2.0,
        metric: str = "minkowski",
        metric_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        task: str = "classification",
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = float(p)
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.task = task

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KNN:
        if y is None:
            raise ValueError("y cannot be None for KNN")
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.n_features_in_ = self.X_train.shape[1]

        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(self.y_train.dtype, np.floating)
                else "classification"
            )

        if self.task == "classification":
            self.classes_ = np.unique(self.y_train)

        return self

    def kneighbors(
        self,
        X: Optional[np.ndarray] = None,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.X_train is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = self.X_train if X is None else np.asarray(X, dtype=np.float64)
        k = int(n_neighbors if n_neighbors is not None else self.n_neighbors)
        k = min(k, len(self.X_train))

        metric_name = "cityblock" if self.metric == "manhattan" else self.metric
        if metric_name == "minkowski":
            dists = cdist(X_arr, self.X_train, metric="minkowski", p=self.p)
        else:
            dists = cdist(X_arr, self.X_train, metric=metric_name)

        indices = np.argsort(dists, axis=1)[:, :k]
        sorted_dists = np.take_along_axis(dists, indices, axis=1)

        if return_distance:
            return sorted_dists, indices
        return indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=np.float64)
        dists, indices = self.kneighbors(X_arr, return_distance=True)
        assert isinstance(dists, np.ndarray)
        assert isinstance(indices, np.ndarray)

        predictions = []
        for i in range(len(X_arr)):
            row_idx = indices[i]
            row_dist = dists[i]
            labels = self.y_train[row_idx]

            if self.weights == "distance":
                w = 1.0 / (row_dist + 1e-15)
            elif callable(self.weights):
                w = self.weights(row_dist)
            else:
                w = np.ones_like(row_dist)

            if self.task == "classification":
                unique_labels = np.unique(labels)
                score_map = {
                    lbl: float(np.sum(w[labels == lbl])) for lbl in unique_labels
                }
                best_label = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                predictions.append(best_label)
            else:
                weighted_mean = float(np.average(labels, weights=w))
                predictions.append(weighted_mean)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")

        X_arr = np.asarray(X, dtype=np.float64)
        dists, indices = self.kneighbors(X_arr, return_distance=True)
        assert isinstance(dists, np.ndarray)
        assert isinstance(indices, np.ndarray)

        n_samples = len(X_arr)
        n_classes = len(self.classes_)
        probas: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_samples):
            row_idx = indices[i]
            row_dist = dists[i]
            labels = self.y_train[row_idx]

            if self.weights == "distance":
                w = 1.0 / (row_dist + 1e-15)
            elif callable(self.weights):
                w = self.weights(row_dist)
            else:
                w = np.ones_like(row_dist)

            total_w = float(np.sum(w))
            for c_idx, c_val in enumerate(self.classes_):
                c_weight = float(np.sum(w[labels == c_val]))
                probas[i, c_idx] = c_weight / (total_w + 1e-15)

        return probas

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y_true))
        else:
            ss_res: float = float(np.sum((y_true - y_pred) ** 2))
            ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
            return float(1.0 - (ss_res / (ss_tot + 1e-15)))
