"""K-Nearest Neighbors Classifiers & Regressors with Spatial Indices in Pure NumPy/SciPy.

References:
- Cover & Hart (1967): "Nearest neighbor pattern classification" (IEEE TIT).
- Bentley (1975): "Multidimensional binary search trees used for associative searching" (CACM).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


class KNNClassifier(ChokkhuModel):
    """K-Nearest Neighbors Classifier supporting uniform and distance-weighted voting."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.p = float(p)

        self.X_train_: np.ndarray = np.array([], dtype=np.float64)
        self.y_train_: np.ndarray = np.array([], dtype=np.int64)
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KNNClassifier:
        if y is None:
            raise ValueError("y cannot be None for KNNClassifier")
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_train_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        n_classes = len(self.classes_)

        if self.metric == "minkowski":
            dists = cdist(x_arr, self.X_train_, metric="minkowski", p=self.p)
        else:
            dists = cdist(x_arr, self.X_train_, metric=self.metric)

        # Find k nearest indices
        k = min(self.n_neighbors, self.X_train_.shape[0])
        nearest_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]

        probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_samples):
            neighbor_indices = nearest_idx[i]
            neighbor_dists = dists[i, neighbor_indices]
            neighbor_labels = self.y_train_[neighbor_indices]

            if self.weights == "distance":
                weights = 1.0 / np.maximum(neighbor_dists, 1e-6)
                for c_idx, c in enumerate(self.classes_):
                    probs[i, c_idx] = np.sum(weights[neighbor_labels == c])
            else:
                for c_idx, c in enumerate(self.classes_):
                    probs[i, c_idx] = np.sum(neighbor_labels == c)

        return probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class KNNRegressor(ChokkhuModel):
    """K-Nearest Neighbors Regressor supporting uniform and inverse-distance averaging."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.p = float(p)

        self.X_train_: np.ndarray = np.array([], dtype=np.float64)
        self.y_train_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KNNRegressor:
        if y is None:
            raise ValueError("y cannot be None for KNNRegressor")
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        if self.metric == "minkowski":
            dists = cdist(x_arr, self.X_train_, metric="minkowski", p=self.p)
        else:
            dists = cdist(x_arr, self.X_train_, metric=self.metric)

        k = min(self.n_neighbors, self.X_train_.shape[0])
        nearest_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]

        preds = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            neighbor_indices = nearest_idx[i]
            neighbor_dists = dists[i, neighbor_indices]
            neighbor_targets = self.y_train_[neighbor_indices]

            if self.weights == "distance":
                weights = 1.0 / np.maximum(neighbor_dists, 1e-6)
                w_sum: float = float(np.sum(weights))
                preds[i] = float(np.sum(weights * neighbor_targets) / max(w_sum, 1e-12))
            else:
                preds[i] = float(np.mean(neighbor_targets))

        return preds
