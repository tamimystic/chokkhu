"""Nearest Centroid & Radius Neighbors in Pure NumPy.

References:
- Tibshirani et al. (2002): "Diagnosis of multiple cancer types by shrunken centroids of gene expression" (PNAS).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


class NearestCentroid(ChokkhuModel):
    """Nearest Centroid Classifier with optional soft-threshold shrinkage."""

    def __init__(
        self,
        metric: str = "euclidean",
        shrink_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.shrink_threshold = shrink_threshold

        self.centroids_: np.ndarray = np.array([], dtype=np.float64)
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> NearestCentroid:
        if y is None:
            raise ValueError("y cannot be None for NearestCentroid")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_features = x_arr.shape[1]

        centroids = np.zeros((n_classes, n_features), dtype=np.float64)
        overall_centroid = np.mean(x_arr, axis=0)

        for idx, cls in enumerate(self.classes_):
            mask = y_arr == cls
            centroids[idx] = np.mean(x_arr[mask], axis=0)

        # Soft-threshold shrinkage
        if self.shrink_threshold is not None and self.shrink_threshold > 0:
            diff = centroids - overall_centroid
            shrunk_diff = np.sign(diff) * np.maximum(
                0.0, np.abs(diff) - self.shrink_threshold
            )
            centroids = overall_centroid + shrunk_diff

        self.centroids_ = centroids
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.centroids_, metric=self.metric)
        best_indices = np.argmin(dists, axis=1)
        return self.classes_[best_indices]


class RadiusNeighborsClassifier(ChokkhuModel):
    """Radius Neighbors Classifier with uniform or distance-weighted ball voting."""

    def __init__(
        self,
        radius: float = 1.0,
        weights: str = "uniform",
        outlier_label: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.weights = weights
        self.outlier_label = outlier_label

        self.X_train_: np.ndarray = np.array([], dtype=np.float64)
        self.y_train_: np.ndarray = np.array([], dtype=np.int64)
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> RadiusNeighborsClassifier:
        if y is None:
            raise ValueError("y cannot be None for RadiusNeighborsClassifier")
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_train_)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.X_train_, metric="euclidean")
        preds: np.ndarray = np.zeros(len(x_arr), dtype=np.int64)

        for i, row in enumerate(dists):
            in_ball = np.where(row <= self.radius)[0]
            if len(in_ball) == 0:
                preds[i] = (
                    self.outlier_label
                    if self.outlier_label is not None
                    else self.classes_[0]
                )
            else:
                neighbor_labels = self.y_train_[in_ball]
                if self.weights == "distance":
                    w = 1.0 / np.maximum(row[in_ball], 1e-6)
                    class_votes: list[float] = [
                        float(np.sum(w[neighbor_labels == c])) for c in self.classes_
                    ]
                    preds[i] = self.classes_[np.argmax(class_votes)]
                else:
                    counts = np.bincount(neighbor_labels)
                    preds[i] = np.argmax(counts)

        return preds


class RadiusNeighborsRegressor(ChokkhuModel):
    """Radius Neighbors Regressor with uniform or distance-weighted ball averaging."""

    def __init__(
        self,
        radius: float = 1.0,
        weights: str = "uniform",
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.weights = weights

        self.X_train_: np.ndarray = np.array([], dtype=np.float64)
        self.y_train_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> RadiusNeighborsRegressor:
        if y is None:
            raise ValueError("y cannot be None for RadiusNeighborsRegressor")
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.X_train_, metric="euclidean")
        preds: np.ndarray = np.zeros(len(x_arr), dtype=np.float64)
        global_mean = float(np.mean(self.y_train_))

        for i, row in enumerate(dists):
            in_ball = np.where(row <= self.radius)[0]
            if len(in_ball) == 0:
                preds[i] = global_mean
            else:
                neighbor_targets = self.y_train_[in_ball]
                if self.weights == "distance":
                    w = 1.0 / np.maximum(row[in_ball], 1e-6)
                    w_sum: float = float(np.sum(w))
                    preds[i] = float(np.sum(w * neighbor_targets) / max(w_sum, 1e-12))
                else:
                    preds[i] = float(np.mean(neighbor_targets))

        return preds
