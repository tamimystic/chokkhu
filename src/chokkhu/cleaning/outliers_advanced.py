from __future__ import annotations

from typing import Any

import numpy as np


class LocalOutlierFactor:
    """Local Outlier Factor (LOF) Unsupervised Outlier Detection.

    Computes the local density deviation of a given data point with respect to its
    k-nearest neighbors. Points that have a substantially lower density than their
    neighbors are considered outliers.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for k-distance queries.
    contamination : float, default=0.1
        The proportion of outliers in the data set. Must be in (0, 0.5].
    metric : str, default='euclidean'
        Metric used for distance computation ('euclidean', 'manhattan', 'cosine').
    novelty : bool, default=False
        Whether to use LocalOutlierFactor for novelty detection on unseen data.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "euclidean",
        novelty: bool = False,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if not (0.0 < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5].")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty

        self.X_fit_: np.ndarray | None = None
        self.k_distances_: np.ndarray | None = None
        self.lrd_: np.ndarray | None = None
        self.negative_outlier_factor_: np.ndarray | None = None
        self.offset_: float = -1.5

    def _compute_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between X1 and X2."""
        if self.metric == "manhattan":
            return np.sum(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)
        elif self.metric == "cosine":
            norm1 = np.linalg.norm(X1, axis=1, keepdims=True) + 1e-12
            norm2 = np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12
            cos_sim = np.dot(X1 / norm1, (X2 / norm2).T)
            return 1.0 - np.clip(cos_sim, -1.0, 1.0)
        else:  # euclidean
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff**2, axis=2) + 1e-12)

    def fit(self, X: np.ndarray | Any, y: Any = None) -> LocalOutlierFactor:
        """Fit the model using X as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        k = min(self.n_neighbors, n_samples - 1)
        if k < 1:
            k = 1

        self.X_fit_ = X_arr
        dist_matrix = self._compute_distances(X_arr, X_arr)
        np.fill_diagonal(dist_matrix, np.inf)

        # Sort distances to find k-nearest neighbors
        sorted_indices = np.argsort(dist_matrix, axis=1)
        k_neighbor_indices = sorted_indices[:, :k]

        # k-distance is the distance to the k-th nearest neighbor
        k_distances = np.take_along_axis(
            dist_matrix, sorted_indices[:, k - 1 : k], axis=1
        ).squeeze(axis=1)
        self.k_distances_ = k_distances

        # Compute reachability distances and local reachability density (lrd)
        # reach_dist(p, o) = max(k_dist(o), dist(p, o))
        lrd = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            neighbors = k_neighbor_indices[i]
            dists = dist_matrix[i, neighbors]
            neighbor_k_dists = k_distances[neighbors]
            reach_dists = np.maximum(neighbor_k_dists, dists)
            avg_reach = np.mean(reach_dists)
            lrd[i] = 1.0 / (avg_reach + 1e-10)

        self.lrd_ = lrd

        # Compute LOF score: average ratio of neighbor lrd to sample lrd
        lof_scores = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            neighbors = k_neighbor_indices[i]
            neighbor_lrds = lrd[neighbors]
            lof_scores[i] = np.mean(neighbor_lrds) / (lrd[i] + 1e-10)

        # Negative outlier factor (higher = more normal / inlier, lower = outlier)
        self.negative_outlier_factor_ = -lof_scores
        self.offset_ = float(
            np.percentile(self.negative_outlier_factor_, 100.0 * self.contamination)
        )
        return self

    def fit_predict(self, X: np.ndarray | Any, y: Any = None) -> np.ndarray:
        """Fit the model and predict whether samples are inliers (+1) or outliers (-1)."""
        self.fit(X, y)
        if self.negative_outlier_factor_ is None:
            raise RuntimeError("Model is not fitted.")
        preds: np.ndarray = np.ones(len(self.negative_outlier_factor_), dtype=int)
        preds[self.negative_outlier_factor_ < self.offset_] = -1
        return preds

    def score_samples(self, X: np.ndarray | Any) -> np.ndarray:
        """Opposite of the Local Outlier Factor of X.

        Higher score means more normal (inlier), lower means more abnormal (outlier).
        """
        if not self.novelty:
            raise AttributeError("score_samples is only available when novelty=True.")
        if self.X_fit_ is None or self.lrd_ is None or self.k_distances_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_queries = X_arr.shape[0]
        dist_matrix = self._compute_distances(X_arr, self.X_fit_)
        k = min(self.n_neighbors, self.X_fit_.shape[0])

        sorted_indices = np.argsort(dist_matrix, axis=1)
        k_neighbor_indices = sorted_indices[:, :k]

        scores = np.zeros(n_queries, dtype=np.float64)
        for i in range(n_queries):
            neighbors = k_neighbor_indices[i]
            dists = dist_matrix[i, neighbors]
            neighbor_k_dists = self.k_distances_[neighbors]
            reach_dists = np.maximum(neighbor_k_dists, dists)
            query_lrd = 1.0 / (np.mean(reach_dists) + 1e-10)
            neighbor_lrds = self.lrd_[neighbors]
            lof = np.mean(neighbor_lrds) / (query_lrd + 1e-10)
            scores[i] = -lof

        return scores

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict whether unseen samples are inliers (+1) or outliers (-1).

        Requires novelty=True.
        """
        if not self.novelty:
            raise AttributeError("predict is only available when novelty=True.")
        scores = self.score_samples(X)
        preds: np.ndarray = np.ones(len(scores), dtype=int)
        preds[scores < self.offset_] = -1
        return preds


class EllipticEnvelope:
    """Robust Covariance and Outlier Detection via FastMCD (Minimum Covariance Determinant).

    Fits a robust multivariate Gaussian distribution to the central data points,
    ignoring outliers to estimate the uncorrupted location and scatter matrix.
    Samples with large Mahalanobis distance are flagged as anomalies.

    Parameters
    ----------
    contamination : float, default=0.1
        The proportion of outliers in the data set. Must be in (0, 0.5].
    assume_centered : bool, default=False
        If True, the data will not be centered before computing scatter.
    support_fraction : float | None, default=None
        The proportion of points to be included in the support of the raw MCD.
        Default is None, which uses (n_samples + n_features + 1) / (2 * n_samples).
    random_state : int | None, default=None
        Seed for reproducibility during subset initialization.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        assume_centered: bool = False,
        support_fraction: float | None = None,
        random_state: int | None = None,
    ) -> None:
        if not (0.0 < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5].")
        self.contamination = contamination
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state

        self.location_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.offset_: float = 0.0

    def fit(self, X: np.ndarray | Any, y: Any = None) -> EllipticEnvelope:
        """Fit the EllipticEnvelope model via FastMCD C-steps."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)

        if self.support_fraction is not None:
            h = int(np.ceil(self.support_fraction * n_samples))
        else:
            h = int(np.floor((n_samples + n_features + 1) / 2))
        h = max(n_features + 1, min(h, n_samples))

        if self.assume_centered:
            mean = np.zeros(n_features, dtype=np.float64)
        else:
            mean = np.mean(X_arr, axis=0)

        # FastMCD: Initial random subsets and C-steps
        n_trials = 10 if n_samples > 100 else 5
        best_det = np.inf
        best_mean = mean
        best_cov = np.cov(X_arr, rowvar=False) + 1e-6 * np.eye(n_features)

        for _ in range(n_trials):
            # Select random seed subset of size (n_features + 1)
            init_idx = rng.choice(
                n_samples, min(n_features + 1, n_samples), replace=False
            )
            sub_X = X_arr[init_idx]
            sub_mean = (
                np.mean(sub_X, axis=0)
                if not self.assume_centered
                else np.zeros(n_features)
            )
            sub_diff = sub_X - sub_mean
            sub_cov = (sub_diff.T @ sub_diff) / max(1, len(sub_X) - 1) + 1e-6 * np.eye(
                n_features
            )

            # Perform 5 C-steps
            for _ in range(5):
                try:
                    inv_cov = np.linalg.pinv(sub_cov)
                except Exception:
                    inv_cov = np.eye(n_features)

                diff = X_arr - sub_mean
                sq_dist = np.sum((diff @ inv_cov) * diff, axis=1)
                h_idx = np.argsort(sq_dist)[:h]

                sub_X = X_arr[h_idx]
                sub_mean = (
                    np.mean(sub_X, axis=0)
                    if not self.assume_centered
                    else np.zeros(n_features)
                )
                sub_diff = sub_X - sub_mean
                sub_cov = (sub_diff.T @ sub_diff) / max(1, h - 1) + 1e-6 * np.eye(
                    n_features
                )

            det = np.linalg.det(sub_cov)
            if det > 0 and det < best_det:
                best_det = det
                best_mean = sub_mean
                best_cov = sub_cov

        self.location_ = best_mean
        self.covariance_ = best_cov
        self.precision_ = np.linalg.pinv(best_cov)

        # Compute Mahalanobis distances of all training points
        sq_mahal = self.mahalanobis(X_arr)
        # Inliers have smaller Mahalanobis distances (higher scores)
        scores = -sq_mahal
        self.offset_ = float(np.percentile(scores, 100.0 * self.contamination))
        return self

    def mahalanobis(self, X: np.ndarray | Any) -> np.ndarray:
        """Compute squared Mahalanobis distances of given samples."""
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        diff = X_arr - self.location_
        return np.sum((diff @ self.precision_) * diff, axis=1)

    def score_samples(self, X: np.ndarray | Any) -> np.ndarray:
        """Compute the negative Mahalanobis distance."""
        return -self.mahalanobis(X)

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict whether samples are inliers (+1) or outliers (-1)."""
        scores = self.score_samples(X)
        preds: np.ndarray = np.ones(len(scores), dtype=int)
        preds[scores < self.offset_] = -1
        return preds
