"""Advanced Clustering Universe: MiniBatchKMeans, GMM, Spectral & Hierarchical in Pure NumPy/SciPy.

References:
- Sculley (2010): "Web-scale k-means clustering" (ACM WWW).
- Dempster, Laird, Rubin (1977): "Maximum Likelihood from Incomplete Data via the EM Algorithm" (JRSS-B).
- Ng, Jordan, Weiss (2001): "On Spectral Clustering: Analysis and an algorithm" (NeurIPS).
- Ward (1963): "Hierarchical Grouping to Optimize an Objective Function" (JASA).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from chokkhu.models.base import ChokkhuModel


class MiniBatchKMeans(ChokkhuModel):
    """Mini-Batch K-Means for streaming and large-scale fast clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 100,
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.cluster_centers_: np.ndarray = np.array([], dtype=np.float64)
        self.counts_: np.ndarray = np.array([], dtype=np.int64)
        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.inertia_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MiniBatchKMeans:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        # Initialize centers with random sample
        init_idx = self.rng.choice(
            n_samples, size=min(self.n_clusters, n_samples), replace=False
        )
        centers = np.copy(x_arr[init_idx])
        counts: np.ndarray = np.zeros(self.n_clusters, dtype=np.int64)

        for _ in range(self.max_iter):
            batch_idx = self.rng.choice(
                n_samples, size=min(self.batch_size, n_samples), replace=False
            )
            batch_X = x_arr[batch_idx]

            dists = cdist(batch_X, centers, metric="sqeuclidean")
            nearest_centers = np.argmin(dists, axis=1)

            for i, center_idx in enumerate(nearest_centers):
                counts[center_idx] += 1
                eta = 1.0 / float(counts[center_idx])
                centers[center_idx] = (1.0 - eta) * centers[center_idx] + eta * batch_X[
                    i
                ]

        self.cluster_centers_ = centers
        self.counts_ = counts
        all_dists = cdist(x_arr, centers, metric="sqeuclidean")
        self.labels_ = np.argmin(all_dists, axis=1)
        self.inertia_ = float(np.sum(np.min(all_dists, axis=1)))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        dists = cdist(x_arr, self.cluster_centers_, metric="sqeuclidean")
        return np.argmin(dists, axis=1)


class GaussianMixture(ChokkhuModel):
    """Gaussian Mixture Model (GMM) with Expectation-Maximization (EM) soft clustering."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.weights_: np.ndarray = np.array([], dtype=np.float64)
        self.means_: np.ndarray = np.array([], dtype=np.float64)
        self.covariances_: list[np.ndarray] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> GaussianMixture:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        # Initialize with random centers
        init_idx = self.rng.choice(n_samples, size=self.n_components, replace=False)
        means = np.copy(x_arr[init_idx])
        weights = np.ones(self.n_components, dtype=np.float64) / float(
            self.n_components
        )
        covariances: list[np.ndarray] = [
            np.eye(n_features, dtype=np.float64) for _ in range(self.n_components)
        ]

        prev_log_lik = -np.inf

        for _ in range(self.max_iter):
            # E-step: compute responsibilities gamma_ik
            resp = np.zeros((n_samples, self.n_components), dtype=np.float64)
            for k in range(self.n_components):
                diff = x_arr - means[k]
                cov_k = covariances[k] + 1e-6 * np.eye(n_features)
                sign, logdet = np.linalg.slogdet(cov_k)
                inv_cov = np.linalg.pinv(cov_k)
                quad = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                log_p = (
                    -0.5 * logdet - 0.5 * quad - 0.5 * n_features * np.log(2.0 * np.pi)
                )
                resp[:, k] = np.log(max(weights[k], 1e-12)) + log_p

            max_resp = np.max(resp, axis=1, keepdims=True)
            exp_resp = np.exp(resp - max_resp)
            sum_exp = np.sum(exp_resp, axis=1, keepdims=True)
            gamma = exp_resp / np.maximum(sum_exp, 1e-12)

            log_lik = float(np.sum(max_resp + np.log(np.maximum(sum_exp, 1e-12))))
            if abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik

            # M-step: update parameters
            N_k = np.sum(gamma, axis=0)  # [K]
            weights = N_k / float(n_samples)

            for k in range(self.n_components):
                n_k_val = max(float(N_k[k]), 1e-12)
                means[k] = np.sum(gamma[:, k : k + 1] * x_arr, axis=0) / n_k_val
                diff_k = x_arr - means[k]
                covariances[k] = np.dot(
                    (gamma[:, k : k + 1] * diff_k).T, diff_k
                ) / n_k_val + 1e-6 * np.eye(n_features)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = x_arr.shape
        resp = np.zeros((n_samples, self.n_components), dtype=np.float64)

        for k in range(self.n_components):
            diff = x_arr - self.means_[k]
            cov_k = self.covariances_[k] + 1e-6 * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(cov_k)
            inv_cov = np.linalg.pinv(cov_k)
            quad = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            log_p = -0.5 * logdet - 0.5 * quad
            resp[:, k] = np.log(max(self.weights_[k], 1e-12)) + log_p

        max_resp = np.max(resp, axis=1, keepdims=True)
        exp_resp = np.exp(resp - max_resp)
        return exp_resp / np.sum(exp_resp, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class SpectralClustering(ChokkhuModel):
    """Spectral Clustering via Normalized Graph Laplacian Eigen-decomposition."""

    def __init__(
        self,
        n_clusters: int = 3,
        gamma: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.gamma = float(gamma)
        self.random_state = random_state

        self.labels_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> SpectralClustering:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        # RBF Affinity matrix
        sq_dists = cdist(x_arr, x_arr, metric="sqeuclidean")
        W = np.exp(-self.gamma * sq_dists)
        np.fill_diagonal(W, 0.0)

        # Degree matrix and Normalized Laplacian L_sym = I - D^(-1/2) W D^(-1/2)
        degrees = np.sum(W, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        L_sym = np.eye(n_samples) - (
            d_inv_sqrt[:, np.newaxis] * W * d_inv_sqrt[np.newaxis, :]
        )

        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        # Select k smallest eigenvectors
        embedding = eigenvectors[:, : self.n_clusters]
        # Row normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding_norm = embedding / np.maximum(norm, 1e-12)

        # Cluster embedding with MiniBatchKMeans / KMeans
        from chokkhu.models.ml.kmeans import KMeans

        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        km.fit(embedding_norm)
        self.labels_ = (
            km.labels_
            if km.labels_ is not None
            else np.zeros(n_samples, dtype=np.int64)
        )
        self.is_fitted = True
        return self
