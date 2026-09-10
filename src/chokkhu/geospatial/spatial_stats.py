"""Spatial Statistics, Spatial Autocorrelation, and Spatial Weight Matrices."""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np


class SpatialWeights:
    """Spatial Weight Matrix constructor for geographic coordinates.

    Supports KNN adjacency, inverse distance, and Gaussian distance decay.

    Parameters
    ----------
    coords : np.ndarray
        Array of 2D geographic coordinates (N x 2) [latitude, longitude] or [x, y].
    method : str
        Spatial kernel method: 'knn', 'inverse_distance', 'gaussian', or 'threshold'.
    k : int
        Number of nearest neighbors for 'knn' method.
    bandwidth : float
        Kernel bandwidth parameter for 'gaussian' or distance threshold.
    row_standardize : bool
        Whether to row-standardize the weight matrix (sum(W[i, :]) == 1.0).
    """

    def __init__(
        self,
        coords: np.ndarray,
        method: str = "knn",
        k: int = 5,
        bandwidth: float = 1.0,
        row_standardize: bool = True,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(self.coords)
        self.method = method.lower()
        self.k = min(int(k), self.n - 1)
        self.bandwidth = float(bandwidth)
        self.row_standardize = row_standardize
        self.W = self._build_weights()

    def _build_weights(self) -> np.ndarray:
        """Construct the N x N spatial weight matrix."""
        # Pairwise Euclidean distance
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        W: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)

        if self.method == "knn":
            for i in range(self.n):
                # Get indices of k nearest neighbors excluding self
                sorted_indices = np.argsort(dists[i])
                knn_indices = [idx for idx in sorted_indices if idx != i][: self.k]
                W[i, knn_indices] = 1.0

        elif self.method == "inverse_distance":
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and dists[i, j] > 0:
                        W[i, j] = 1.0 / (dists[i, j] ** self.bandwidth)

        elif self.method == "gaussian":
            W = np.exp(-0.5 * (dists / (self.bandwidth + 1e-12)) ** 2)
            np.fill_diagonal(W, 0.0)

        elif self.method == "threshold":
            W = (dists <= self.bandwidth).astype(np.float64)
            np.fill_diagonal(W, 0.0)

        else:
            raise ValueError(f"Unknown spatial weight method: {self.method}")

        if self.row_standardize:
            row_sums = np.sum(W, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            W = W / row_sums

        return W

    def to_matrix(self) -> np.ndarray:
        """Return the spatial weight matrix W."""
        return self.W.copy()


def morans_i(
    values: np.ndarray,
    weights: Union[np.ndarray, SpatialWeights],
    permutations: int = 99,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Calculate Global Moran's I spatial autocorrelation statistic and permutation p-value.

    Parameters
    ----------
    values : np.ndarray
        Array of spatial attribute values (length N).
    weights : np.ndarray or SpatialWeights
        N x N spatial weight matrix.
    permutations : int
        Number of random permutations to compute pseudo p-value.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    Dict[str, float]
        {'I': Moran's I value, 'expected_I': theoretical mean, 'z_score': z-score, 'p_value': p-value}
    """
    z = np.asarray(values, dtype=np.float64).flatten()
    n = len(z)
    W = (
        weights.to_matrix()
        if isinstance(weights, SpatialWeights)
        else np.asarray(weights, dtype=np.float64)
    )

    z_dev = z - np.mean(z)
    s0 = float(np.sum(W))
    denom = float(np.sum(z_dev**2))

    if denom < 1e-12 or s0 < 1e-12:
        return {"I": 0.0, "expected_I": -1.0 / (n - 1), "z_score": 0.0, "p_value": 1.0}

    # Moran's I numerator
    numer = float(np.sum(W * np.outer(z_dev, z_dev)))
    observed_i = (n / s0) * (numer / denom)
    expected_i = -1.0 / (n - 1)

    # Permutation test
    rng = np.random.default_rng(random_state)
    perm_i = []
    for _ in range(permutations):
        z_perm = rng.permutation(z_dev)
        p_num = float(np.sum(W * np.outer(z_perm, z_perm)))
        perm_i.append((n / s0) * (p_num / denom))

    perm_arr = np.array(perm_i)
    p_val = (np.sum(np.abs(perm_arr) >= np.abs(observed_i)) + 1) / (permutations + 1)
    std_i = float(np.std(perm_arr)) if np.std(perm_arr) > 0 else 1.0
    z_score = (observed_i - expected_i) / std_i

    return {
        "I": float(observed_i),
        "expected_I": float(expected_i),
        "z_score": float(z_score),
        "p_value": float(p_val),
    }


def local_morans_i(
    values: np.ndarray,
    weights: Union[np.ndarray, SpatialWeights],
) -> np.ndarray:
    """Calculate Local Moran's I statistic for each observation (LISA / Hotspot analysis)."""
    z = np.asarray(values, dtype=np.float64).flatten()
    n = len(z)
    W = (
        weights.to_matrix()
        if isinstance(weights, SpatialWeights)
        else np.asarray(weights, dtype=np.float64)
    )

    z_dev = z - np.mean(z)
    s2 = np.sum(z_dev**2) / float(n)
    if s2 < 1e-12:
        return np.zeros(n, dtype=np.float64)

    # Local I_i = (z_i / s^2) * sum_j (w_ij * z_j)
    spatial_lag = np.dot(W, z_dev)
    local_i = (z_dev / s2) * spatial_lag
    return local_i
