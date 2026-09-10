"""Persistence Diagrams, Persistence Landscapes, and Topological Feature Extraction."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np


class PersistenceDiagram:
    """Topological Persistence Diagram storing (birth, death) pairs.

    Parameters
    ----------
    pairs : np.ndarray
        Array of shape (N, 2) where each row is [birth, death].
    dimension : int
        Homology dimension (e.g. 0, 1).
    """

    def __init__(self, pairs: np.ndarray, dimension: int = 0) -> None:
        self.pairs = np.asarray(pairs, dtype=np.float64)
        if len(self.pairs.shape) == 1 and len(self.pairs) == 2:
            self.pairs = self.pairs.reshape(1, 2)
        elif len(self.pairs.shape) == 0:
            self.pairs = np.empty((0, 2), dtype=np.float64)
        self.dimension = int(dimension)

    def lifetimes(self, remove_infinite: bool = True) -> np.ndarray:
        """Return persistence lifetimes (death - birth)."""
        if len(self.pairs) == 0:
            return np.empty(0, dtype=np.float64)
        p = self.pairs
        if remove_infinite:
            finite_mask = np.isfinite(p[:, 1])
            p = p[finite_mask]
        return p[:, 1] - p[:, 0]

    def total_persistence(self, p: float = 1.0) -> float:
        """Calculate total p-persistence sum((death - birth)^p)."""
        lt = self.lifetimes(remove_infinite=True)
        if len(lt) == 0:
            return 0.0
        return float(np.sum(lt**p))

    def persistent_entropy(self) -> float:
        """Compute topological persistent entropy of finite features."""
        lt = self.lifetimes(remove_infinite=True)
        total: float = float(np.sum(lt))
        if total < 1e-12 or len(lt) == 0:
            return 0.0
        probs = lt / total
        val = float(np.sum(probs * np.log(probs + 1e-12)))
        return -val


class PersistenceLandscape:
    """Persistence Landscape vectorizer for Topological Machine Learning.

    Maps a persistence diagram to continuous piecewise-linear functions:
    Lambda_i(t) = max(0, min(t - b, d - t)) for (b, d).
    The k-th landscape lambda_k(t) is the k-th largest value of Lambda_i(t).

    Parameters
    ----------
    num_landscapes : int
        Number of landscape layers (k = 1, 2, ..., K).
    resolution : int
        Number of discretization evaluation grid points.
    """

    def __init__(self, num_landscapes: int = 3, resolution: int = 50) -> None:
        self.num_landscapes = int(num_landscapes)
        self.resolution = int(resolution)

    def transform(
        self,
        pairs: Union[np.ndarray, PersistenceDiagram],
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """Compute discretized persistence landscape vector.

        Returns
        -------
        landscape_vector : np.ndarray
            Flattened array of shape (num_landscapes * resolution,).
        """
        p_arr = (
            pairs.pairs
            if isinstance(pairs, PersistenceDiagram)
            else np.asarray(pairs, dtype=np.float64)
        )
        if len(p_arr.shape) == 1 and len(p_arr) == 2:
            p_arr = p_arr.reshape(1, 2)

        # Filter finite pairs with positive lifetime
        if len(p_arr) > 0:
            finite_mask = np.isfinite(p_arr[:, 1]) & (p_arr[:, 1] > p_arr[:, 0])
            p_arr = p_arr[finite_mask]

        if len(p_arr) == 0:
            return np.zeros(self.num_landscapes * self.resolution, dtype=np.float64)

        if t_min is None:
            t_min = float(np.min(p_arr[:, 0]))
        if t_max is None:
            t_max = float(np.max(p_arr[:, 1]))

        if t_min >= t_max:
            t_max = t_min + 1.0

        t_grid = np.linspace(t_min, t_max, self.resolution)
        landscapes: np.ndarray = np.zeros(
            (self.num_landscapes, self.resolution), dtype=np.float64
        )

        for idx, t in enumerate(t_grid):
            # Compute tent functions Lambda_i(t) for all (b, d) pairs
            b = p_arr[:, 0]
            d = p_arr[:, 1]
            tents = np.maximum(0.0, np.minimum(t - b, d - t))
            # Sort in descending order
            sorted_tents = np.sort(tents)[::-1]

            for k in range(min(self.num_landscapes, len(sorted_tents))):
                landscapes[k, idx] = sorted_tents[k]

        return landscapes.flatten()


def bottleneck_distance(diag1: np.ndarray, diag2: np.ndarray) -> float:
    """Calculate the Bottleneck distance between two persistence diagrams."""
    d1 = np.asarray(diag1, dtype=np.float64)
    d2 = np.asarray(diag2, dtype=np.float64)

    # Filter finite pairs
    if len(d1) > 0:
        d1 = d1[np.isfinite(d1[:, 1])]
    if len(d2) > 0:
        d2 = d2[np.isfinite(d2[:, 1])]

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        return float(np.max((d2[:, 1] - d2[:, 0]) / 2.0))
    if len(d2) == 0:
        return float(np.max((d1[:, 1] - d1[:, 0]) / 2.0))

    # Pad with diagonal projections (b + d)/2
    n1, n2 = len(d1), len(d2)

    cost_matrix: np.ndarray = np.zeros((n1 + n2, n1 + n2), dtype=np.float64)

    # Pairwise L_inf distance between points in diag1 and diag2
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = max(abs(d1[i, 0] - d2[j, 0]), abs(d1[i, 1] - d2[j, 1]))

    # Distance to diagonal
    for i in range(n1):
        cost_matrix[i, n2 + i] = (d1[i, 1] - d1[i, 0]) / 2.0
    for j in range(n2):
        cost_matrix[n1 + j, j] = (d2[j, 1] - d2[j, 0]) / 2.0

    return (
        float(np.min(cost_matrix[cost_matrix > 0])) if np.any(cost_matrix > 0) else 0.0
    )
