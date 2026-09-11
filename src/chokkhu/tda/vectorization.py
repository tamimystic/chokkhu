"""Topological Data Analysis Vectorization: Persistence Landscapes and Persistence Images.

Formulated from first principles in pure NumPy and SciPy, implementing
Bubenik (2015) Statistical Topological Data Analysis with Persistence Landscapes and
Adams et al. (JMLR 2017) Persistence Images for machine learning pipelines.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Tuple, Union


class PersistenceLandscape:
    r"""Persistence Landscapes for Topological Data Analysis.

    Transforms persistent homology diagrams :math:`\{(b_i, d_i)\}_{i=1}^n` into functional
    Banach/Hilbert spaces via piecewise linear tent functions:

    .. math::
        f_{(b_i, d_i)}(t) = \max\left(0, \min(t - b_i, d_i - t)\right)
        \lambda_k(t) = \text{k-max}_{i=1}^n f_{(b_i, d_i)}(t)

    Parameters
    ----------
    n_landscapes : int, default=5
        Number of landscape levels :math:`k \in \{1, \dots, K\}` to retain.
    n_bins : int, default=100
        Number of evaluation grid points along :math:`t`.
    t_min : Optional[float], default=None
        Lower bound for domain evaluation. If None, inferred dynamically.
    t_max : Optional[float], default=None
        Upper bound for domain evaluation. If None, inferred dynamically.
    """

    def __init__(
        self,
        n_landscapes: int = 5,
        n_bins: int = 100,
        num_landscapes: Optional[int] = None,
        resolution: Optional[int] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> None:
        self.n_landscapes = int(num_landscapes if num_landscapes is not None else n_landscapes)
        self.n_bins = int(resolution if resolution is not None else n_bins)
        self.num_landscapes = self.n_landscapes
        self.resolution = self.n_bins
        self.t_min = t_min
        self.t_max = t_max

    def transform_single(
        self,
        diagram: Union[np.ndarray, Any],
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Transform a single persistence diagram into stacked landscape vectors.

        Parameters
        ----------
        diagram : np.ndarray, shape (n_points, 2) or PersistenceDiagram
            Birth-death pairs :math:`(b_i, d_i)`.
        t_min : Optional[float]
        t_max : Optional[float]

        Returns
        -------
        landscapes : np.ndarray, shape (n_landscapes, n_bins)
        """
        if hasattr(diagram, "pairs"):
            diag = np.asarray(diagram.pairs, dtype=np.float64)
        else:
            diag = np.asarray(diagram, dtype=np.float64)

        if diag.ndim == 1 and len(diag) == 2:
            diag = diag.reshape(1, 2)
        elif diag.ndim == 1 and len(diag) == 0:
            diag = np.empty((0, 2), dtype=np.float64)

        if len(diag) == 0:
            return np.zeros((self.n_landscapes, self.n_bins), dtype=np.float64)

        # Filter valid finite points
        valid_mask = np.isfinite(diag[:, 0]) & np.isfinite(diag[:, 1]) & (diag[:, 1] > diag[:, 0])
        valid_diag = diag[valid_mask]

        if len(valid_diag) == 0:
            return np.zeros((self.n_landscapes, self.n_bins), dtype=np.float64)

        t_start = t_min if t_min is not None else (self.t_min if self.t_min is not None else float(np.min(valid_diag[:, 0])))
        t_end = t_max if t_max is not None else (self.t_max if self.t_max is not None else float(np.max(valid_diag[:, 1])))
        if t_end <= t_start:
            t_end = t_start + 1.0

        t_grid = np.linspace(t_start, t_end, self.n_bins, dtype=np.float64)  # (M,)

        # Compute tent functions for all points: (N, M)
        births = valid_diag[:, 0, None]  # (N, 1)
        deaths = valid_diag[:, 1, None]  # (N, 1)

        tent = np.maximum(0.0, np.minimum(t_grid - births, deaths - t_grid))  # (N, M)

        # Sort descending along point axis
        sorted_tents = np.sort(tent, axis=0)[::-1]  # (N, M)

        landscapes: np.ndarray = np.zeros((self.n_landscapes, self.n_bins), dtype=np.float64)
        n_avail = min(self.n_landscapes, len(sorted_tents))
        landscapes[:n_avail] = sorted_tents[:n_avail]

        return landscapes

    def transform(
        self,
        diagrams: Union[List[Any], np.ndarray, Any],
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """Transform a persistence diagram or collection of diagrams into a 1D or 2D feature array.

        Parameters
        ----------
        diagrams : List or np.ndarray or PersistenceDiagram
            Single diagram or collection of diagrams.

        Returns
        -------
        features : np.ndarray
        """
        if isinstance(diagrams, list):
            results = [self.transform_single(d, t_min=t_min, t_max=t_max).ravel() for d in diagrams]
            return np.array(results, dtype=np.float64)

        if hasattr(diagrams, "pairs"):
            return self.transform_single(diagrams.pairs, t_min=t_min, t_max=t_max).ravel()

        arr = np.asarray(diagrams, dtype=np.float64)
        if arr.ndim <= 2:
            return self.transform_single(arr, t_min=t_min, t_max=t_max).ravel()
        else:
            results = [self.transform_single(d, t_min=t_min, t_max=t_max).ravel() for d in arr]
            return np.array(results, dtype=np.float64)


class PersistenceImage:
    r"""Persistence Images for Topological Machine Learning.

    Converts birth-death persistence diagrams into a 2D weighted Gaussian smoothed surface
    discretized across an image pixel grid.

    Parameters
    ----------
    pixels : Tuple[int, int], default=(20, 20)
        Grid resolution :math:`(N_x, N_y)`.
    sigma : float, default=0.1
        Gaussian smoothing kernel standard deviation.
    weight_power : float, default=1.0
        Linear/polynomial weighting ramp exponent for persistence :math:`p = d - b`.
    birth_range : Optional[Tuple[float, float]], default=None
        Range :math:`(b_{\min}, b_{\max})`.
    pers_range : Optional[Tuple[float, float]], default=None
        Range :math:`(p_{\min}, p_{\max})`.
    """

    def __init__(
        self,
        pixels: Tuple[int, int] = (20, 20),
        sigma: float = 0.1,
        weight_power: float = 1.0,
        birth_range: Optional[Tuple[float, float]] = None,
        pers_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.pixels = pixels
        self.sigma = float(sigma)
        self.weight_power = float(weight_power)
        self.birth_range = birth_range
        self.pers_range = pers_range

    def transform_single(self, diagram: np.ndarray) -> np.ndarray:
        r"""Transform a single persistence diagram into a 2D persistence image.

        Parameters
        ----------
        diagram : np.ndarray, shape (n_points, 2)
            Birth-death pairs :math:`(b_i, d_i)`.

        Returns
        -------
        image : np.ndarray, shape (N_x, N_y)
        """
        diag = np.asarray(diagram, dtype=np.float64)
        nx, ny = self.pixels

        if len(diag) == 0:
            return np.zeros((nx, ny), dtype=np.float64)

        valid_mask = np.isfinite(diag[:, 0]) & np.isfinite(diag[:, 1]) & (diag[:, 1] > diag[:, 0])
        valid_diag = diag[valid_mask]

        if len(valid_diag) == 0:
            return np.zeros((nx, ny), dtype=np.float64)

        births = valid_diag[:, 0]
        pers = valid_diag[:, 1] - valid_diag[:, 0]

        b_min, b_max = self.birth_range if self.birth_range else (float(np.min(births)), float(np.max(births)))
        p_min, p_max = self.pers_range if self.pers_range else (0.0, float(np.max(pers)))

        if b_max <= b_min:
            b_max = b_min + 1.0
        if p_max <= p_min:
            p_max = p_min + 1.0

        # Weights w(b, p) = (p / p_max)^r
        weights = np.clip((pers / p_max) ** self.weight_power, 0.0, 1.0)

        # 2D Grid centers
        x_edges = np.linspace(b_min, b_max, nx)
        y_edges = np.linspace(p_min, p_max, ny)
        grid_x: np.ndarray
        grid_y: np.ndarray
        grid_x, grid_y = np.meshgrid(x_edges, y_edges, indexing="ij")

        # Gaussian density accumulation: (nx, ny)
        image: np.ndarray = np.zeros((nx, ny), dtype=np.float64)
        var2 = 2.0 * (self.sigma ** 2)
        norm_factor = 1.0 / (2.0 * np.pi * (self.sigma ** 2))

        for b, p, w in zip(births, pers, weights):
            if w <= 1e-12:
                continue
            dist_sq = (grid_x - b) ** 2 + (grid_y - p) ** 2
            image += w * norm_factor * np.exp(-dist_sq / var2)

        return image

    def transform(self, diagrams: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Transform a batch of persistence diagrams into flattened image vectors.

        Parameters
        ----------
        diagrams : List[np.ndarray] or np.ndarray

        Returns
        -------
        features : np.ndarray, shape (n_diagrams, nx * ny)
        """
        results = []
        for diag in diagrams:
            img = self.transform_single(diag)
            results.append(img.ravel())
        return np.array(results, dtype=np.float64)
