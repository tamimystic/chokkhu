"""Spatial Kriging and Variogram Analysis for Geostatistical Interpolation."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class OrdinaryKriging:
    """Ordinary Kriging Spatial Interpolator with Variogram Modeling.

    Solves the Best Linear Unbiased Predictor (BLUP) Kriging system:
    [ Gamma  1 ] [ lambda ] = [ gamma_0 ]
    [ 1^T    0 ] [   mu   ]   [    1    ]

    Parameters
    ----------
    variogram_model : str
        Theoretical variogram model: 'spherical', 'exponential', or 'gaussian'.
    nugget : float
        Nugget effect (measurement error / micro-scale variance).
    sill : float
        Sill variance parameter (asymptote of semivariance).
    range_val : float
        Spatial correlation range distance.
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        nugget: float = 0.0,
        sill: float = 1.0,
        range_val: float = 1.0,
    ) -> None:
        self.variogram_model = variogram_model.lower()
        self.nugget = max(0.0, float(nugget))
        self.sill = max(1e-6, float(sill))
        self.range_val = max(1e-6, float(range_val))
        self.coords_train: Optional[np.ndarray] = None
        self.z_train: Optional[np.ndarray] = None

    def _semivariance(self, h: np.ndarray) -> np.ndarray:
        """Compute theoretical semivariance gamma(h)."""
        h_abs = np.abs(h)
        a = self.range_val
        c0 = self.nugget
        c = self.sill - c0

        if self.variogram_model == "spherical":
            gamma = np.where(
                h_abs <= a,
                c0 + c * (1.5 * (h_abs / a) - 0.5 * (h_abs / a) ** 3),
                c0 + c,
            )
        elif self.variogram_model == "exponential":
            gamma = c0 + c * (1.0 - np.exp(-3.0 * h_abs / a))
        elif self.variogram_model == "gaussian":
            gamma = c0 + c * (1.0 - np.exp(-3.0 * (h_abs / a) ** 2))
        else:
            gamma = c0 + c * (1.0 - np.exp(-3.0 * h_abs / a))

        return np.where(h_abs == 0.0, 0.0, gamma)

    def fit(self, coords: np.ndarray, z: np.ndarray) -> OrdinaryKriging:
        """Store spatial observation locations and values."""
        self.coords_train = np.asarray(coords, dtype=np.float64)
        self.z_train = np.asarray(z, dtype=np.float64).flatten()
        return self

    def predict(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict interpolated values and Kriging estimation variances at target coordinates.

        Returns
        -------
        z_pred : np.ndarray
            Estimated values at target coordinates.
        variance : np.ndarray
            Estimation error variances.
        """
        if self.coords_train is None or self.z_train is None:
            raise RuntimeError("OrdinaryKriging model has not been fitted.")

        coords_test = np.asarray(coords, dtype=np.float64)
        n = len(self.coords_train)
        m = len(coords_test)

        # Build pairwise distance matrix among training points
        diff_train = self.coords_train[:, None, :] - self.coords_train[None, :, :]
        dist_train = np.sqrt(np.sum(diff_train**2, axis=-1))

        # Build Kriging LHS matrix (n+1 x n+1)
        K: np.ndarray = np.zeros((n + 1, n + 1), dtype=np.float64)
        K[:n, :n] = self._semivariance(dist_train)
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        K_inv = np.linalg.pinv(K)

        z_preds: np.ndarray = np.zeros(m, dtype=np.float64)
        variances: np.ndarray = np.zeros(m, dtype=np.float64)

        for i in range(m):
            dists_to_train = np.sqrt(
                np.sum((self.coords_train - coords_test[i]) ** 2, axis=-1)
            )
            k_vec: np.ndarray = np.zeros(n + 1, dtype=np.float64)
            k_vec[:n] = self._semivariance(dists_to_train)
            k_vec[n] = 1.0

            # Solve weights: [lambda; mu] = K^-1 @ k_vec
            weights_mu = np.dot(K_inv, k_vec)
            weights = weights_mu[:n]
            mu = weights_mu[n]

            z_preds[i] = float(np.dot(weights, self.z_train))
            variances[i] = max(0.0, float(np.dot(weights, k_vec[:n]) + mu))

        return z_preds, variances
