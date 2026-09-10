"""Geographically Weighted Regression (GWR) for spatial heterogeneous parameter estimation."""

from __future__ import annotations

from typing import Optional
import numpy as np


class GeographicallyWeightedRegression:
    """Geographically Weighted Regression (GWR).

    Fits localized linear regression models at each spatial coordinate:
    beta(u_i, v_i) = (X^T W_i X)^(-1) X^T W_i y

    Parameters
    ----------
    bandwidth : float
        Spatial kernel bandwidth distance.
    kernel : str
        Kernel weighting function: 'gaussian', 'exponential', or 'bisquare'.
    fit_intercept : bool
        Whether to fit local intercept.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        fit_intercept: bool = True,
    ) -> None:
        self.bandwidth = float(bandwidth)
        self.kernel = kernel.lower()
        self.fit_intercept = fit_intercept
        self.coords_train: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.local_betas: Optional[np.ndarray] = None

    def _kernel_weights(
        self, target_coord: np.ndarray, source_coords: np.ndarray
    ) -> np.ndarray:
        """Compute spatial kernel diagonal weight vector for a target location."""
        dists = np.sqrt(np.sum((source_coords - target_coord) ** 2, axis=-1))
        b = max(1e-6, self.bandwidth)

        if self.kernel == "gaussian":
            w = np.exp(-0.5 * (dists / b) ** 2)
        elif self.kernel == "exponential":
            w = np.exp(-dists / b)
        elif self.kernel == "bisquare":
            w = np.where(dists < b, (1.0 - (dists / b) ** 2) ** 2, 0.0)
        else:
            w = np.exp(-0.5 * (dists / b) ** 2)

        return w

    def fit(
        self,
        coords: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> GeographicallyWeightedRegression:
        """Fit local regression coefficients at each training location."""
        self.coords_train = np.asarray(coords, dtype=np.float64)
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64).flatten()
        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train[:, None]

        n = len(self.coords_train)
        if self.fit_intercept:
            X_mat = np.column_stack([np.ones((n, 1)), self.X_train])
        else:
            X_mat = self.X_train

        p = X_mat.shape[1]
        self.local_betas = np.zeros((n, p), dtype=np.float64)

        for i in range(n):
            w = self._kernel_weights(self.coords_train[i], self.coords_train)
            W_diag = np.diag(w)

            XtW = np.dot(X_mat.T, W_diag)
            XtWX = np.dot(XtW, X_mat)
            XtWy = np.dot(XtW, self.y_train)

            # Local GLS solution
            beta_i = np.dot(np.linalg.pinv(XtWX), XtWy)
            self.local_betas[i] = beta_i

        return self

    def predict(self, coords: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict target values for new spatial locations."""
        if self.coords_train is None or self.X_train is None or self.y_train is None:
            raise RuntimeError("GWR model has not been fitted.")

        coords_test = np.asarray(coords, dtype=np.float64)
        X_test = np.asarray(X, dtype=np.float64)
        if len(X_test.shape) == 1:
            X_test = X_test[:, None]

        m = len(coords_test)
        preds: np.ndarray = np.zeros(m, dtype=np.float64)

        n_train = len(self.coords_train)
        if self.fit_intercept:
            X_train_mat = np.column_stack([np.ones((n_train, 1)), self.X_train])
            X_test_mat = np.column_stack([np.ones((m, 1)), X_test])
        else:
            X_train_mat = self.X_train
            X_test_mat = X_test

        for i in range(m):
            w = self._kernel_weights(coords_test[i], self.coords_train)
            W_diag = np.diag(w)
            XtW = np.dot(X_train_mat.T, W_diag)
            XtWX = np.dot(XtW, X_train_mat)
            XtWy = np.dot(XtW, self.y_train)
            beta_local = np.dot(np.linalg.pinv(XtWX), XtWy)

            preds[i] = np.dot(X_test_mat[i], beta_local)

        return preds
