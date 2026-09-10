"""Spatial Autoregressive Models (SAR): Spatial Lag Model and Spatial Error Model."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from .spatial_stats import SpatialWeights


class SpatialAutoregression:
    """Spatial Autoregressive (SAR) model for spatial econometric regression.

    Supports:
    1. Spatial Lag Model: y = rho * W * y + X * beta + epsilon
    2. Spatial Error Model: y = X * beta + u, where u = lambda_param * W * u + epsilon

    Parameters
    ----------
    model_type : str
        'lag' for Spatial Lag Model, 'error' for Spatial Error Model.
    fit_intercept : bool
        Whether to include intercept term.
    """

    def __init__(self, model_type: str = "lag", fit_intercept: bool = True) -> None:
        self.model_type = model_type.lower()
        self.fit_intercept = fit_intercept
        self.rho: float = 0.0
        self.beta: np.ndarray = np.array([])
        self.intercept: float = 0.0
        self.W: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Union[np.ndarray, SpatialWeights],
    ) -> SpatialAutoregression:
        """Fit spatial autoregression model using Two-Stage Least Squares (2SLS) / IV."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).flatten()
        n = len(y_arr)
        if len(X_arr.shape) == 1:
            X_arr = X_arr[:, None]

        self.W = (
            weights.to_matrix()
            if isinstance(weights, SpatialWeights)
            else np.asarray(weights, dtype=np.float64)
        )
        W = self.W

        if self.fit_intercept:
            X_mat = np.column_stack([np.ones((n, 1)), X_arr])
        else:
            X_mat = X_arr

        if self.model_type == "lag":
            # Spatial Lag: y = rho * W * y + X * beta
            # Instruments: H = [X, W @ X, W^2 @ X]
            Wy = np.dot(W, y_arr)
            WX = np.dot(W, X_mat)
            W2X = np.dot(W, WX)
            H = np.column_stack([X_mat, WX, W2X])

            # 2SLS: 1st stage predict Wy from H
            Z = np.column_stack([Wy, X_mat])  # Regressors [Wy, X]
            # Projection P_H = H (H^T H)^-1 H^T
            HtH_inv = np.linalg.pinv(np.dot(H.T, H))
            P_H = np.dot(H, np.dot(HtH_inv, H.T))
            Z_hat = np.dot(P_H, Z)

            # 2nd stage: (Z_hat^T Z_hat)^-1 Z_hat^T y
            gamma = np.dot(
                np.linalg.pinv(np.dot(Z_hat.T, Z_hat)), np.dot(Z_hat.T, y_arr)
            )

            self.rho = float(gamma[0])
            if self.fit_intercept:
                self.intercept = float(gamma[1])
                self.beta = gamma[2:]
            else:
                self.intercept = 0.0
                self.beta = gamma[1:]

        elif self.model_type == "error":
            # Spatial Error: y = X * beta + u, u = lambda * W * u + e
            # OLS for beta
            beta_ols = np.dot(
                np.linalg.pinv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, y_arr)
            )
            residuals = y_arr - np.dot(X_mat, beta_ols)
            W_res = np.dot(W, residuals)
            # Estimate lambda from residuals: res = lambda * W * res
            self.rho = float(
                np.dot(W_res.T, residuals) / (np.dot(W_res.T, W_res) + 1e-12)
            )
            self.rho = max(-0.99, min(0.99, self.rho))

            if self.fit_intercept:
                self.intercept = float(beta_ols[0])
                self.beta = beta_ols[1:]
            else:
                self.intercept = 0.0
                self.beta = beta_ols

        return self

    def predict(
        self, X: np.ndarray, y_known: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict target values."""
        X_arr = np.asarray(X, dtype=np.float64)
        if len(X_arr.shape) == 1:
            X_arr = X_arr[:, None]

        base_pred = np.dot(X_arr, self.beta) + self.intercept
        if self.model_type == "lag" and y_known is not None and self.W is not None:
            Wy = np.dot(self.W, y_known)
            return self.rho * Wy + base_pred
        return base_pred
