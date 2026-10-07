from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..base import ChokkhuModel


class LinearRegression(ChokkhuModel):
    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        positive: bool = False,
        method: str = "ols",
        solver: Optional[str] = None,
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        max_iter: Optional[int] = None,
        tolerance: float = 1e-4,
        tol: Optional[float] = None,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.method = solver if solver is not None else method
        self.regularization = regularization
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.learning_rate = float(learning_rate)
        self.epochs = int(max_iter if max_iter is not None else epochs)
        self.tolerance = float(tol if tol is not None else tolerance)

        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        self.n_features_in_: int = 0
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> LinearRegression:
        if y is None:
            raise ValueError("y cannot be None for Linear Regression")

        X_arr = np.array(X, dtype=np.float64, copy=self.copy_X)
        y_arr = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            X_mean = np.mean(X_arr, axis=0)
            y_mean = float(np.mean(y_arr))
            X_centered = X_arr - X_mean
            y_centered = y_arr - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
            X_centered = X_arr
            y_centered = y_arr

        if (
            self.method in ("ols", "svd", "cholesky")
            and self.regularization is None
            and not self.positive
        ):
            try:
                theta = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]
            except Exception:
                theta = np.linalg.pinv(X_centered.T @ X_centered) @ (
                    X_centered.T @ y_centered
                )

            self.coef_ = np.asarray(theta, dtype=np.float64).ravel()
            self.intercept_ = float(y_mean - np.dot(X_mean, self.coef_))
        elif (
            self.method == "ols"
            and self.regularization in ("l2", "ridge")
            and not self.positive
        ):
            # Analytic Ridge
            A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
            theta = np.linalg.solve(A, X_centered.T @ y_centered)
            self.coef_ = np.asarray(theta, dtype=np.float64).ravel()
            self.intercept_ = float(y_mean - np.dot(X_mean, self.coef_))
        else:
            # Iterative Gradient Descent / Projected Gradient Descent (for positive=True or L1/ElasticNet)
            w = np.zeros(n_features, dtype=np.float64)
            b = 0.0

            for _ in range(self.epochs):
                y_pred = X_arr @ w + (b if self.fit_intercept else 0.0)
                error = y_pred - y_arr

                dw = (1.0 / n_samples) * (X_arr.T @ error)
                db = (
                    (1.0 / n_samples) * float(np.sum(error))
                    if self.fit_intercept
                    else 0.0
                )

                if self.regularization in ("l2", "ridge"):
                    dw += (self.alpha / n_samples) * w
                elif self.regularization in ("l1", "lasso"):
                    dw += (self.alpha / n_samples) * np.sign(w)
                elif self.regularization in ("elastic_net", "elasticnet"):
                    l1_term = self.l1_ratio * np.sign(w)
                    l2_term = (1.0 - self.l1_ratio) * w
                    dw += (self.alpha / n_samples) * (l1_term + l2_term)

                w -= self.learning_rate * dw
                if self.fit_intercept:
                    b -= self.learning_rate * db

                if self.positive:
                    w = np.maximum(0.0, w)

                if float(np.max(np.abs(self.learning_rate * dw))) < self.tolerance:
                    break

            self.coef_ = w
            self.intercept_ = b

        self.weights = self.coef_
        self.bias = self.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or len(self.coef_) == 0:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        y_true = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res: float = float(np.sum((y_true - y_pred) ** 2))
        ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float(1.0 - (ss_res / (ss_tot + 1e-15)))
