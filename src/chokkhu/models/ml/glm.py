"""Generalized Linear Models (GLMs) & Online Linear Solvers in Pure NumPy.

References:
- Hoerl & Kennard (1970): "Ridge Regression" (Technometrics).
- Tibshirani (1996): "Regression Shrinkage and Selection via the Lasso" (JRSS-B).
- Zou & Hastie (2005): "Regularization and variable selection via the elastic net" (JRSS-B).
- Huber (1981): "Robust Statistics" (Wiley).
- MacKay (1992): "Bayesian Interpolation" (Neural Computation).
- Crammer et al. (2006): "Online Passive-Aggressive Algorithms" (JMLR).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from chokkhu.models.base import ChokkhuModel


def _soft_threshold(z: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Soft-thresholding shrinkage operator S(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    return np.sign(z) * np.maximum(0.0, np.abs(z) - gamma)


class RidgeRegression(ChokkhuModel):
    """Ridge Regression (L2 regularized OLS) with analytical Normal Equation solver."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> RidgeRegression:
        if y is None:
            raise ValueError("y cannot be None for RidgeRegression")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        if self.fit_intercept:
            x_mean = np.mean(x_arr, axis=0)
            y_mean = float(np.mean(y_arr))
            x_c = x_arr - x_mean
            y_c = y_arr - y_mean
        else:
            x_mean = np.zeros(n_features, dtype=np.float64)
            y_mean = 0.0
            x_c = x_arr
            y_c = y_arr

        # Solve (X^T X + alpha * I) w = X^T y
        A = np.dot(x_c.T, x_c) + self.alpha * np.eye(n_features)
        b_vec = np.dot(x_c.T, y_c)
        try:
            self.coef_ = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            self.coef_ = np.dot(np.linalg.pinv(A), b_vec)

        if self.fit_intercept:
            self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        else:
            self.intercept_ = 0.0

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_


class LassoRegression(ChokkhuModel):
    """Lasso Regression (L1 regularized) via exact cyclical Coordinate Descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> LassoRegression:
        if y is None:
            raise ValueError("y cannot be None for LassoRegression")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        if self.fit_intercept:
            x_mean = np.mean(x_arr, axis=0)
            y_mean = float(np.mean(y_arr))
            x_c = x_arr - x_mean
            y_c = y_arr - y_mean
        else:
            x_mean = np.zeros(n_features, dtype=np.float64)
            y_mean = 0.0
            x_c = x_arr
            y_c = y_arr

        # Column norms squared
        z = np.sum(x_c**2, axis=0)
        w = np.zeros(n_features, dtype=np.float64)
        residual = y_c - np.dot(x_c, w)

        for _ in range(self.max_iter):
            max_change = 0.0
            for j in range(n_features):
                if z[j] == 0:
                    continue
                # Partial residual without feature j
                rho_j = float(np.dot(x_c[:, j], residual + x_c[:, j] * w[j]))
                new_w_j = float(_soft_threshold(rho_j, n_samples * self.alpha) / z[j])
                diff = new_w_j - w[j]
                if abs(diff) > 0:
                    residual -= diff * x_c[:, j]
                    w[j] = new_w_j
                    max_change = max(max_change, abs(diff))

            if max_change < self.tol:
                break

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        else:
            self.intercept_ = 0.0

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_


class ElasticNet(ChokkhuModel):
    """ElasticNet Regression combining L1 and L2 penalties via Coordinate Descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ElasticNet:
        if y is None:
            raise ValueError("y cannot be None for ElasticNet")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        if self.fit_intercept:
            x_mean = np.mean(x_arr, axis=0)
            y_mean = float(np.mean(y_arr))
            x_c = x_arr - x_mean
            y_c = y_arr - y_mean
        else:
            x_mean = np.zeros(n_features, dtype=np.float64)
            y_mean = 0.0
            x_c = x_arr
            y_c = y_arr

        z = np.sum(x_c**2, axis=0)
        w = np.zeros(n_features, dtype=np.float64)
        residual = y_c - np.dot(x_c, w)

        l1_pen = n_samples * self.alpha * self.l1_ratio
        l2_pen = n_samples * self.alpha * (1.0 - self.l1_ratio)

        for _ in range(self.max_iter):
            max_change = 0.0
            for j in range(n_features):
                denom = z[j] + l2_pen
                if denom == 0:
                    continue
                rho_j = float(np.dot(x_c[:, j], residual + x_c[:, j] * w[j]))
                new_w_j = float(_soft_threshold(rho_j, l1_pen) / denom)
                diff = new_w_j - w[j]
                if abs(diff) > 0:
                    residual -= diff * x_c[:, j]
                    w[j] = new_w_j
                    max_change = max(max_change, abs(diff))

            if max_change < self.tol:
                break

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        else:
            self.intercept_ = 0.0

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_


class HuberRegressor(ChokkhuModel):
    """Linear regression robust to outliers minimizing the smooth Huber loss."""

    def __init__(
        self,
        epsilon: float = 1.35,
        alpha: float = 0.0001,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.scale_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> HuberRegressor:
        if y is None:
            raise ValueError("y cannot be None for HuberRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        # Augment with bias
        X_b = np.c_[x_arr, np.ones(n_samples)]
        w = np.zeros(n_features + 1, dtype=np.float64)
        # Initial OLS guess
        try:
            w = np.linalg.lstsq(X_b, y_arr, rcond=None)[0]
        except Exception:
            pass

        sigma = float(np.median(np.abs(y_arr - np.dot(X_b, w))) / 0.6745)
        sigma = max(sigma, 1e-4)

        # Iteratively Reweighted Least Squares (IRLS)
        for _ in range(self.max_iter):
            residuals = np.abs(y_arr - np.dot(X_b, w)) / sigma
            # Huber weights: w_i = 1 if |r| <= epsilon else epsilon / |r|
            weights = np.ones(n_samples, dtype=np.float64)
            outlier_mask = residuals > self.epsilon
            weights[outlier_mask] = self.epsilon / np.maximum(
                residuals[outlier_mask], 1e-6
            )

            # Weighted ridge solve
            W_mat = np.sqrt(weights)[:, np.newaxis]
            X_w = X_b * W_mat
            y_w = y_arr * np.sqrt(weights)

            reg_diag = np.full(n_features + 1, self.alpha)
            reg_diag[-1] = 0.0  # Do not regularize intercept

            A = np.dot(X_w.T, X_w) + np.diag(reg_diag)
            b_vec = np.dot(X_w.T, y_w)
            try:
                new_w = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                new_w = np.dot(np.linalg.pinv(A), b_vec)

            # Update sigma
            residuals_new = np.abs(y_arr - np.dot(X_b, new_w))
            new_sigma = float(np.median(residuals_new) / 0.6745)
            sigma = max(new_sigma, 1e-4)

            if np.max(np.abs(new_w - w)) < self.tol:
                w = new_w
                break
            w = new_w

        self.coef_ = w[:-1]
        self.intercept_ = float(w[-1])
        self.scale_ = sigma
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_


class BayesianRidge(ChokkhuModel):
    """Bayesian Ridge Regression estimating posterior weights and noise precisions via Evidence Maximization."""

    def __init__(
        self,
        n_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.alpha_: float = 1.0
        self.lambda_: float = 1.0
        self.sigma_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BayesianRidge:
        if y is None:
            raise ValueError("y cannot be None for BayesianRidge")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        x_mean = np.mean(x_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        x_c = x_arr - x_mean
        y_c = y_arr - y_mean

        # SVD of centered feature matrix: X = U S V^T
        U, S, Vt = np.linalg.svd(x_c, full_matrices=False)
        eigenvalues = S**2
        V = Vt.T
        Uy = np.dot(U.T, y_c)

        alpha = 1.0
        lambda_ = 1.0 / max(float(np.var(y_c)), 1e-6)

        for _ in range(self.n_iter):
            # Compute posterior mean coefficients: mu = lambda * V (S^2 lambda + alpha I)^(-1) S U^T y
            diag_inv = 1.0 / (lambda_ * eigenvalues + alpha)
            w = lambda_ * np.dot(V, diag_inv * S * Uy)

            # Effective number of parameters gamma = sum( (lambda * s_j^2) / (lambda * s_j^2 + alpha) )
            gamma = float(np.sum((lambda_ * eigenvalues) * diag_inv))

            old_alpha = alpha
            old_lambda = lambda_

            alpha = (gamma + 2.0 * self.alpha_1) / (
                float(np.sum(w**2)) + 2.0 * self.alpha_2
            )
            residuals_norm = float(np.sum((y_c - np.dot(x_c, w)) ** 2))
            lambda_ = (n_samples - gamma + 2.0 * self.lambda_1) / (
                residuals_norm + 2.0 * self.lambda_2
            )

            if (
                abs(alpha - old_alpha) < self.tol
                and abs(lambda_ - old_lambda) < self.tol
            ):
                break

        self.coef_ = w
        self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        self.alpha_ = alpha
        self.lambda_ = lambda_
        self.sigma_ = np.dot(V * diag_inv, V.T)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(X, dtype=np.float64)
        mean_preds = np.dot(x_arr, self.coef_) + self.intercept_
        std = np.sqrt(
            1.0 / self.lambda_ + np.sum(np.dot(x_arr, self.sigma_) * x_arr, axis=1)
        )
        return mean_preds, std


class ARDRegression(ChokkhuModel):
    """Automatic Relevance Determination (ARD) Regression with per-feature weight precisions."""

    def __init__(
        self,
        n_iter: int = 300,
        tol: float = 1e-3,
        threshold_lambda: float = 1e4,
    ) -> None:
        super().__init__()
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.threshold_lambda = float(threshold_lambda)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.alpha_: np.ndarray = np.array([], dtype=np.float64)
        self.lambda_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ARDRegression:
        if y is None:
            raise ValueError("y cannot be None for ARDRegression")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        x_mean = np.mean(x_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        x_c = x_arr - x_mean
        y_c = y_arr - y_mean

        alpha = np.ones(n_features, dtype=np.float64)
        lambda_ = 1.0 / max(float(np.var(y_c)), 1e-6)

        XtX = np.dot(x_c.T, x_c)
        Xty = np.dot(x_c.T, y_c)
        w = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_iter):
            # Posterior precision: A = lambda * X^T X + diag(alpha)
            A = lambda_ * XtX + np.diag(alpha)
            try:
                Sigma = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Sigma = np.linalg.pinv(A)

            w = lambda_ * np.dot(Sigma, Xty)
            gamma = 1.0 - alpha * np.diag(Sigma)

            old_alpha = np.copy(alpha)
            alpha = np.clip(gamma / (w**2 + 1e-12), 1e-6, self.threshold_lambda)
            residuals_norm = float(np.sum((y_c - np.dot(x_c, w)) ** 2))
            lambda_ = float((n_samples - np.sum(gamma)) / (residuals_norm + 1e-12))

            if np.max(np.abs(alpha - old_alpha)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        self.alpha_ = alpha
        self.lambda_ = lambda_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_) + self.intercept_


class PassiveAggressiveClassifier(ChokkhuModel):
    """Passive-Aggressive Online Margin-Based Classifier for streaming data."""

    def __init__(
        self,
        C: float = 1.0,
        loss: str = "hinge",
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.C = float(C)
        self.loss = loss
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> PassiveAggressiveClassifier:
        if y is None:
            raise ValueError("y cannot be None for PassiveAggressiveClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                "Binary classification only supported in core PassiveAggressiveClassifier"
            )

        y_mapped = np.where(y_arr == self.classes_[0], -1.0, 1.0)
        w = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                x_i = x_arr[i]
                y_i = y_mapped[i]
                margin = float(y_i * np.dot(w, x_i))
                loss_val = max(0.0, 1.0 - margin)

                if loss_val > 0.0:
                    norm_sq = float(np.dot(x_i, x_i))
                    if norm_sq > 0:
                        if self.loss == "hinge":
                            tau = min(self.C, loss_val / norm_sq)
                        else:  # squared_hinge (PA-II)
                            tau = loss_val / (norm_sq + 1.0 / (2.0 * self.C))
                        w += tau * y_i * x_i

        self.coef_ = w
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        scores = np.dot(x_arr, self.coef_)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])


class PassiveAggressiveRegressor(ChokkhuModel):
    """Passive-Aggressive Online Epsilon-Insensitive Regressor for streaming data."""

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        loss: str = "epsilon_insensitive",
        max_iter: int = 1000,
    ) -> None:
        super().__init__()
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.loss = loss
        self.max_iter = int(max_iter)

        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> PassiveAggressiveRegressor:
        if y is None:
            raise ValueError("y cannot be None for PassiveAggressiveRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        w = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                x_i = x_arr[i]
                y_i = y_arr[i]
                pred_i = float(np.dot(w, x_i))
                loss_val = max(0.0, abs(pred_i - y_i) - self.epsilon)

                if loss_val > 0.0:
                    norm_sq = float(np.dot(x_i, x_i))
                    if norm_sq > 0:
                        sign_err = float(np.sign(y_i - pred_i))
                        if self.loss == "epsilon_insensitive":
                            tau = min(self.C, loss_val / norm_sq)
                        else:  # squared_epsilon_insensitive
                            tau = loss_val / (norm_sq + 1.0 / (2.0 * self.C))
                        w += tau * sign_err * x_i

        self.coef_ = w
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_)
