r"""Conformalized Quantile Regression (CQR) for Distribution-Free Uncertainty Quantification.

Formulated from first principles in pure NumPy and SciPy following Romano, Patterson & Candès (NeurIPS 2019).
Combines pinball loss quantile regression with split conformal calibration scores to provide
exact finite-sample distribution-free prediction intervals with mathematically guaranteed coverage:
P(Y \in \hat{C}(X)) \ge 1 - \alpha.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class QuantileRegressor:
    """Linear Quantile Regressor minimizing Pinball / Check Loss via iteratively reweighted least squares."""

    def __init__(
        self, quantile: float = 0.5, max_iter: int = 100, tol: float = 1e-4
    ) -> None:
        self.quantile = float(quantile)
        self.max_iter = max(10, int(max_iter))
        self.tol = float(tol)
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressor":
        """Fit linear quantile regression weights using Huberized Iteratively Reweighted Least Squares."""
        X_mat = np.atleast_2d(np.asarray(X, dtype=float))
        N, d = X_mat.shape
        y_vec = np.asarray(y, dtype=float).flatten()

        # Add bias column
        X_bias = np.hstack([X_mat, np.ones((N, 1))])
        tau = self.quantile

        # Initial ordinary least squares estimate
        try:
            w = np.linalg.lstsq(X_bias, y_vec, rcond=None)[0]
        except np.linalg.LinAlgError:
            w = np.zeros(d + 1)

        delta = 1e-4

        for _ in range(self.max_iter):
            w_prev = w.copy()
            residuals = y_vec - np.dot(X_bias, w)

            # Huberized asymmetric weights
            # W_i = 1 / max(delta, |r_i|) * (tau if r_i >= 0 else 1 - tau)
            abs_r = np.maximum(np.abs(residuals), delta)
            asym_factor = np.where(residuals >= 0, tau, 1.0 - tau)
            weights = asym_factor / abs_r

            # Weighted least squares update
            W_sqrt = np.sqrt(weights)[:, None]
            X_w = X_bias * W_sqrt
            y_w = y_vec * W_sqrt.flatten()

            try:
                w = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(w - w_prev) < self.tol:
                break

        self.weights_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict conditional quantile."""
        if self.weights_ is None:
            raise ValueError("QuantileRegressor has not been fitted yet.")
        X_mat = np.atleast_2d(np.asarray(X, dtype=float))
        N = X_mat.shape[0]
        X_bias = np.hstack([X_mat, np.ones((N, 1))])
        return np.dot(X_bias, self.weights_)


class ConformalizedQuantileRegression:
    r"""Conformalized Quantile Regression (CQR) with Exact Finite-Sample Validity.

    Given a target significance level $\alpha \in (0, 1)$, CQR fits a lower quantile $\hat{q}_{\alpha/2}$
    and upper quantile $\hat{q}_{1-\alpha/2}$, then calibrates the empirical conformality error:
    E_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))

    yielding valid non-parametric prediction intervals:
    \hat{C}(x) = [\hat{q}_{\alpha/2}(x) - \hat{Q}_{1-\alpha}(E), \hat{q}_{1-\alpha/2}(x) + \hat{Q}_{1-\alpha}(E)]

    Parameters
    ----------
    alpha : float, default=0.1
        Significance error rate ($1 - \alpha$ target coverage, e.g. $\alpha=0.1 \implies 90\%$ coverage).
    max_iter : int, default=100
        Maximum quantile regressor iterations.
    """

    def __init__(self, alpha: float = 0.1, max_iter: int = 100) -> None:
        self.alpha = float(alpha)
        self.max_iter = max(10, int(max_iter))

        self.low_quantile = self.alpha / 2.0
        self.high_quantile = 1.0 - self.alpha / 2.0

        self.q_low = QuantileRegressor(
            quantile=self.low_quantile, max_iter=self.max_iter
        )
        self.q_high = QuantileRegressor(
            quantile=self.high_quantile, max_iter=self.max_iter
        )

        self.conformal_adjustment_: float = 0.0
        self.is_calibrated_: bool = False

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "ConformalizedQuantileRegression":
        """Fit base lower and upper quantile models on training dataset."""
        X_mat = np.asarray(X_train, dtype=float)
        y_vec = np.asarray(y_train, dtype=float).flatten()

        self.q_low.fit(X_mat, y_vec)
        self.q_high.fit(X_mat, y_vec)
        return self

    def calibrate(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> "ConformalizedQuantileRegression":
        """Calibrate conformal correction factor Q_{1-alpha} on independent holdout calibration data."""
        X_cal = np.asarray(X_calib, dtype=float)
        y_cal = np.asarray(y_calib, dtype=float).flatten()
        n_cal = X_cal.shape[0]

        # Predict base unadjusted quantiles
        pred_low = self.q_low.predict(X_cal)
        pred_high = self.q_high.predict(X_cal)

        # Conformality non-conformity scores: E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
        scores = np.maximum(pred_low - y_cal, y_cal - pred_high)

        # Finite-sample quantile level: ceil((n+1) * (1-alpha)) / n
        quantile_level = float(
            min(1.0, np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal)
        )
        self.conformal_adjustment_ = float(
            np.quantile(scores, quantile_level, method="higher")
        )
        self.is_calibrated_ = True

        return self

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Predict calibrated lower and upper conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            Test query observations.

        Returns
        -------
        lower_bound : np.ndarray of shape (N,)
            Calibrated lower prediction bound \hat{q}_{low}(x) - \hat{Q}.
        upper_bound : np.ndarray of shape (N,)
            Calibrated upper prediction bound \hat{q}_{high}(x) + \hat{Q}.
        """
        X_mat = np.asarray(X, dtype=float)
        base_low = self.q_low.predict(X_mat)
        base_high = self.q_high.predict(X_mat)

        adj = self.conformal_adjustment_ if self.is_calibrated_ else 0.0

        lower_bound = base_low - adj
        upper_bound = base_high + adj

        # Ensure lower <= upper
        upper_bound = np.maximum(lower_bound, upper_bound)
        return lower_bound, upper_bound

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute empirical test coverage rate and average interval width."""
        y_true = np.asarray(y_test, dtype=float).flatten()
        low, high = self.predict_interval(X_test)

        covered = (y_true >= low) & (y_true <= high)
        coverage_rate = float(np.mean(covered))
        avg_width = float(np.mean(high - low))

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": coverage_rate,
            "average_width": avg_width,
        }
