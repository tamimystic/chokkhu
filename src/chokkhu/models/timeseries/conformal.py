"""Sovereign Distribution-Free Conformal Prediction & Uncertainty Intervals in Pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


class ConformalPredictor:
    """Distribution-Free Split & Inductive Conformal Prediction for Regression & Time Series."""

    def __init__(
        self,
        base_estimator: Any,
        alpha: float = 0.1,
        residual_type: str = "absolute",
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.residual_type = residual_type
        self.random_state = random_state
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        alpha: Optional[float] = None,
    ) -> float:
        """Compute non-conformity scores and calculate empirical quantile threshold."""
        if alpha is not None:
            self.alpha = alpha

        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).squeeze()
        n_cal = X_cal.shape[0]

        if hasattr(self.base_estimator, "predict"):
            preds = np.asarray(self.base_estimator.predict(X_cal)).squeeze()
        else:
            preds = np.asarray(self.base_estimator(X_cal)).squeeze()

        if self.residual_type == "absolute":
            scores = np.abs(y_cal - preds)
        elif self.residual_type == "signed":
            scores = y_cal - preds
        else:
            raise ValueError(f"Unknown residual_type: {self.residual_type}. Choose from 'absolute', 'signed'.")

        self.calibration_scores = np.sort(scores)

        # Finite-sample correction index: ceil((n + 1) * (1 - alpha)) / n
        p = np.clip(np.ceil((n_cal + 1.0) * (1.0 - self.alpha)) / float(n_cal), 0.0, 1.0)
        self.q_hat = float(np.quantile(self.calibration_scores, p, method="higher"))
        return self.q_hat

    def predict_interval(
        self,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate point predictions and lower/upper conformal bounds."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor must be calibrated with calibrate(X_cal, y_cal) before predicting intervals.")

        X_test = np.asarray(X_test)
        if hasattr(self.base_estimator, "predict"):
            point_preds = np.asarray(self.base_estimator.predict(X_test)).squeeze()
        else:
            point_preds = np.asarray(self.base_estimator(X_test)).squeeze()

        lower_bounds = point_preds - self.q_hat
        upper_bounds = point_preds + self.q_hat

        return point_preds, lower_bounds, upper_bounds

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute empirical coverage percentage and mean prediction interval width."""
        y_test = np.asarray(y_test).squeeze()
        preds, lower, upper = self.predict_interval(X_test)

        covered = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": 1.0 - self.alpha,
            "mean_interval_width": mean_width,
            "coverage_gap": empirical_coverage - (1.0 - self.alpha),
        }


def conformal_interval(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """1-Line helper computing distribution-free lower and upper conformal bounds."""
    y_true_cal = np.asarray(y_true_cal).squeeze()
    y_pred_cal = np.asarray(y_pred_cal).squeeze()
    y_pred_test = np.asarray(y_pred_test).squeeze()

    scores = np.abs(y_true_cal - y_pred_cal)
    n = len(scores)
    p = np.clip(np.ceil((n + 1.0) * (1.0 - alpha)) / float(n), 0.0, 1.0)
    q = float(np.quantile(scores, p, method="higher"))

    lower = y_pred_test - q
    upper = y_pred_test + q
    return lower, upper
