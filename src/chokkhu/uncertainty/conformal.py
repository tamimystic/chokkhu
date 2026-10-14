"""Distribution-Free Conformal Prediction and Uncertainty Quantification from First Principles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SplitConformalRegressor:
    """Split Conformal Regressor guaranteeing finite-sample marginal coverage: P(Y in C(X)) >= 1 - alpha."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.quantile_: float = 0.0
        self.is_calibrated_: bool = False

    def calibrate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> SplitConformalRegressor:
        """Calibrate conformal residual cutoff on holdout calibration data."""
        y_t = np.asarray(y_true, dtype=np.float64).flatten()
        y_p = np.asarray(y_pred, dtype=np.float64).flatten()
        if len(y_t) != len(y_p):
            raise ValueError("y_true and y_pred must have identical length")

        n = len(y_t)
        residuals = np.abs(y_t - y_p)
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
        self.quantile_ = float(np.sort(residuals)[k - 1])
        self.is_calibrated_ = True
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate calibrated prediction intervals [y_pred - q, y_pred + q]."""
        if not self.is_calibrated_:
            raise ValueError(
                "Conformal regressor must be calibrated before predicting intervals."
            )
        y_p = np.asarray(y_pred, dtype=np.float64)
        return y_p - self.quantile_, y_p + self.quantile_

    def evaluate_coverage(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute empirical coverage and mean interval width."""
        y_t = np.asarray(y_true, dtype=np.float64).flatten()
        lower, upper = self.predict_interval(y_pred)
        covered = (y_t >= lower) & (y_t <= upper)
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(covered)),
            "average_width": float(np.mean(upper - lower)),
        }


class ConformalClassifier:
    """Conformal Classification with Adaptive Prediction Sets (APS) or Least Ambiguous Sets (LAC)."""

    def __init__(
        self,
        alpha: float = 0.1,
        method: str = "aps",
        random_state: Optional[int] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.method = method.lower()
        self.random_state = random_state
        self.quantile_: float = 0.0
        self.is_calibrated_: bool = False

    def calibrate(
        self, probs_cal: np.ndarray, y_cal: np.ndarray
    ) -> ConformalClassifier:
        """Calibrate non-conformity score quantile on calibration set."""
        probs = np.asarray(probs_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.int64).flatten()
        n, num_classes = probs.shape

        rng = np.random.RandomState(self.random_state)
        scores = np.zeros(n, dtype=np.float64)

        if self.method == "lac":
            for i in range(n):
                scores[i] = 1.0 - probs[i, y[i]]
        elif self.method == "aps":
            # Adaptive Prediction Sets (Romano et al., 2020)
            u = rng.uniform(0.0, 1.0, size=n)
            for i in range(n):
                order = np.argsort(-probs[i])
                rank = int(np.where(order == y[i])[0][0])
                cumsum = float(np.sum(probs[i, order[:rank]]))
                scores[i] = cumsum + u[i] * probs[i, y[i]]
        else:
            raise ValueError(f"Unknown conformal classification method: {self.method}")

        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
        self.quantile_ = float(np.sort(scores)[k - 1])
        self.is_calibrated_ = True
        return self

    def predict_set(self, probs: np.ndarray) -> List[List[int]]:
        """Construct prediction sets for test instances."""
        if not self.is_calibrated_:
            raise ValueError("Classifier must be calibrated first.")
        P = np.asarray(probs, dtype=np.float64)
        n, num_classes = P.shape
        prediction_sets: List[List[int]] = []

        if self.method == "lac":
            for i in range(n):
                pset = [
                    c for c in range(num_classes) if 1.0 - P[i, c] <= self.quantile_
                ]
                prediction_sets.append(pset if pset else [int(np.argmax(P[i]))])
        elif self.method == "aps":
            for i in range(n):
                order = np.argsort(-P[i])
                sorted_p = P[i, order]
                cum = 0.0
                pset = []
                for idx, c in enumerate(order):
                    pset.append(int(c))
                    cum += sorted_p[idx]
                    if cum >= self.quantile_:
                        break
                prediction_sets.append(pset)
        return prediction_sets

    def evaluate_coverage(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute empirical coverage rate and mean set size."""
        psets = self.predict_set(probs)
        y = np.asarray(y_true, dtype=np.int64).flatten()
        covered = [y[i] in psets[i] for i in range(len(y))]
        sizes = [len(s) for s in psets]
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(covered)),
            "average_set_size": float(np.mean(sizes)),
        }


class MondrianConformalClassifier:
    """Group-Conditional (Mondrian) Conformal Prediction with exact coverage per class."""

    def __init__(self, alpha: float = 0.1, method: str = "lac") -> None:
        self.alpha = float(alpha)
        self.method = method.lower()
        self.calibrators_: Dict[int, ConformalClassifier] = {}
        self.is_calibrated_: bool = False

    def calibrate(
        self, probs_cal: np.ndarray, y_cal: np.ndarray
    ) -> MondrianConformalClassifier:
        """Calibrate separate conformal predictors per class label."""
        probs = np.asarray(probs_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.int64).flatten()
        classes = np.unique(y)

        for c in classes:
            mask = y == c
            clf = ConformalClassifier(alpha=self.alpha, method=self.method)
            clf.calibrate(probs[mask], y[mask])
            self.calibrators_[int(c)] = clf

        self.is_calibrated_ = True
        return self

    def predict_set(self, probs: np.ndarray) -> List[List[int]]:
        """Construct group-calibrated prediction sets."""
        if not self.is_calibrated_:
            raise ValueError("Mondrian classifier must be calibrated first.")
        P = np.asarray(probs, dtype=np.float64)
        n = P.shape[0]
        prediction_sets: List[List[int]] = []

        for i in range(n):
            pset = []
            for c, clf in self.calibrators_.items():
                if 1.0 - P[i, c] <= clf.quantile_:
                    pset.append(c)
            prediction_sets.append(pset if pset else [int(np.argmax(P[i]))])
        return prediction_sets

    def evaluate_coverage(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Compute overall coverage and per-class coverage breakdown."""
        psets = self.predict_set(probs)
        y = np.asarray(y_true, dtype=np.int64).flatten()
        covered = np.array([y[i] in psets[i] for i in range(len(y))])

        class_cov: Dict[int, float] = {}
        for c in np.unique(y):
            mask = y == c
            class_cov[int(c)] = float(np.mean(covered[mask]))

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(covered)),
            "average_set_size": float(np.mean([len(s) for s in psets])),
            "per_class_coverage": class_cov,
        }
