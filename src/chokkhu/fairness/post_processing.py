"""Post-Processing Fair Threshold Optimization.

Pure NumPy implementation of Hardt et al. (2016) Equality of Opportunity in Supervised Learning:
- Optimizes group-specific classification thresholds to satisfy fairness criteria:
    - demographic_parity: equal selection rates across groups
    - equal_opportunity: equal true positive rates across groups
    - equalized_odds: equal true positive and false positive rates
"""

from typing import Dict
import numpy as np


class ThresholdOptimizer:
    r"""Group-Specific Post-Processing Threshold Optimizer.

    Finds thresholds :math:`\{\tau_a\}` such that :math:`\hat{Y} = \mathbb{I}(P(Y=1|X) \ge \tau_A)`
    satisfies target fairness constraints.

    Parameters
    ----------
    constraint : str, default="equal_opportunity"
        Fairness constraint ("demographic_parity", "equal_opportunity", "equalized_odds").
    grid_size : int, default=100
        Number of threshold candidates evaluated.
    """

    def __init__(
        self,
        constraint: str = "equal_opportunity",
        grid_size: int = 100,
    ) -> None:
        if constraint not in (
            "demographic_parity",
            "equal_opportunity",
            "equalized_odds",
        ):
            raise ValueError(f"Unknown constraint: '{constraint}'")
        self.constraint = constraint
        self.grid_size = grid_size
        self.group_thresholds_: Dict[int, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> "ThresholdOptimizer":
        """Optimize group-specific classification thresholds."""
        y_t = np.asarray(y_true, dtype=np.int32).ravel()
        y_p = np.asarray(y_proba, dtype=np.float32).ravel()
        sens = np.asarray(sensitive_features, dtype=np.int32).ravel()

        groups = np.unique(sens)
        candidates = np.linspace(0.05, 0.95, self.grid_size)

        best_score = -1e9
        best_thresholds: Dict[int, float] = {int(g): 0.5 for g in groups}

        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            mask0 = sens == g0
            mask1 = sens == g1

            for t0 in candidates:
                p0 = (y_p[mask0] >= t0).astype(int)
                for t1 in candidates:
                    p1 = (y_p[mask1] >= t1).astype(int)

                    # Compute fairness penalty
                    if self.constraint == "demographic_parity":
                        diff = abs(float(np.mean(p0 == 1)) - float(np.mean(p1 == 1)))
                    elif self.constraint == "equal_opportunity":
                        tpr0 = (
                            float(np.mean(p0[y_t[mask0] == 1] == 1))
                            if np.sum(y_t[mask0] == 1) > 0
                            else 0.0
                        )
                        tpr1 = (
                            float(np.mean(p1[y_t[mask1] == 1] == 1))
                            if np.sum(y_t[mask1] == 1) > 0
                            else 0.0
                        )
                        diff = abs(tpr0 - tpr1)
                    else:  # equalized_odds
                        tpr0 = (
                            float(np.mean(p0[y_t[mask0] == 1] == 1))
                            if np.sum(y_t[mask0] == 1) > 0
                            else 0.0
                        )
                        tpr1 = (
                            float(np.mean(p1[y_t[mask1] == 1] == 1))
                            if np.sum(y_t[mask1] == 1) > 0
                            else 0.0
                        )
                        fpr0 = (
                            float(np.mean(p0[y_t[mask0] == 0] == 1))
                            if np.sum(y_t[mask0] == 0) > 0
                            else 0.0
                        )
                        fpr1 = (
                            float(np.mean(p1[y_t[mask1] == 0] == 1))
                            if np.sum(y_t[mask1] == 0) > 0
                            else 0.0
                        )
                        diff = max(abs(tpr0 - tpr1), abs(fpr0 - fpr1))

                    # Accuracy utility
                    acc0 = float(np.mean(p0 == y_t[mask0]))
                    acc1 = float(np.mean(p1 == y_t[mask1]))
                    total_acc = 0.5 * (acc0 + acc1)

                    # Objective: maximize accuracy minus penalty
                    score = total_acc - 5.0 * diff
                    if score > best_score:
                        best_score = score
                        best_thresholds = {int(g0): float(t0), int(g1): float(t1)}
        else:
            # Fallback for > 2 groups
            for g in groups:
                best_thresholds[int(g)] = 0.5

        self.group_thresholds_ = best_thresholds
        self.is_fitted = True
        return self

    def predict(
        self,
        y_proba: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Apply fair group-specific thresholds to predicted probabilities."""
        if not self.is_fitted:
            raise RuntimeError("ThresholdOptimizer must be fitted before predict.")

        y_p = np.asarray(y_proba, dtype=np.float32).ravel()
        sens = np.asarray(sensitive_features, dtype=np.int32).ravel()

        y_pred: np.ndarray = np.zeros(len(y_p), dtype=np.int32)
        for i in range(len(y_p)):
            g = int(sens[i])
            thresh = self.group_thresholds_.get(g, 0.5)
            y_pred[i] = 1 if y_p[i] >= thresh else 0

        return y_pred
