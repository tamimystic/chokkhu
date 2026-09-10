"""Pre-Processing Bias Mitigation: Reweighing & Disparate Impact Remover.

Pure NumPy implementations of:
- ReweighingTransformer: Computes instance sample weights to eliminate group-outcome correlation
- DisparateImpactRemover: CDF repair algorithm shifting group feature distributions towards median
"""

from typing import Dict, Optional, Tuple
import numpy as np


class ReweighingTransformer:
    r"""Reweighing Sample Weight Pre-processing Transformer (Kamiran & Calders, 2012).

    Assigns weight to each instance:

    .. math::
        W(A=a, Y=y) = \frac{P(A=a) P(Y=y)}{P(A=a, Y=y)}

    so that :math:`A` and :math:`Y` become statistically independent in weighted training.
    """

    def __init__(self) -> None:
        self.weights_dict_: Dict[Tuple[int, int], float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> "ReweighingTransformer":
        """Compute reweighing factor table for all (A, Y) combinations."""
        y_arr = np.asarray(y, dtype=np.int32).ravel()
        a_arr = np.asarray(sensitive_features, dtype=np.int32).ravel()

        if len(y_arr) != len(a_arr):
            raise ValueError("y and sensitive_features must have identical length.")

        n = len(y_arr)
        if n == 0:
            raise ValueError("Input data cannot be empty.")

        groups = np.unique(a_arr)
        labels = np.unique(y_arr)

        self.weights_dict_ = {}

        for a in groups:
            p_a = float(np.sum(a_arr == a)) / n
            for y_val in labels:
                p_y = float(np.sum(y_arr == y_val)) / n
                p_ay = float(np.sum((a_arr == a) & (y_arr == y_val))) / n

                if p_ay > 1e-12:
                    w = (p_a * p_y) / p_ay
                else:
                    w = 1.0

                self.weights_dict_[(int(a), int(y_val))] = float(w)

        self.is_fitted = True
        return self

    def transform(
        self,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Return array of sample weights for given (y, A)."""
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before transform.")

        y_arr = np.asarray(y, dtype=np.int32).ravel()
        a_arr = np.asarray(sensitive_features, dtype=np.int32).ravel()

        weights: np.ndarray = np.zeros(len(y_arr), dtype=np.float32)
        for i in range(len(y_arr)):
            key = (int(a_arr[i]), int(y_arr[i]))
            weights[i] = self.weights_dict_.get(key, 1.0)

        return weights

    def fit_transform(
        self,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Fit and return instance sample weights."""
        self.fit(y, sensitive_features)
        return self.transform(y, sensitive_features)


class DisparateImpactRemover:
    r"""Disparate Impact Remover Feature Repair Transformer (Feldman et al., 2015).

    Repairs continuous features by moving protected group CDFs towards the median CDF
    with repair level :math:`\lambda \in [0.0, 1.0]`:

    .. math::
        \tilde{X} = (1 - \lambda) X + \lambda F_{\text{median}}^{-1}(F_A(X))
    """

    def __init__(self, repair_level: float = 1.0) -> None:
        if not (0.0 <= repair_level <= 1.0):
            raise ValueError("repair_level must be between 0.0 and 1.0")
        self.repair_level = float(repair_level)
        self.group_quantiles_: Dict[int, np.ndarray] = {}
        self.median_quantiles_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit_transform(
        self,
        X: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Repair features across sensitive groups."""
        X_arr = np.asarray(X, dtype=np.float32).copy()
        sens = np.asarray(sensitive_features, dtype=np.int32).ravel()

        if len(X_arr) != len(sens):
            raise ValueError("X and sensitive_features must have identical length.")

        groups = np.unique(sens)
        if len(groups) < 2 or self.repair_level == 0.0:
            return X_arr

        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 1
        X_2d = X_arr if X_arr.ndim == 2 else X_arr[:, None]
        X_repaired = X_2d.copy()

        for j in range(n_features):
            col = X_2d[:, j]
            # Overall median distribution quantiles
            qs = np.linspace(0, 100, 101)
            med_q = np.percentile(col, qs)

            for g in groups:
                mask = sens == g
                if np.sum(mask) == 0:
                    continue
                group_vals = col[mask]
                # Rank of each element in group in [0, 100]
                ranks = (
                    np.searchsorted(np.sort(group_vals), group_vals)
                    / max(len(group_vals) - 1, 1)
                    * 100.0
                )
                target_vals = np.interp(ranks, qs, med_q)

                repaired = (
                    1.0 - self.repair_level
                ) * group_vals + self.repair_level * target_vals
                X_repaired[mask, j] = repaired

        return X_repaired if X_arr.ndim == 2 else X_repaired.ravel()
