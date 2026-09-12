"""Covariate Shift, Concept Drift & Maximum Mean Discrepancy (MMD) Hypothesis Testing in pure NumPy/SciPy.

References:
- Gretton et al. (2012): "A Kernel Two-Sample Test" (JMLR).
- Rabanser et al. (2019): "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" (NeurIPS).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


class MaximumMeanDiscrepancyTest:
    """Non-parametric Maximum Mean Discrepancy (MMD) Kernel Two-Sample Test for Distribution Drift.

    Computes unbiased MMD^2 statistic between reference distribution P and production distribution Q
    using an RBF Gaussian kernel with automatic median heuristic bandwidth selection and permutation bootstrap p-values.
    """

    def __init__(
        self,
        gamma: Optional[float] = None,
        num_permutations: int = 100,
        unbiased: bool = True,
        random_state: int = 42,
    ) -> None:
        self.gamma = gamma
        self.num_permutations = int(num_permutations)
        self.unbiased = bool(unbiased)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def _compute_gamma(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate RBF bandwidth gamma = 1 / (2 * median_distance^2) via median heuristic."""
        if self.gamma is not None:
            return float(self.gamma)

        # Subsample if large for speed
        n_sub = min(len(X), 200)
        m_sub = min(len(Y), 200)
        idx_x = self.rng.choice(len(X), n_sub, replace=False)
        idx_y = self.rng.choice(len(Y), m_sub, replace=False)

        combined = np.vstack([X[idx_x], Y[idx_y]])
        dists = cdist(combined, combined, metric="sqeuclidean")
        triu_dists = dists[np.triu_indices_from(dists, k=1)]
        med_sq_dist = np.median(triu_dists)
        if med_sq_dist <= 1e-12:
            return 1.0
        return float(1.0 / (2.0 * med_sq_dist))

    def _rbf_kernel(self, A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix K(A, B) = exp(-gamma * ||A_i - B_j||^2)."""
        dists_sq = cdist(A, B, metric="sqeuclidean")
        return np.exp(-gamma * dists_sq)

    def compute_mmd_squared(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Compute MMD^2 statistic between reference dataset X and target dataset Y.

        Returns:
            mmd_squared: MMD^2 test statistic
            gamma: Bandwidth parameter used
        """
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(Y, dtype=np.float64)
        n = x_arr.shape[0]
        m = y_arr.shape[0]

        gamma = self._compute_gamma(x_arr, y_arr)

        K_xx = self._rbf_kernel(x_arr, x_arr, gamma)
        K_yy = self._rbf_kernel(y_arr, y_arr, gamma)
        K_xy = self._rbf_kernel(x_arr, y_arr, gamma)

        if self.unbiased:
            # Zero out diagonals for unbiased U-statistic
            np.fill_diagonal(K_xx, 0.0)
            np.fill_diagonal(K_yy, 0.0)
            term_xx = np.sum(K_xx) / float(n * (n - 1)) if n > 1 else 0.0
            term_yy = np.sum(K_yy) / float(m * (m - 1)) if m > 1 else 0.0
            term_xy = 2.0 * np.sum(K_xy) / float(n * m)
            mmd_sq = term_xx + term_yy - term_xy
        else:
            term_xx = np.sum(K_xx) / float(n * n)
            term_yy = np.sum(K_yy) / float(m * m)
            term_xy = 2.0 * np.sum(K_xy) / float(n * m)
            mmd_sq = term_xx + term_yy - term_xy

        return float(mmd_sq), gamma

    def test(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Union[float, bool]]:
        """Perform MMD Two-Sample Hypothesis Test with permutation bootstrap p-value.

        Null hypothesis H_0: P_ref == P_test (no drift)
        Alternative H_1: P_ref != P_test (drift detected)

        Returns:
            Dictionary containing:
            - 'mmd_squared': float
            - 'p_value': float
            - 'drift_detected': bool (p_value < alpha)
            - 'threshold_alpha': float
        """
        x_ref = np.asarray(X_ref, dtype=np.float64)
        x_test = np.asarray(X_test, dtype=np.float64)
        n, m = len(x_ref), len(x_test)

        observed_stat, gamma = self.compute_mmd_squared(x_ref, x_test)

        # Pooled sample kernel matrix for fast permutation evaluation
        pooled = np.vstack([x_ref, x_test])
        K_pooled = self._rbf_kernel(pooled, pooled, gamma)
        if self.unbiased:
            np.fill_diagonal(K_pooled, 0.0)

        total_N = n + m
        perm_stats: List[float] = []

        for _ in range(self.num_permutations):
            perm = self.rng.permutation(total_N)
            idx_1 = perm[:n]
            idx_2 = perm[n:]

            K_11 = K_pooled[np.ix_(idx_1, idx_1)]
            K_22 = K_pooled[np.ix_(idx_2, idx_2)]
            K_12 = K_pooled[np.ix_(idx_1, idx_2)]

            if self.unbiased:
                s11 = np.sum(K_11) / float(n * (n - 1)) if n > 1 else 0.0
                s22 = np.sum(K_22) / float(m * (m - 1)) if m > 1 else 0.0
                s12 = 2.0 * np.sum(K_12) / float(n * m)
                perm_stat = s11 + s22 - s12
            else:
                s11 = np.sum(K_11) / float(n * n)
                s22 = np.sum(K_22) / float(m * m)
                s12 = 2.0 * np.sum(K_12) / float(n * m)
                perm_stat = s11 + s22 - s12

            perm_stats.append(perm_stat)

        perm_arr = np.array(perm_stats, dtype=np.float64)
        p_val = float(np.mean(perm_arr >= observed_stat))

        return {
            "mmd_squared": float(observed_stat),
            "p_value": float(p_val),
            "drift_detected": bool(p_val < float(alpha)),
            "threshold_alpha": float(alpha),
        }


class PopulationStabilityIndex:
    """Population Stability Index (PSI) for Feature Drift Detection.

    Measures shift in binned empirical probability distributions between reference and production data.
    PSI = sum_i (P_i - Q_i) * ln(P_i / Q_i)
    """

    def __init__(self, num_bins: int = 10, eps: float = 1e-6) -> None:
        self.num_bins = int(num_bins)
        self.eps = float(eps)

    def compute_feature_psi(
        self, ref_feature: np.ndarray, test_feature: np.ndarray
    ) -> float:
        """Compute PSI for a single 1D numerical feature."""
        x_ref = np.asarray(ref_feature, dtype=np.float64)
        x_test = np.asarray(test_feature, dtype=np.float64)

        if len(x_ref) == 0 or len(x_test) == 0:
            return 0.0

        # Quantile bin edges based on reference distribution
        percentiles = np.linspace(0.0, 100.0, self.num_bins + 1)
        bin_edges = np.percentile(x_ref, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        ref_counts, _ = np.histogram(x_ref, bins=bin_edges)
        test_counts, _ = np.histogram(x_test, bins=bin_edges)

        # Proportions with Laplace smoothing / pseudocount
        n_bins = len(ref_counts)
        P = (ref_counts.astype(np.float64) + 1.0) / float(len(x_ref) + n_bins)
        Q = (test_counts.astype(np.float64) + 1.0) / float(len(x_test) + n_bins)

        # PSI = sum (P - Q) * ln(P / Q)
        psi_val: float = float(np.sum((P - Q) * np.log(P / Q)))
        return psi_val

    def compute_dataset_psi(
        self, X_ref: np.ndarray, X_test: np.ndarray
    ) -> Dict[str, Union[float, List[float], str]]:
        """Compute feature-wise and average PSI across full datasets."""
        x_ref = np.asarray(X_ref, dtype=np.float64)
        x_test = np.asarray(X_test, dtype=np.float64)

        n_features = x_ref.shape[1]
        psi_per_feature = [
            self.compute_feature_psi(x_ref[:, d], x_test[:, d])
            for d in range(n_features)
        ]
        mean_psi = float(np.mean(psi_per_feature))

        if mean_psi < 0.1:
            severity = "no_drift"
        elif mean_psi < 0.2:
            severity = "moderate_drift"
        else:
            severity = "significant_drift"

        return {
            "mean_psi": mean_psi,
            "feature_psi": psi_per_feature,
            "severity": severity,
            "drift_detected": bool(mean_psi >= 0.1),
        }
