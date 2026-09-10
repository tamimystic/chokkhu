"""Sovereign Fast Matrix Profile (STOMP / STAMP) for Motif & Discord Discovery in Pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class MatrixProfile:
    """Fast Time Series Matrix Profile for Motif & Anomaly Discovery."""

    def __init__(
        self,
        window_size: int = 20,
        exclusion_zone: Optional[float] = 0.5,
    ) -> None:
        self.m = window_size
        self.exclusion_zone = exclusion_zone  # fraction of window size
        self.profile: Optional[np.ndarray] = None
        self.profile_index: Optional[np.ndarray] = None
        self.ts: Optional[np.ndarray] = None

    def compute(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the matrix profile distance vector P and index vector profile_idx."""
        ts = np.asarray(ts, dtype=np.float64).squeeze()
        self.ts = ts
        n = len(ts)
        m = self.m

        if n < m * 2:
            raise ValueError(f"Time series length ({n}) must be at least twice window size ({m}).")

        k = n - m + 1
        dist_profile: np.ndarray = np.full(k, np.inf, dtype=np.float64)
        profile_idx: np.ndarray = np.full(k, -1, dtype=np.int64)

        # Precompute rolling mean and rolling std for all subsequences of length m
        cum_ts = np.concatenate([[0.0], np.cumsum(ts)])
        cum_ts2 = np.concatenate([[0.0], np.cumsum(ts ** 2)])

        sub_sums = cum_ts[m:] - cum_ts[:-m]
        sub_sums2 = cum_ts2[m:] - cum_ts2[:-m]

        means = sub_sums / float(m)
        variances = np.maximum(0.0, (sub_sums2 / float(m)) - (means ** 2))
        stds = np.sqrt(variances)
        stds[stds < 1e-10] = 1e-10

        # Subsequence matrix (k x m)
        subseqs = np.lib.stride_tricks.sliding_window_view(ts, window_shape=m)
        # Z-normalize subsequences
        norm_subseqs = (subseqs - means[:, None]) / stds[:, None]

        # Exclusion zone radius
        ex_radius = int(m * self.exclusion_zone) if self.exclusion_zone is not None else int(m // 2)

        # Compute pairwise distance matrix using vectorized dot products
        # d^2 = 2 * m * (1 - (Q / m))
        dot_matrix = np.dot(norm_subseqs, norm_subseqs.T)
        dist_matrix = np.sqrt(np.maximum(0.0, 2.0 * m * (1.0 - (dot_matrix / float(m)))))

        # Apply exclusion zone on diagonals
        for i in range(k):
            low = max(0, i - ex_radius)
            high = min(k, i + ex_radius + 1)
            dist_matrix[i, low:high] = np.inf

        # Find 1-NN
        dist_profile = np.min(dist_matrix, axis=1)
        profile_idx = np.argmin(dist_matrix, axis=1)

        self.profile = dist_profile
        self.profile_index = profile_idx
        return dist_profile, profile_idx

    def find_motifs(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Identify top-k most repeating subsequence pairs (motifs)."""
        if self.profile is None or self.profile_index is None:
            raise ValueError("Must run compute(ts) before finding motifs.")

        k = len(self.profile)
        ex_radius = int(self.m * (self.exclusion_zone or 0.5))
        p_copy = np.copy(self.profile)

        motifs = []
        for _ in range(top_k):
            if np.all(np.isinf(p_copy)):
                break
            idx1 = int(np.argmin(p_copy))
            idx2 = int(self.profile_index[idx1])
            dist = float(p_copy[idx1])

            motifs.append({
                "index1": idx1,
                "index2": idx2,
                "distance": dist,
                "subsequence1": self.ts[idx1 : idx1 + self.m].tolist() if self.ts is not None else [],
                "subsequence2": self.ts[idx2 : idx2 + self.m].tolist() if self.ts is not None else [],
            })

            # Mask out both neighborhoods to find distinct motifs
            for idx in [idx1, idx2]:
                low = max(0, idx - ex_radius)
                high = min(k, idx + ex_radius + 1)
                p_copy[low:high] = np.inf

        return motifs

    def find_discords(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Identify top-k most unusual/anomalous subsequences (discords)."""
        if self.profile is None:
            raise ValueError("Must run compute(ts) before finding discords.")

        k = len(self.profile)
        ex_radius = int(self.m * (self.exclusion_zone or 0.5))
        p_copy = np.copy(self.profile)
        p_copy[np.isinf(p_copy)] = -1.0

        discords = []
        for _ in range(top_k):
            if np.all(p_copy <= 0.0):
                break
            idx = int(np.argmax(p_copy))
            dist = float(p_copy[idx])

            discords.append({
                "index": idx,
                "distance": dist,
                "subsequence": self.ts[idx : idx + self.m].tolist() if self.ts is not None else [],
            })

            low = max(0, idx - ex_radius)
            high = min(k, idx + ex_radius + 1)
            p_copy[low:high] = -1.0

        return discords


def find_motifs(ts: np.ndarray, window_size: int = 20, top_k: int = 3) -> List[Dict[str, Any]]:
    """1-Line helper to discover motifs in a time series."""
    mp = MatrixProfile(window_size=window_size)
    mp.compute(ts)
    return mp.find_motifs(top_k=top_k)


def find_discords(ts: np.ndarray, window_size: int = 20, top_k: int = 3) -> List[Dict[str, Any]]:
    """1-Line helper to discover anomaly discords in a time series."""
    mp = MatrixProfile(window_size=window_size)
    mp.compute(ts)
    return mp.find_discords(top_k=top_k)
