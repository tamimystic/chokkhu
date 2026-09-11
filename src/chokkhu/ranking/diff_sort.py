"""Differentiable Sorting, Continuous Permutations, and Soft Ranking Metrics.

Formulated from first principles using NeuralSort continuous relaxations,
soft rank differentials, and differentiable Spearman/NDCG losses in pure NumPy.
"""

from typing import Optional

import numpy as np


class NeuralSort:
    """Continuous Differentiable Sorting Operator and Permutation Relaxer (NeuralSort / SoftSort).

    Parameters
    ----------
    tau : float, default=1.0
        Temperature parameter controlling smoothness of permutation relaxation (tau -> 0 approaches hard sort).
    """

    def __init__(self, tau: float = 1.0) -> None:
        self.tau = max(1e-5, float(tau))

    def soft_ranks(self, scores: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        """Compute smooth continuous 1-based ranks r_i for score vector s."""
        s = np.asarray(scores, dtype=float)
        t = self.tau if tau is None else max(1e-5, float(tau))

        diff = s[..., None, :] - s[..., :, None]
        sigmoid_vals = 1.0 / (1.0 + np.exp(np.clip(-diff / t, -50.0, 50.0)))
        if s.ndim == 1:
            np.fill_diagonal(sigmoid_vals, 0.0)
            ranks = 1.0 + np.sum(sigmoid_vals, axis=-1)
        else:
            for b in range(s.shape[0]):
                np.fill_diagonal(sigmoid_vals[b], 0.0)
            ranks = 1.0 + np.sum(sigmoid_vals, axis=-1)

        return ranks

    def permutation_matrix(
        self, scores: np.ndarray, tau: Optional[float] = None
    ) -> np.ndarray:
        """Compute smooth doubly-stochastic permutation matrix P_tau of shape (..., n, n)."""
        s = np.asarray(scores, dtype=float)
        t = self.tau if tau is None else max(1e-5, float(tau))
        n = s.shape[-1]

        ranks = self.soft_ranks(s, tau=t)
        target_ranks = np.arange(1, n + 1, dtype=float)

        if s.ndim == 1:
            dist = -((ranks[None, :] - target_ranks[:, None]) ** 2) / t
            exp_d = np.exp(dist - np.max(dist, axis=-1, keepdims=True))
            P = exp_d / (np.sum(exp_d, axis=-1, keepdims=True) + 1e-12)
        else:
            dist = -((ranks[:, None, :] - target_ranks[None, :, None]) ** 2) / t
            exp_d = np.exp(dist - np.max(dist, axis=-1, keepdims=True))
            P = exp_d / (np.sum(exp_d, axis=-1, keepdims=True) + 1e-12)

        return P

    def soft_sort(
        self,
        values: np.ndarray,
        scores: Optional[np.ndarray] = None,
        tau: Optional[float] = None,
    ) -> np.ndarray:
        """Continuously sort values according to scores (or values themselves)."""
        v = np.asarray(values, dtype=float)
        s = v if scores is None else np.asarray(scores, dtype=float)
        P = self.permutation_matrix(s, tau=tau)
        return np.matmul(P, v[..., :, None]).squeeze(-1)

    def soft_topk(
        self,
        values: np.ndarray,
        k: int,
        scores: Optional[np.ndarray] = None,
        tau: Optional[float] = None,
    ) -> np.ndarray:
        """Continuously extract the top-K elements."""
        sorted_v = self.soft_sort(values, scores=scores, tau=tau)
        return sorted_v[..., :k]


class DifferentiableRankingLoss:
    """Differentiable Surrogate Losses for Learning-to-Rank (LTR)."""

    @staticmethod
    def spearman_loss(
        pred_scores: np.ndarray,
        true_scores: np.ndarray,
        tau: float = 1.0,
    ) -> float:
        """Compute 1 - Pearson correlation of soft continuous ranks."""
        sorter = NeuralSort(tau=tau)
        pred_r = sorter.soft_ranks(pred_scores)
        true_r = sorter.soft_ranks(true_scores)

        pred_dev = pred_r - np.mean(pred_r)
        true_dev = true_r - np.mean(true_r)

        cov: float = float(np.sum(pred_dev * true_dev))
        var_pred: float = float(np.sum(pred_dev**2))
        var_true: float = float(np.sum(true_dev**2))

        denom = np.sqrt(var_pred * var_true) + 1e-12
        spearman_corr = cov / denom
        return float(1.0 - spearman_corr)

    @staticmethod
    def soft_ndcg_loss(
        pred_scores: np.ndarray,
        true_relevance: np.ndarray,
        k: Optional[int] = None,
        tau: float = 1.0,
    ) -> float:
        """Compute smooth 1 - NDCG@K loss."""
        sorter = NeuralSort(tau=tau)
        pred_r = sorter.soft_ranks(pred_scores)

        y = np.asarray(true_relevance, dtype=float)
        gain = (2.0**y) - 1.0
        discount = 1.0 / np.log2(1.0 + pred_r)

        dcg: float = float(np.sum(gain * discount))

        ideal_r = np.argsort(np.argsort(-y)) + 1.0
        ideal_discount = 1.0 / np.log2(1.0 + ideal_r)
        idcg = np.sum(gain * ideal_discount) + 1e-12

        ndcg = dcg / idcg
        return float(1.0 - np.clip(ndcg, 0.0, 1.0))
