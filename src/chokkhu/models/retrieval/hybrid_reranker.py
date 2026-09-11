from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class HybridReranker:
    """Hybrid Sparse-Dense Search Reranker & Rank Fusion Engine in pure NumPy.

    Combines dense neural/vector retrieval with sparse lexical BM25 retrieval
    using Reciprocal Rank Fusion (RRF) and convex normalized score blending.
    """

    def __init__(self, rrf_k: int = 60, alpha: float = 0.5) -> None:
        self.rrf_k = int(rrf_k)
        self.alpha = float(alpha)

    def reciprocal_rank_fusion(
        self,
        ranked_lists: Sequence[Sequence[int]],
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Combines multiple ranked doc index lists into a single fused ranking via RRF."""
        if not ranked_lists:
            return []

        if weights is None:
            weights = [1.0] * len(ranked_lists)
        elif len(weights) != len(ranked_lists):
            raise ValueError("Length of weights must match length of ranked_lists.")

        scores: Dict[int, float] = {}

        for weight, r_list in zip(weights, ranked_lists):
            for rank, doc_idx in enumerate(r_list, start=1):
                rrf_score = weight / (self.rrf_k + rank)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + rrf_score

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]

        return sorted_docs

    def blend_scores(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Blends dense and sparse score arrays via min-max normalized convex combination."""
        a = self.alpha if alpha is None else float(alpha)
        d_arr = np.asarray(dense_scores, dtype=np.float64)
        s_arr = np.asarray(sparse_scores, dtype=np.float64)

        if d_arr.shape != s_arr.shape:
            raise ValueError(
                f"Dense scores shape {d_arr.shape} must match sparse shape {s_arr.shape}"
            )

        def _min_max_norm(x: np.ndarray) -> np.ndarray:
            ptp: float = float(np.ptp(x))
            if ptp < 1e-12:
                return np.ones_like(x) * 0.5
            return (x - np.min(x)) / ptp

        norm_d = _min_max_norm(d_arr)
        norm_s = _min_max_norm(s_arr)

        return a * norm_d + (1.0 - a) * norm_s
