"""Recommender Evaluation & Ranking Metrics.

Pure NumPy implementations of standard top-K ranking evaluation metrics:
- HitRate@K (HR@K)
- Normalized Discounted Cumulative Gain@K (NDCG@K)
- Mean Reciprocal Rank@K (MRR@K)
- Precision@K
- Recall@K
- Mean Average Precision@K (MAP@K)
"""

from typing import Any, List, Set, Union
import numpy as np


def hit_rate_at_k(
    actual: Union[Any, List[Any], Set[Any]], predicted: List[Any], k: int = 10
) -> float:
    """Calculate Hit Rate at K (HR@K)."""
    top_k = predicted[:k]
    if isinstance(actual, (list, set, tuple)):
        return 1.0 if any(item in actual for item in top_k) else 0.0
    return 1.0 if actual in top_k else 0.0


def ndcg_at_k(
    actual: Union[Any, List[Any], Set[Any]], predicted: List[Any], k: int = 10
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    top_k = predicted[:k]
    actual_set = set(actual) if isinstance(actual, (list, set, tuple)) else {actual}

    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(idx + 2.0)

    # Ideal DCG
    n_hits = min(k, len(actual_set))
    if n_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(idx + 2.0) for idx in range(n_hits))

    return float(dcg / idcg)


def mrr_at_k(
    actual: Union[Any, List[Any], Set[Any]], predicted: List[Any], k: int = 10
) -> float:
    """Calculate Mean Reciprocal Rank at K (MRR@K)."""
    top_k = predicted[:k]
    actual_set = set(actual) if isinstance(actual, (list, set, tuple)) else {actual}

    for idx, item in enumerate(top_k):
        if item in actual_set:
            return float(1.0 / (idx + 1.0))
    return 0.0


def precision_at_k(
    actual: Union[List[Any], Set[Any]], predicted: List[Any], k: int = 10
) -> float:
    """Calculate Precision at K."""
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    actual_set = set(actual) if isinstance(actual, (list, set, tuple)) else {actual}
    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / len(top_k))


def recall_at_k(
    actual: Union[List[Any], Set[Any]], predicted: List[Any], k: int = 10
) -> float:
    """Calculate Recall at K."""
    top_k = predicted[:k]
    actual_set = set(actual) if isinstance(actual, (list, set, tuple)) else {actual}
    if not actual_set:
        return 0.0
    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / len(actual_set))


def map_at_k(
    actual_list: List[List[Any]], predicted_list: List[List[Any]], k: int = 10
) -> float:
    """Calculate Mean Average Precision at K across multiple query evaluations."""
    if not actual_list or not predicted_list or len(actual_list) != len(predicted_list):
        raise ValueError(
            "actual_list and predicted_list must be non-empty and have matching lengths."
        )

    ap_scores: List[float] = []
    for actual, predicted in zip(actual_list, predicted_list):
        top_k = predicted[:k]
        actual_set = set(actual) if isinstance(actual, (list, set, tuple)) else {actual}
        if not actual_set:
            continue

        score = 0.0
        num_hits = 0.0
        for idx, item in enumerate(top_k):
            if item in actual_set:
                num_hits += 1.0
                score += num_hits / (idx + 1.0)

        ap = score / min(len(actual_set), k)
        ap_scores.append(ap)

    return float(np.mean(ap_scores)) if ap_scores else 0.0
