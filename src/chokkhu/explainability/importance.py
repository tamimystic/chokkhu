from __future__ import annotations

from typing import Any, List
import numpy as np
from chokkhu.models.base import ChokkhuModel
from chokkhu.evaluation.metrics import accuracy_score, r2_score


def permutation_feature_importance(
    model: ChokkhuModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task: str = "auto",
    n_repeats: int = 5,
    random_state: int = 42,
) -> Any:
    from .engine import ExplanationResult

    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    if task == "auto":
        task = "regression" if np.issubdtype(y.dtype, np.floating) else "classification"

    baseline_preds = model.predict(X)
    if task == "classification":
        base_score = accuracy_score(y, baseline_preds)
    else:
        base_score = r2_score(y, baseline_preds)

    importances = np.zeros(n_features, dtype=np.float64)
    importances_std = np.zeros(n_features, dtype=np.float64)

    for col_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            shuffled_col = rng.permutation(X_shuffled[:, col_idx])
            X_shuffled[:, col_idx] = shuffled_col

            preds_shuffled = model.predict(X_shuffled)
            if task == "classification":
                score_shuffled = accuracy_score(y, preds_shuffled)
            else:
                score_shuffled = r2_score(y, preds_shuffled)

            drop = base_score - score_shuffled
            drops.append(drop)

        importances[col_idx] = np.mean(drops)
        importances_std[col_idx] = np.std(drops)

    # Normalize to positive contributions
    importances = np.maximum(importances, 0.0)
    total: float = float(np.sum(importances))
    if total > 0:
        importances_norm = importances / total
    else:
        importances_norm = importances

    return ExplanationResult(
        method="feature_importance",
        feature_names=feature_names,
        importances=importances_norm,
        importance_std=importances_std,
    )
