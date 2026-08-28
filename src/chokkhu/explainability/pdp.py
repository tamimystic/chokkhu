from __future__ import annotations

from typing import Tuple
import numpy as np
from chokkhu.models.base import ChokkhuModel


def partial_dependence(
    model: ChokkhuModel,
    X: np.ndarray,
    feature_idx: int,
    grid_resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Partial Dependence values for a single feature across a grid."""
    vals = X[:, feature_idx]
    grid = np.linspace(np.nanmin(vals), np.nanmax(vals), num=grid_resolution)
    avg_predictions: np.ndarray = np.zeros(grid_resolution, dtype=np.float64)

    for idx, v in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = v
        preds = model.predict(X_mod)
        avg_predictions[idx] = float(np.mean(preds))

    return grid, avg_predictions
