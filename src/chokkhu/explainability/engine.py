from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger
from chokkhu.models.base import ChokkhuModel
from .importance import permutation_feature_importance
from .shap import kernel_shap
from .pdp import partial_dependence


class ExplanationResult:
    """Encapsulates model interpretability and explainability results."""

    def __init__(
        self,
        method: str,
        feature_names: List[str],
        importances: Optional[np.ndarray] = None,
        importance_std: Optional[np.ndarray] = None,
        shap_values: Optional[np.ndarray] = None,
        expected_value: Optional[float] = None,
        pdp_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.feature_names = feature_names
        self.importances = importances
        self.importance_std = importance_std
        self.shap_values = shap_values
        self.expected_value = expected_value
        self.pdp_data = pdp_data or {}

    def to_dataframe(self) -> pd.DataFrame:
        """Converts feature rankings to a pandas DataFrame."""
        if self.importances is not None:
            df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.importances,
                }
            )
            if self.importance_std is not None:
                df["std"] = self.importance_std
            return df.sort_values(by="importance", ascending=False).reset_index(
                drop=True
            )
        elif self.shap_values is not None:
            mean_abs = np.mean(np.abs(self.shap_values), axis=0)
            df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                }
            )
            return df.sort_values(by="mean_abs_shap", ascending=False).reset_index(
                drop=True
            )
        return pd.DataFrame()

    def summary(self) -> str:
        """Returns a formatted textual summary of the explanation."""
        lines = [
            "=" * 60,
            f"          CHOKKHU EXPLAINABILITY REPORT ({self.method.upper()})",
            "=" * 60,
        ]
        df = self.to_dataframe()
        if not df.empty:
            for idx, row in df.iterrows():
                val_col = (
                    "importance" if "importance" in df.columns else "mean_abs_shap"
                )
                lines.append(
                    f"  {idx + 1:2d}. {row['feature']:<25}: {row[val_col]:.4f}"
                )
        elif self.pdp_data:
            feat = self.pdp_data.get("feature", "unknown")
            lines.append(f"  Partial Dependence computed for feature: '{feat}'")
            lines.append(
                f"  Grid points evaluated: {len(self.pdp_data.get('grid_values', []))}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def explain(
    model: ChokkhuModel,
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "feature_importance",
    feature_names: Optional[List[str]] = None,
    task: str = "auto",
    n_repeats: int = 5,
    n_samples: int = 100,
    pdp_feature: Optional[Union[int, str]] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> ExplanationResult:
    """Unified entry point for Explainable AI (XAI) in Chokkhu.

    Supports:
    - method="feature_importance": Permutation Feature Importance
    - method="shap": KernelSHAP model-agnostic feature attribution
    - method="pdp": Partial Dependence Plot calculation
    """
    if verbose:
        Logger.info(f"Generating explanation using method: '{method}'")

    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y) if y is not None else None

    if method in ("feature_importance", "permutation"):
        if y_arr is None:
            raise ValueError(
                "Target 'y' is required for permutation feature importance."
            )
        return permutation_feature_importance(
            model=model,
            X=X_arr,
            y=y_arr,
            feature_names=feature_names,
            task=task,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    elif method in ("shap", "kernel_shap"):
        return kernel_shap(
            model=model,
            X=X_arr,
            feature_names=feature_names,
            n_samples=n_samples,
            random_state=random_state,
        )
    elif method in ("pdp", "partial_dependence"):
        target_idx = 0
        target_name = feature_names[0]
        if pdp_feature is not None:
            if isinstance(pdp_feature, str) and pdp_feature in feature_names:
                target_idx = feature_names.index(pdp_feature)
                target_name = pdp_feature
            elif isinstance(pdp_feature, int):
                target_idx = pdp_feature
                target_name = feature_names[pdp_feature]
        grid_vals, avg_preds = partial_dependence(
            model=model, X=X_arr, feature_idx=target_idx
        )
        return ExplanationResult(
            method="pdp",
            feature_names=feature_names,
            pdp_data={
                "feature": target_name,
                "feature_idx": target_idx,
                "grid_values": grid_vals,
                "average_predictions": avg_preds,
            },
        )
    else:
        raise ValueError(
            f"Unsupported explanation method: '{method}'. Use 'feature_importance', 'shap', or 'pdp'."
        )
