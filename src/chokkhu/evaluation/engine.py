from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chokkhu.core.logger import Logger
from chokkhu.core.visualizer import PlotVisualizer
from chokkhu.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    pr_auc_score,
    precision_recall_f1,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)


class EvaluationResult:
    """Encapsulates model evaluation metrics and visualizations."""

    def __init__(
        self,
        task: str,
        metrics: Dict[str, Any],
        confusion_matrix: Optional[np.ndarray] = None,
        classes: Optional[List[Any]] = None,
    ) -> None:
        self.task = task
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix
        self.classes = classes

    def to_dict(self) -> Dict[str, Any]:
        return {**self.metrics, "task": self.task}

    def to_dataframe(self) -> pd.DataFrame:
        clean_metrics = {
            k: v
            for k, v in self.metrics.items()
            if isinstance(v, (int, float, str, bool))
        }
        return pd.DataFrame(list(clean_metrics.items()), columns=["Metric", "Value"])

    def summary(self) -> str:
        lines = [
            "=" * 50,
            f"       CHOKKHU EVALUATION REPORT ({self.task.upper()})",
            "=" * 50,
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<20}: {v:.4f}")
            elif isinstance(v, (int, str)):
                lines.append(f"  {k:<20}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def evaluate(
    model: Any,
    X_test: Any,
    y_test: Any,
    task: str = "auto",
    average: str = "macro",
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
) -> Dict[str, Any]:
    if isinstance(X_test, (pd.DataFrame, pd.Series)):
        X = X_test.values
    else:
        X = np.asarray(X_test)

    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y = y_test.values
    else:
        y = np.asarray(y_test)

    y = y.flatten()

    Logger.info(f"Generating predictions for {len(y)} samples...")
    y_pred = model.predict(X)

    if task == "auto":
        if hasattr(model, "task"):
            task = model.task
        else:
            task = (
                "classification"
                if len(np.unique(y)) <= 20 and not np.issubdtype(y.dtype, np.floating)
                else "regression"
            )

    if save_reports:
        os.makedirs(save_dir, exist_ok=True)

    results: Dict[str, Any] = {}
    cm = None
    classes_list = None

    if task == "classification":
        acc = accuracy_score(y, y_pred)
        prec, rec, f1 = precision_recall_f1(y, y_pred, average=average)
        cm, classes = confusion_matrix(y, y_pred)
        classes_list = classes.tolist()

        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classes": classes_list,
        }

        # Calculate ROC-AUC & Log-Loss if probabilities are supported
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if len(classes) == 2:
                    results["roc_auc"] = roc_auc_score(y, probs)
                    results["pr_auc"] = pr_auc_score(y, probs)
                results["log_loss"] = log_loss(y, probs)
            except Exception:
                pass

        md_text = (
            "## Model Evaluation (Classification)\n"
            f"- **Accuracy**: {acc:.4f}\n"
            f"- **Precision ({average})**: {prec:.4f}\n"
            f"- **Recall ({average})**: {rec:.4f}\n"
            f"- **F1-Score ({average})**: {f1:.4f}\n"
        )
        if "roc_auc" in results:
            md_text += f"- **ROC-AUC**: {results['roc_auc']:.4f}\n"
        PlotVisualizer.display_markdown(md_text)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        PlotVisualizer.save_and_show(
            fig, "confusion_matrix.png", save_dir, save_reports
        )

    elif task == "regression":
        mse = mean_squared_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }

        md_text = (
            "## Model Evaluation (Regression)\n"
            f"- **MSE**: {mse:.4f}\n"
            f"- **RMSE**: {rmse:.4f}\n"
            f"- **MAE**: {mae:.4f}\n"
            f"- **R2-Score**: {r2:.4f}\n"
        )
        PlotVisualizer.display_markdown(md_text)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y, y_pred, alpha=0.5, color="b")
        min_val = float(min(np.min(y), np.min(y_pred)))
        max_val = float(max(np.max(y), np.max(y_pred)))
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        PlotVisualizer.save_and_show(
            fig, "actual_vs_predicted.png", save_dir, save_reports
        )

    else:
        raise ValueError(
            f"Unknown task type: {task}. Use 'classification' or 'regression'."
        )

    Logger.info("Model evaluation completed successfully.")
    return results
