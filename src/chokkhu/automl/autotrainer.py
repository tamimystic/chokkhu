"""Sovereign AutoTrainer: Zero-Dependency End-to-End Automated Machine Learning Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger
from chokkhu.models.base import ChokkhuModel
from chokkhu.models.engine import train


class AutoMLResult:
    """Encapsulates AutoML run results, leaderboard, and best model."""

    def __init__(
        self,
        best_model: ChokkhuModel,
        best_model_name: str,
        best_score: float,
        leaderboard: pd.DataFrame,
        task: str,
    ) -> None:
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_score = best_score
        self.leaderboard = leaderboard
        self.task = task

    def predict(self, X: Any) -> np.ndarray:
        return self.best_model.predict(X)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "              CHOKKHU AUTO-ML LEADERBOARD",
            "=" * 60,
            f"  Detected Task : {self.task.upper()}",
            f"  Winning Model : {self.best_model_name.upper()} (Score: {self.best_score:.4f})",
            "-" * 60,
        ]
        for idx, row in self.leaderboard.iterrows():
            lines.append(
                f"  {idx + 1:2d}. {row['model']:<22} | CV Score: {row['score']:.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class AutoTrainer:
    """Zero-Dependency Automated Model Selection and Tuning Engine."""

    def __init__(
        self,
        task: str = "auto",
        time_budget_secs: int = 60,
        candidate_models: Optional[List[str]] = None,
        tune: bool = True,
        cv: int = 3,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.task = task
        self.time_budget = time_budget_secs
        self.candidate_models = candidate_models
        self.tune = tune
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _infer_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            return self.task
        if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 20:
            return "regression"
        return "classification"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> AutoMLResult:
        """Benchmark candidate models and select/tune the champion."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)

        task = self._infer_task(y_arr)
        if self.verbose:
            Logger.info(f"AutoTrainer starting. Inferred Task: {task.upper()}")

        if self.candidate_models is None:
            if task == "classification":
                candidates = [
                    "logistic_regression",
                    "random_forest",
                    "gradient_boosting",
                    "knn",
                    "decision_tree",
                ]
            else:
                candidates = [
                    "linear_regression",
                    "ridge",
                    "lasso",
                    "random_forest",
                    "gradient_boosting",
                ]
        else:
            candidates = self.candidate_models

        # Split for cross-validation
        n = len(X_arr)
        np.random.seed(self.random_state)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = max(1, n // self.cv)

        leaderboard_rows: List[Dict[str, Any]] = []
        trained_models: Dict[str, Tuple[ChokkhuModel, float]] = {}

        for model_name in candidates:
            if self.verbose:
                Logger.info(f"Evaluating candidate: {model_name}...")

            scores = []
            for f in range(self.cv):
                val_idx = indices[f * fold_size : (f + 1) * fold_size]
                tr_idx = np.setdiff1d(indices, val_idx)

                try:
                    m = train(
                        model=model_name,
                        X_train=X_arr[tr_idx],
                        y_train=y_arr[tr_idx],
                        task=task,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    preds = m.predict(X_arr[val_idx])

                    if task == "classification":
                        score = float(np.mean(y_arr[val_idx] == preds))
                    else:
                        ss_res = float(np.sum((y_arr[val_idx] - preds) ** 2))
                        ss_tot = float(
                            np.sum((y_arr[val_idx] - np.mean(y_arr[val_idx])) ** 2)
                        )
                        score = 1.0 - (ss_res / max(1e-8, ss_tot))
                    scores.append(score)
                except Exception as e:
                    if self.verbose:
                        Logger.warning(f"Model {model_name} fold {f} failed: {e}")
                    scores.append(float("-inf"))

            mean_score = float(np.mean(scores))
            # Fit final champion on full dataset
            try:
                final_m = train(
                    model=model_name,
                    X_train=X_arr,
                    y_train=y_arr,
                    task=task,
                    random_state=self.random_state,
                    verbose=False,
                )
                trained_models[model_name] = (final_m, mean_score)
                leaderboard_rows.append({"model": model_name, "score": mean_score})
            except Exception:
                pass

        if not leaderboard_rows:
            raise RuntimeError(
                "All candidate models failed during AutoTrainer execution."
            )

        leaderboard_df = (
            pd.DataFrame(leaderboard_rows)
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )
        winning_name = str(leaderboard_df.iloc[0]["model"])
        winning_score = float(leaderboard_df.iloc[0]["score"])
        winning_model = trained_models[winning_name][0]

        if self.verbose:
            Logger.info(
                f"AutoTrainer finished. Champion: {winning_name} (CV Score: {winning_score:.4f})"
            )

        return AutoMLResult(
            best_model=winning_model,
            best_model_name=winning_name,
            best_score=winning_score,
            leaderboard=leaderboard_df,
            task=task,
        )


def auto_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    task: str = "auto",
    time_budget_secs: int = 60,
    time_budget: Optional[int] = None,
    candidate_models: Optional[List[str]] = None,
    tune: bool = True,
    cv: int = 3,
    verbose: bool = True,
    **kwargs: Any,
) -> AutoMLResult:
    """Convenience top-level API for sovereign AutoML."""
    budget = time_budget if time_budget is not None else time_budget_secs
    trainer = AutoTrainer(
        task=task,
        time_budget_secs=budget,
        candidate_models=candidate_models,
        tune=tune,
        cv=cv,
        verbose=verbose,
    )
    return trainer.fit(X, y)
