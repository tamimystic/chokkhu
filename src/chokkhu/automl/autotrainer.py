"""Sovereign AutoTrainer: Zero-Dependency End-to-End Automated Machine Learning Engine."""

from __future__ import annotations

import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from chokkhu.cleaning import clean as clean_data
from chokkhu.core.logger import Logger
from chokkhu.models.base import ChokkhuModel
from chokkhu.models.engine import train
from chokkhu.preprocessing import PreprocessorState
from chokkhu.preprocessing import preprocess as preprocess_data


class AutoMLResult:
    """Encapsulates AutoML run results, leaderboard, fitted pipeline, and champion model."""

    def __init__(
        self,
        best_model: ChokkhuModel,
        best_model_name: str,
        best_score: float,
        leaderboard: pd.DataFrame,
        task: str,
        preprocessor: Optional[PreprocessorState] = None,
        best_params: Optional[Dict[str, Any]] = None,
        trained_models: Optional[Dict[str, ChokkhuModel]] = None,
        impute_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_score = best_score
        self.leaderboard = leaderboard
        self.task = task
        self.preprocessor = preprocessor
        self.best_params = best_params or {}
        self.trained_models = trained_models or {}
        self.impute_values = impute_values or {}

    def _transform_input(self, X: Any) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            df = X.to_frame() if isinstance(X, pd.Series) else X.copy()
            if self.impute_values:
                for col, val in self.impute_values.items():
                    if col in df.columns:
                        df[col] = df[col].fillna(val)
            if self.preprocessor is not None:
                df_proc = self.preprocessor.transform(df)
                return df_proc.to_numpy(dtype=np.float64)
            return df.to_numpy(dtype=np.float64)

        X_arr = np.asarray(X)
        if self.preprocessor is not None and getattr(
            self.preprocessor, "feature_names_out", None
        ):
            cols = (
                [f"f_{i}" for i in range(X_arr.shape[1])] if X_arr.ndim > 1 else ["f_0"]
            )
            df = pd.DataFrame(X_arr, columns=cols)
            if self.impute_values:
                for col, val in self.impute_values.items():
                    if col in df.columns:
                        df[col] = df[col].fillna(val)
            df_proc = self.preprocessor.transform(df)
            return df_proc.to_numpy(dtype=np.float64)

        if pd.api.types.is_numeric_dtype(X_arr) and np.isnan(X_arr).any():
            col_meds = (
                np.nanmedian(X_arr, axis=0) if X_arr.ndim > 1 else np.nanmedian(X_arr)
            )
            X_arr = np.nan_to_num(X_arr, nan=col_meds)
        return np.asarray(X_arr, dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        X_proc = self._transform_input(X)
        return self.best_model.predict(X_proc)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )
        X_proc = self._transform_input(X)
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X_proc)
        preds = self.best_model.predict(X_proc)
        classes = getattr(self.best_model, "classes_", np.unique(preds))
        n_classes = len(classes)
        probs: np.ndarray = np.zeros((len(preds), n_classes), dtype=np.float64)
        for i, c in enumerate(classes):
            probs[preds == c, i] = 1.0
        return probs

    def score(self, X: Any, y: Any) -> float:
        X_proc = self._transform_input(X)
        preds = self.best_model.predict(X_proc)
        y_arr = np.asarray(y)
        if self.task == "classification":
            return float(np.mean(preds == y_arr))
        else:
            y_f = np.asarray(y_arr, dtype=np.float64)
            ss_res = float(np.sum((y_f - preds) ** 2))
            ss_tot = float(np.sum((y_f - np.mean(y_f)) ** 2))
            return float(1.0 - (ss_res / max(1e-8, ss_tot)))

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
            std_str = (
                f" \u00b1 {row['std']:.4f}"
                if "std" in row and not pd.isna(row["std"])
                else ""
            )
            lines.append(
                f"  {idx + 1:2d}. {row['model']:<22} | CV Score: {row['score']:.4f}{std_str}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> AutoMLResult:
        with open(filepath, "rb") as f:
            return pickle.load(f)


class _VotingEnsemble(ChokkhuModel):
    """Simple weighted voting / averaging ensemble of top candidate models."""

    def __init__(self, models: List[Tuple[str, ChokkhuModel]], task: str) -> None:
        self.models = models
        self.task = task
        self.classes_ = getattr(models[0][1], "classes_", None) if models else None

    def fit(self, X: Any, y: Any = None) -> _VotingEnsemble:
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.task == "classification":
            if all(hasattr(m, "predict_proba") for _, m in self.models):
                avg_prob = np.mean([m.predict_proba(X) for _, m in self.models], axis=0)
                if self.classes_ is not None:
                    return self.classes_[np.argmax(avg_prob, axis=1)]
                return np.argmax(avg_prob, axis=1)
            all_preds = np.array([m.predict(X) for _, m in self.models])
            from collections import Counter

            return np.array(
                [
                    Counter(all_preds[:, i]).most_common(1)[0][0]
                    for i in range(all_preds.shape[1])
                ]
            )
        else:
            return np.mean([m.predict(X) for _, m in self.models], axis=0)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        probs = [
            m.predict_proba(X) if hasattr(m, "predict_proba") else None
            for _, m in self.models
        ]
        valid_probs = [p for p in probs if p is not None]
        if valid_probs:
            return np.mean(valid_probs, axis=0)
        preds = self.predict(X)
        classes = self.classes_ if self.classes_ is not None else np.unique(preds)
        out: np.ndarray = np.zeros((len(preds), len(classes)), dtype=np.float64)
        for i, c in enumerate(classes):
            out[preds == c, i] = 1.0
        return out


class AutoTrainer:
    """Zero-Dependency Automated Model Selection, Preprocessing, and Hyperparameter Tuning Engine."""

    def __init__(
        self,
        task: str = "auto",
        time_budget_secs: int = 60,
        time_budget: Optional[int] = None,
        candidate_models: Optional[List[str]] = None,
        tune: bool = False,
        cv: int = 3,
        scoring: Optional[str] = None,
        clean: bool = True,
        preprocess: bool = True,
        scale: Optional[str] = "standard",
        encode: Optional[str] = "onehot",
        missing_strategy: str = "median",
        outliers: Optional[str] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        ensemble: bool = False,
        random_state: int = 42,
        verbose: bool = True,
        target: Optional[str] = None,
        tuner: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.time_budget = time_budget if time_budget is not None else time_budget_secs
        self.candidate_models = candidate_models
        self.tune = tune
        self.cv = cv
        self.scoring = scoring
        self.clean = clean
        self.preprocess = preprocess
        self.scale = scale
        self.encode = encode
        self.missing_strategy = missing_strategy
        self.outliers = outliers
        self.model_params = model_params or {}
        self.ensemble = ensemble
        self.random_state = random_state
        self.verbose = verbose
        self.target = target
        self.tuner = tuner
        self.extra_kwargs = kwargs
        self.preprocessor_: Optional[PreprocessorState] = None
        self.impute_values_: Dict[str, Any] = {}

    def _infer_task(self, y: Any) -> str:
        if self.task != "auto":
            return self.task
        if pd.api.types.is_float_dtype(y) and len(np.unique(y)) > 20:
            return "regression"
        return "classification"

    def _safe_train(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        m_kwargs: Dict[str, Any],
    ) -> ChokkhuModel:
        curr_kwargs = dict(m_kwargs)
        for _ in range(len(m_kwargs) + 1):
            try:
                return train(
                    model=model_name,
                    X_train=X,
                    y_train=y,
                    task=task,
                    random_state=self.random_state,
                    tune=self.tune,
                    verbose=False,
                    **curr_kwargs,
                )
            except TypeError as e:
                err_msg = str(e)
                if "unexpected keyword argument" in err_msg:
                    match = re.search(r"unexpected keyword argument '([^']+)'", err_msg)
                    if match:
                        bad_key = match.group(1)
                        if bad_key in curr_kwargs:
                            del curr_kwargs[bad_key]
                            continue
                break
        return train(
            model=model_name,
            X_train=X,
            y_train=y,
            task=task,
            random_state=self.random_state,
            tune=self.tune,
            verbose=False,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, str],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> AutoMLResult:
        """Benchmark candidate models, tune parameters, and construct champion pipeline."""
        start_time = time.time()

        if isinstance(X, str):
            from chokkhu.io.loader import load

            X = load(X)

        t_col = target or self.target or fit_kwargs.get("target")
        if isinstance(X, pd.DataFrame) and y is None:
            if t_col is not None and t_col in X.columns:
                y = X[t_col]
                X = X.drop(columns=[t_col])
            else:
                raise ValueError(
                    "When passing a single DataFrame without y, specify target='column_name'."
                )

        if y is None:
            raise ValueError("Target y cannot be None for AutoTrainer.")

        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        task = self._infer_task(y_arr)
        if self.verbose:
            Logger.info(f"AutoTrainer starting. Inferred Task: {task.upper()}")

        # Data Cleaning and Preprocessing Pipeline
        if self.clean or self.preprocess:
            if isinstance(X, pd.DataFrame):
                df_X = X.copy()
            else:
                df_X = pd.DataFrame(
                    X, columns=[f"f_{i}" for i in range(np.asarray(X).shape[1])]
                )

            # Track imputation values for future raw inference
            self.impute_values_ = {}
            for col in df_X.columns:
                if pd.api.types.is_numeric_dtype(df_X[col]):
                    med = (
                        float(df_X[col].median()) if not df_X[col].isna().all() else 0.0
                    )
                    self.impute_values_[col] = med
                else:
                    mode_val = df_X[col].mode()
                    self.impute_values_[col] = (
                        mode_val.iloc[0] if len(mode_val) > 0 else ""
                    )

            if self.clean:
                df_X = clean_data(
                    df_X,
                    missing=self.missing_strategy,
                    duplicates=False,
                    outliers=self.outliers,
                    verbose=False,
                )

            if self.preprocess:
                df_proc, prep_state = preprocess_data(
                    df_X,
                    scale=self.scale,
                    encode=self.encode,
                    verbose=False,
                )
                self.preprocessor_ = prep_state
                X_arr = df_proc.to_numpy(dtype=np.float64)
            else:
                X_arr = df_X.to_numpy(dtype=np.float64)
        else:
            if isinstance(X, pd.DataFrame):
                X_arr = X.to_numpy(dtype=np.float64)
            else:
                X_arr = np.asarray(X, dtype=np.float64)

        if self.candidate_models is None:
            if task == "classification":
                candidates = [
                    "random_forest",
                    "logistic_regression",
                    "decision_tree",
                    "knn",
                    "gradient_boosting",
                    "neural_network",
                ]
            else:
                candidates = [
                    "random_forest",
                    "linear_regression",
                    "ridge",
                    "decision_tree",
                    "knn",
                    "gradient_boosting",
                    "neural_network",
                ]
        else:
            candidates = self.candidate_models

        n = len(X_arr)
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)
        fold_size = max(1, n // max(1, self.cv))

        leaderboard_rows: List[Dict[str, Any]] = []
        trained_models: Dict[str, ChokkhuModel] = {}
        model_scores: Dict[str, float] = {}

        automl_reserved = {
            "target",
            "clean",
            "preprocess",
            "scale",
            "encode",
            "missing",
            "missing_strategy",
            "outliers",
            "tune",
            "cv",
            "scoring",
            "time_budget",
            "time_budget_secs",
            "candidate_models",
            "ensemble",
            "tuner",
            "random_state",
            "verbose",
            "budget",
            "time_limit",
            "n_jobs",
        }
        all_extra = dict(self.extra_kwargs)
        all_extra.update(fit_kwargs)
        base_kwargs = {k: v for k, v in all_extra.items() if k not in automl_reserved}

        for model_name in candidates:
            if time.time() - start_time > self.time_budget:
                if self.verbose:
                    Logger.info(
                        f"Time budget ({self.time_budget}s) reached. Finalizing evaluated models."
                    )
                break

            if self.verbose:
                Logger.info(f"Evaluating candidate: {model_name}...")

            scores: List[float] = []
            m_kwargs = dict(base_kwargs)
            if model_name in self.model_params:
                m_kwargs.update(self.model_params[model_name])

            for f in range(self.cv):
                val_idx = indices[f * fold_size : (f + 1) * fold_size]
                tr_idx = np.setdiff1d(indices, val_idx)

                try:
                    m = self._safe_train(
                        model_name=model_name,
                        X=X_arr[tr_idx],
                        y=y_arr[tr_idx],
                        task=task,
                        m_kwargs=m_kwargs,
                    )
                    preds = m.predict(X_arr[val_idx])

                    if task == "classification":
                        score = float(np.mean(y_arr[val_idx] == preds))
                    else:
                        ss_res = float(np.sum((y_arr[val_idx] - preds) ** 2))
                        ss_tot = float(
                            np.sum((y_arr[val_idx] - np.mean(y_arr[val_idx])) ** 2)
                        )
                        score = float(1.0 - (ss_res / max(1e-8, ss_tot)))
                    scores.append(score)
                except Exception as e:
                    if self.verbose:
                        Logger.warning(f"Model {model_name} fold {f} failed: {e}")
                    scores.append(float("-inf"))

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            # Train final champion model on full dataset
            try:
                final_m = self._safe_train(
                    model_name=model_name,
                    X=X_arr,
                    y=y_arr,
                    task=task,
                    m_kwargs=m_kwargs,
                )
                trained_models[model_name] = final_m
                model_scores[model_name] = mean_score
                leaderboard_rows.append(
                    {
                        "model": model_name,
                        "score": mean_score,
                        "std": std_score,
                    }
                )
            except Exception as e:
                if self.verbose:
                    Logger.warning(f"Final training of {model_name} failed: {e}")

        if not leaderboard_rows:
            raise RuntimeError(
                "All candidate models failed during AutoTrainer execution."
            )

        leaderboard_df = (
            pd.DataFrame(leaderboard_rows)
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )

        if self.ensemble and len(leaderboard_df) >= 2:
            top_names = leaderboard_df["model"].head(3).tolist()
            top_models = [(name, trained_models[name]) for name in top_names]
            ensemble_model = _VotingEnsemble(top_models, task=task)
            winning_name = "ensemble"
            winning_model: ChokkhuModel = ensemble_model
            winning_score = float(leaderboard_df.iloc[0]["score"])
        else:
            winning_name = str(leaderboard_df.iloc[0]["model"])
            winning_score = float(leaderboard_df.iloc[0]["score"])
            winning_model = trained_models[winning_name]

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
            preprocessor=self.preprocessor_,
            best_params=getattr(winning_model, "best_params_", {}),
            trained_models=trained_models,
            impute_values=self.impute_values_,
        )


def auto_train(
    X: Union[np.ndarray, pd.DataFrame, str],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    task: str = "auto",
    time_budget_secs: int = 60,
    time_budget: Optional[int] = None,
    candidate_models: Optional[List[str]] = None,
    tune: bool = False,
    cv: int = 3,
    scoring: Optional[str] = None,
    clean: bool = True,
    preprocess: bool = True,
    scale: Optional[str] = "standard",
    encode: Optional[str] = "onehot",
    missing_strategy: str = "median",
    outliers: Optional[str] = None,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ensemble: bool = False,
    random_state: int = 42,
    verbose: bool = True,
    target: Optional[str] = None,
    tuner: Optional[str] = None,
    **kwargs: Any,
) -> AutoMLResult:
    """Convenience top-level API for sovereign single-line AutoML."""
    budget = time_budget if time_budget is not None else time_budget_secs
    trainer = AutoTrainer(
        task=task,
        time_budget_secs=budget,
        candidate_models=candidate_models,
        tune=tune,
        cv=cv,
        scoring=scoring,
        clean=clean,
        preprocess=preprocess,
        scale=scale,
        encode=encode,
        missing_strategy=missing_strategy,
        outliers=outliers,
        model_params=model_params,
        ensemble=ensemble,
        random_state=random_state,
        verbose=verbose,
        target=target,
        tuner=tuner,
        **kwargs,
    )
    return trainer.fit(X, y, target=target, **kwargs)


automl = auto_train
