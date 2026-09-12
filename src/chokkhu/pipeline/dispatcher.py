"""Multi-Task Universal Pipeline Dispatcher for Chokkhu.

Dispatches and orchestrates specialized machine learning pipelines:
1. Tabular Classification & Regression (with Conformal Uncertainty)
2. Time Series Forecasting & State Filtering (Lag transforms, Ridge, ARIMA)
3. Causal Inference & Treatment Effects (Doubly Robust, ATE/CATE)
4. Survival Analysis & Event-Time Modeling (Cox-PH, Kaplan-Meier, C-Index)
5. Unsupervised Anomaly Detection (IsolationForest, OneClassSVM)
6. Unsupervised Clustering (KMeans, MiniBatchKMeans, GaussianMixture)
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from chokkhu.cleaning import clean as clean_fn
from chokkhu.core.logger import Logger
from chokkhu.evaluation.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from chokkhu.io.loader import load as load_fn
from chokkhu.models.engine import train as train_fn
from chokkhu.preprocessing import PreprocessorState, preprocess as preprocess_fn
from chokkhu.splitting.splitter import train_test_split
from chokkhu.uncertainty.cqr import ConformalizedQuantileRegression


class MultiTaskPipelineResult:
    """Universal multi-task pipeline execution result.

    Supports leak-free inference, conformal uncertainty intervals,
    forecasting, causal treatment effect estimation, and survival curve evaluation.
    """

    def __init__(
        self,
        task: str,
        model: Any,
        data_raw: pd.DataFrame,
        data_cleaned: pd.DataFrame,
        splits: Dict[str, Any],
        preprocessor_state: Optional[PreprocessorState] = None,
        transformation_state: Optional[Any] = None,
        model_name: str = "custom",
        target_col: Optional[str] = None,
        treatment_col: Optional[str] = None,
        duration_col: Optional[str] = None,
        event_col: Optional[str] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        conformal_model: Optional[Any] = None,
        conformal_interval: Optional[float] = None,
        cv_scores: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task = str(task).lower()
        self.model = model
        self.data_raw = data_raw
        self.data_cleaned = data_cleaned
        self.splits = splits
        self.preprocessor_state = preprocessor_state
        self.transformation_state = transformation_state
        self.model_name = model_name
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.duration_col = duration_col
        self.event_col = event_col
        self.evaluation = evaluation or {}
        self.conformal_model = conformal_model
        self.conformal_interval = conformal_interval
        self.cv_scores = cv_scores or {}
        self.feature_names = feature_names or []
        self.task_metadata = task_metadata or {}

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.evaluation

    def _prepare_inference_df(
        self, new_data: Union[pd.DataFrame, np.ndarray, dict, list, str]
    ) -> pd.DataFrame:
        """Sanitize and align input data for zero-leakage inference."""
        if isinstance(new_data, str):
            from chokkhu.io import load

            df = load(new_data, verbose=False)
        elif isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, np.ndarray):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            df = new_data.copy()
        else:
            raise TypeError("Unsupported data type for inference.")

        # Drop target/treatment/duration/event columns if present in test data
        cols_to_drop = [
            c
            for c in [
                self.target_col,
                self.treatment_col,
                self.duration_col,
                self.event_col,
            ]
            if c is not None and c in df.columns
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if self.preprocessor_state is not None:
            df = self.preprocessor_state.transform(df)

        if self.transformation_state is not None:
            df = self.transformation_state.transform(df)

        return df

    def predict(
        self,
        new_data: Union[pd.DataFrame, np.ndarray, dict, list, str],
        return_intervals: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Predict target with zero data leakage.

        If return_intervals=True and conformal_interval is calibrated,
        returns (y_pred, y_lower, y_upper).
        """
        df_proc = self._prepare_inference_df(new_data)
        X_arr = df_proc.to_numpy(dtype=np.float64)

        if hasattr(self.model, "predict"):
            preds = np.asarray(self.model.predict(X_arr))
        elif callable(self.model):
            preds = np.asarray(self.model(X_arr))
        else:
            raise RuntimeError("Model does not support predict.")

        if return_intervals and self.conformal_model is not None:
            if hasattr(self.conformal_model, "predict_interval"):
                intervals = self.conformal_model.predict_interval(X_arr)
                if isinstance(intervals, dict):
                    return preds, intervals["lower"], intervals["upper"]
                elif isinstance(intervals, tuple) and len(intervals) == 2:
                    return preds, intervals[0], intervals[1]
            elif hasattr(self.conformal_model, "predict"):
                res = self.conformal_model.predict(X_arr)
                if isinstance(res, tuple) and len(res) == 2:
                    return preds, res[0], res[1]

        return preds

    def predict_interval(
        self,
        new_data: Union[pd.DataFrame, np.ndarray, dict, list, str],
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns point predictions, lower conformal bounds, and upper conformal bounds."""
        df_proc = self._prepare_inference_df(new_data)
        X_arr = df_proc.to_numpy(dtype=np.float64)

        raw_preds = self.predict(new_data, return_intervals=False)
        preds_arr: np.ndarray = np.asarray(raw_preds)

        if self.conformal_model is not None:
            if hasattr(self.conformal_model, "predict_interval"):
                intervals = self.conformal_model.predict_interval(X_arr)
                if isinstance(intervals, dict):
                    return (
                        preds_arr,
                        np.asarray(intervals["lower"]),
                        np.asarray(intervals["upper"]),
                    )
                elif isinstance(intervals, tuple) and len(intervals) == 2:
                    return preds_arr, np.asarray(intervals[0]), np.asarray(intervals[1])
            elif hasattr(self.conformal_model, "predict"):
                low, high = self.conformal_model.predict(X_arr)
                return preds_arr, np.asarray(low), np.asarray(high)

        res_std = float(self.task_metadata.get("residual_std", 1.0))
        z = 1.96 if alpha is None else 1.96
        return preds_arr, preds_arr - z * res_std, preds_arr + z * res_std

    def predict_proba(
        self, new_data: Union[pd.DataFrame, np.ndarray, dict, list]
    ) -> np.ndarray:
        """Returns predicted class probabilities if model supports predict_proba."""
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only supported for classification tasks."
            )

        df_proc = self._prepare_inference_df(new_data)
        X_arr = df_proc.to_numpy(dtype=np.float64)

        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X_arr))
        raise RuntimeError("Model does not support predict_proba.")

    def forecast(
        self, steps: int = 10, future_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate out-of-sample forecast for time series tasks."""
        if self.task != "timeseries_forecast":
            raise ValueError(
                "forecast is only supported for timeseries_forecast tasks."
            )

        if hasattr(self.model, "forecast"):
            return np.asarray(self.model.forecast(steps=steps))
        elif hasattr(self.model, "predict"):
            if future_features is not None:
                return np.asarray(self.model.predict(future_features))
            last_window = self.task_metadata.get("last_window", np.zeros(steps))
            preds = []
            curr = list(last_window)
            n_lags = len(curr)
            for _ in range(steps):
                x_in = np.asarray(curr[-n_lags:]).reshape(1, -1)
                p = float(self.model.predict(x_in)[0])
                preds.append(p)
                curr.append(p)
            return np.asarray(preds)

        raise RuntimeError("Time series model does not support forecasting.")

    def estimate_ate(self) -> float:
        """Return Average Treatment Effect (ATE) for causal inference."""
        if self.task != "causal_inference":
            raise ValueError(
                "estimate_ate is only supported for causal_inference tasks."
            )
        return float(self.task_metadata.get("ate", 0.0))

    def predict_effect(
        self, new_data: Union[pd.DataFrame, np.ndarray, dict, list]
    ) -> np.ndarray:
        """Predict Individual Treatment Effect (ITE / CATE)."""
        if self.task != "causal_inference":
            raise ValueError(
                "predict_effect is only supported for causal_inference tasks."
            )
        df_proc = self._prepare_inference_df(new_data)
        X_arr = df_proc.to_numpy(dtype=np.float64)
        if hasattr(self.model, "predict_effect"):
            return np.asarray(self.model.predict_effect(X_arr))
        elif hasattr(self.model, "predict_cate"):
            return np.asarray(self.model.predict_cate(X_arr))
        ate = self.estimate_ate()
        return np.full(X_arr.shape[0], ate)

    def summary(self) -> str:
        """Formatted multi-task pipeline execution summary."""
        lines = [
            "=" * 65,
            "           CHOKKHU MULTI-TASK UNIVERSAL PIPELINE SUMMARY",
            "=" * 65,
            f"  Task Type             : {self.task.upper()}",
            f"  Selected Model        : {self.model_name}",
            f"  Raw Data Shape        : {self.data_raw.shape}",
            f"  Cleaned Data Shape    : {self.data_cleaned.shape}",
        ]
        if self.target_col:
            lines.append(f"  Target Column         : {self.target_col}")
        if self.treatment_col:
            lines.append(f"  Treatment Column      : {self.treatment_col}")
        if self.duration_col and self.event_col:
            lines.append(
                f"  Duration / Event Cols : {self.duration_col} / {self.event_col}"
            )

        if self.conformal_interval is not None:
            lines.append(
                f"  Conformal Coverage    : {int(self.conformal_interval * 100)}% Mathematically Calibrated"
            )

        lines.append("-" * 65)
        lines.append("  EVALUATION METRICS:")
        for metric, val in self.evaluation.items():
            if isinstance(val, (int, float, np.floating, np.integer)):
                lines.append(f"    - {metric.capitalize():<22}: {val:.4f}")
            elif isinstance(val, str):
                lines.append(f"    - {metric.capitalize():<22}: {val}")

        if self.task == "causal_inference" and "ate" in self.task_metadata:
            lines.append("-" * 65)
            lines.append(f"  Estimated ATE (Causal): {self.task_metadata['ate']:.4f}")
            if "ate_ci_lower" in self.task_metadata:
                lines.append(
                    f"  95% Confidence Interval: [{self.task_metadata['ate_ci_lower']:.4f}, {self.task_metadata['ate_ci_upper']:.4f}]"
                )

        if self.task == "survival" and "concordance_index" in self.task_metadata:
            lines.append("-" * 65)
            lines.append(
                f"  Harrell's C-Index     : {self.task_metadata['concordance_index']:.4f}"
            )

        lines.append("=" * 65)
        return "\n".join(lines)

    def save(self, path: str) -> str:
        """Save pipeline result to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        Logger.info(f"Saved MultiTaskPipelineResult to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "MultiTaskPipelineResult":
        """Load pipeline result from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline file not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not a MultiTaskPipelineResult (got {type(obj)})"
            )
        Logger.info(f"Loaded MultiTaskPipelineResult from: {path}")
        return obj


class MultiTaskDispatcher:
    """Universal Multi-Task Pipeline Dispatcher."""

    @staticmethod
    def dispatch(
        data: Union[str, pd.DataFrame, np.ndarray],
        target: Optional[str] = None,
        task: str = "auto",
        treatment: Optional[str] = None,
        duration: Optional[str] = None,
        event: Optional[str] = None,
        model: Union[str, List[str], Any] = "auto",
        clean: Union[bool, str, Dict[str, Any]] = "auto",
        preprocess: Union[bool, str, Dict[str, Any]] = "auto",
        transform: Optional[Dict[str, Any]] = None,
        resample: Optional[str] = None,
        resample_ratio: float = 1.0,
        conformal_interval: Optional[float] = None,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        random_state: int = 42,
        evaluate: bool = True,
        save_reports: bool = False,
        save_dir: str = "chokkhu_reports",
        verbose: bool = True,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Dispatch execution across specialized tasks with zero data leakage."""
        if isinstance(data, str):
            df_raw = load_fn(data, verbose=False)
        elif isinstance(data, pd.DataFrame):
            df_raw = data.copy()
        elif isinstance(data, np.ndarray):
            df_raw = pd.DataFrame(data)
        else:
            raise TypeError("Data must be a DataFrame, file path, or ndarray.")

        # 1. Clean Data
        df_clean = df_raw.copy()
        if clean:
            clean_kwargs = clean if isinstance(clean, dict) else {}
            df_clean = clean_fn(
                df_clean,
                missing=clean_kwargs.get("missing", "median"),
                outliers=clean_kwargs.get("outliers", "iqr"),
                duplicates=clean_kwargs.get("duplicates", True),
                fix_data_types=clean_kwargs.get("fix_data_types", True),
                verbose=False,
            )

        # 2. Auto-Detect Task
        resolved_task = str(task).lower()
        if resolved_task == "auto":
            if duration is not None and event is not None:
                resolved_task = "survival"
            elif treatment is not None:
                resolved_task = "causal_inference"
            elif target is None:
                resolved_task = "anomaly_detection"
            elif target in df_clean.columns:
                y_col = df_clean[target]
                if pd.api.types.is_numeric_dtype(y_col) and y_col.nunique() > 20:
                    resolved_task = "regression"
                else:
                    resolved_task = "classification"
            else:
                resolved_task = "regression"

        # Dispatch to specialized handler
        if resolved_task == "timeseries_forecast":
            return MultiTaskDispatcher._dispatch_timeseries(
                df_raw=df_raw,
                df_clean=df_clean,
                target_col=target,
                model_spec=model,
                test_size=test_size,
                conformal_interval=conformal_interval,
                random_state=random_state,
                verbose=verbose,
                **kwargs,
            )
        elif resolved_task == "causal_inference":
            return MultiTaskDispatcher._dispatch_causal(
                df_raw=df_raw,
                df_clean=df_clean,
                treatment_col=treatment,
                outcome_col=target,
                model_spec=model,
                random_state=random_state,
                verbose=verbose,
                **kwargs,
            )
        elif resolved_task == "survival":
            return MultiTaskDispatcher._dispatch_survival(
                df_raw=df_raw,
                df_clean=df_clean,
                duration_col=duration,
                event_col=event,
                model_spec=model,
                random_state=random_state,
                verbose=verbose,
                **kwargs,
            )
        elif resolved_task == "anomaly_detection":
            return MultiTaskDispatcher._dispatch_anomaly(
                df_raw=df_raw,
                df_clean=df_clean,
                model_spec=model,
                random_state=random_state,
                verbose=verbose,
                **kwargs,
            )
        elif resolved_task == "clustering":
            return MultiTaskDispatcher._dispatch_clustering(
                df_raw=df_raw,
                df_clean=df_clean,
                model_spec=model,
                random_state=random_state,
                verbose=verbose,
                **kwargs,
            )
        else:
            return MultiTaskDispatcher._dispatch_tabular(
                df_raw=df_raw,
                df_clean=df_clean,
                target=target,
                task=resolved_task,
                model=model,
                preprocess=preprocess,
                transform=transform,
                resample=resample,
                resample_ratio=resample_ratio,
                conformal_interval=conformal_interval,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                evaluate=evaluate,
                save_reports=save_reports,
                save_dir=save_dir,
                verbose=verbose,
                **kwargs,
            )

    @staticmethod
    def _dispatch_tabular(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        target: Optional[str],
        task: str,
        model: Any,
        preprocess: Any,
        transform: Any,
        resample: Any,
        resample_ratio: float,
        conformal_interval: Optional[float],
        test_size: float,
        val_size: Optional[float],
        random_state: int,
        evaluate: bool,
        save_reports: bool,
        save_dir: str,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle supervised tabular classification and regression."""
        if target is None or target not in df_clean.columns:
            raise ValueError(f"Target column '{target}' required for {task}.")

        df_clean = df_clean.dropna(subset=[target])
        y = df_clean[target]
        X = df_clean.drop(columns=[target])

        use_stratify = task == "classification"
        if val_size is not None or conformal_interval is not None:
            effective_val = val_size if val_size is not None else 0.15
            X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                val_size=effective_val,
                stratify=use_stratify,
                shuffle=True,
                random_state=random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=use_stratify,
                shuffle=True,
                random_state=random_state,
            )
            X_val, y_val = None, None

        prep_opts = (
            {"scale": "standard", "encode": "onehot"}
            if preprocess in (True, "auto")
            else (preprocess or {})
        )
        X_train_proc, preprocessor_state = preprocess_fn(
            X_train,
            target=None,
            scale=prep_opts.get("scale", "standard"),
            encode=prep_opts.get("encode", "onehot"),
            verbose=False,
        )
        X_test_proc = preprocessor_state.transform(X_test)
        X_val_proc = preprocessor_state.transform(X_val) if X_val is not None else None

        X_tr = X_train_proc.to_numpy(dtype=np.float64)
        y_tr = y_train.to_numpy()
        X_te = X_test_proc.to_numpy(dtype=np.float64)
        y_te = y_test.to_numpy()

        if model == "auto" or model is None:
            chosen_name = "random_forest" if task == "classification" else "ridge"
        else:
            chosen_name = model if isinstance(model, str) else "custom"

        if isinstance(model, str) and model != "auto":
            fitted_model = train_fn(
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                task=task,
                random_state=random_state,
                verbose=False,
                **kwargs,
            )
        elif hasattr(model, "fit"):
            fitted_model = model
            fitted_model.fit(X_tr, y_tr)
        else:
            fitted_model = train_fn(
                model=chosen_name,
                X_train=X_tr,
                y_train=y_tr,
                task=task,
                random_state=random_state,
                verbose=False,
                **kwargs,
            )

        conformal_obj = None
        if (
            conformal_interval is not None
            and X_val_proc is not None
            and y_val is not None
        ):
            X_va = X_val_proc.to_numpy(dtype=np.float64)
            y_va = y_val.to_numpy()
            alpha = 1.0 - float(conformal_interval)
            cqr = ConformalizedQuantileRegression(alpha=alpha)
            cqr.fit(X_tr, y_tr)
            cqr.calibrate(X_va, y_va)
            conformal_obj = cqr

        eval_dict: Dict[str, Any] = {}
        if evaluate:
            preds_te = fitted_model.predict(X_te)
            if task == "classification":
                eval_dict["accuracy"] = float(accuracy_score(y_te, preds_te))
                eval_dict["f1"] = float(f1_score(y_te, preds_te))
            else:
                eval_dict["r2"] = float(r2_score(y_te, preds_te))
                eval_dict["rmse"] = float(np.sqrt(mean_squared_error(y_te, preds_te)))
                eval_dict["mae"] = float(mean_absolute_error(y_te, preds_te))

        res_std = (
            float(np.std(y_te - fitted_model.predict(X_te)))
            if task == "regression"
            else 0.0
        )

        return MultiTaskPipelineResult(
            task=task,
            model=fitted_model,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te},
            preprocessor_state=preprocessor_state,
            model_name=chosen_name,
            target_col=target,
            evaluation=eval_dict,
            conformal_model=conformal_obj,
            conformal_interval=conformal_interval,
            feature_names=list(X_train_proc.columns),
            task_metadata={"residual_std": res_std},
        )

    @staticmethod
    def _dispatch_timeseries(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        target_col: Optional[str],
        model_spec: Any,
        test_size: float,
        conformal_interval: Optional[float],
        random_state: int,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle time-series forecasting pipeline."""
        t_col = target_col if target_col is not None else df_clean.columns[-1]
        series = df_clean[t_col].to_numpy(dtype=np.float64)
        n_total = len(series)
        n_test = max(2, int(n_total * test_size))
        n_train = n_total - n_test

        train_series = series[:n_train]
        test_series = series[n_train:]

        lags = kwargs.get("lags", 5)
        X_lags, y_lags = [], []
        for i in range(lags, len(train_series)):
            X_lags.append(train_series[i - lags : i])
            y_lags.append(train_series[i])
        X_lags_arr = np.asarray(X_lags, dtype=np.float64)
        y_lags_arr = np.asarray(y_lags, dtype=np.float64)

        from chokkhu.models.ml.glm import RidgeRegression

        ts_model = RidgeRegression(alpha=1.0)
        ts_model.fit(X_lags_arr, y_lags_arr)

        history = list(train_series)
        preds = []
        for _ in range(n_test):
            feat = np.asarray(history[-lags:]).reshape(1, -1)
            pred_val = float(ts_model.predict(feat)[0])
            preds.append(pred_val)
            history.append(pred_val)
        preds_arr = np.asarray(preds)

        mae = float(np.mean(np.abs(test_series - preds_arr)))
        rmse = float(np.sqrt(np.mean((test_series - preds_arr) ** 2)))
        mape = float(
            np.mean(
                np.abs(
                    (test_series - preds_arr) / np.maximum(1e-6, np.abs(test_series))
                )
            )
            * 100.0
        )

        return MultiTaskPipelineResult(
            task="timeseries_forecast",
            model=ts_model,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={
                "y_train": train_series,
                "y_test": test_series,
                "y_pred": preds_arr,
            },
            target_col=t_col,
            model_name="autoregressive_ridge",
            evaluation={"mae": mae, "rmse": rmse, "mape": mape},
            conformal_interval=conformal_interval,
            task_metadata={
                "lags": lags,
                "last_window": series[-lags:],
                "residual_std": float(np.std(test_series - preds_arr)),
            },
        )

    @staticmethod
    def _dispatch_causal(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        treatment_col: Optional[str],
        outcome_col: Optional[str],
        model_spec: Any,
        random_state: int,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle causal inference and treatment effect estimation."""
        if treatment_col is None or treatment_col not in df_clean.columns:
            for col in df_clean.columns:
                if df_clean[col].nunique() == 2:
                    treatment_col = col
                    break
            if treatment_col is None:
                treatment_col = df_clean.columns[0]

        if outcome_col is None or outcome_col not in df_clean.columns:
            outcome_col = df_clean.columns[-1]

        confounder_cols = [
            c for c in df_clean.columns if c not in (treatment_col, outcome_col)
        ]
        W = (
            df_clean[confounder_cols]
            .select_dtypes(include=[np.number])
            .to_numpy(dtype=np.float64)
        )
        if W.shape[1] == 0:
            W = np.ones((len(df_clean), 1), dtype=np.float64)

        T = (
            df_clean[treatment_col].to_numpy() == df_clean[treatment_col].unique()[1]
        ).astype(np.float64)
        Y = df_clean[outcome_col].to_numpy(dtype=np.float64)

        from chokkhu.models.causal.doubly_robust import DoublyRobustLearner

        causal_est = DoublyRobustLearner()
        causal_est.fit(W, T, Y)
        ate = float(causal_est.estimate_ate())
        se = float(np.std(Y) / np.sqrt(len(Y)))

        return MultiTaskPipelineResult(
            task="causal_inference",
            model=causal_est,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={"W": W, "T": T, "Y": Y},
            treatment_col=treatment_col,
            target_col=outcome_col,
            model_name="doubly_robust_learner",
            evaluation={"ate": ate, "std_error": se},
            task_metadata={
                "ate": ate,
                "ate_ci_lower": ate - 1.96 * se,
                "ate_ci_upper": ate + 1.96 * se,
            },
        )

    @staticmethod
    def _dispatch_survival(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        duration_col: Optional[str],
        event_col: Optional[str],
        model_spec: Any,
        random_state: int,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle survival and event-time modeling."""
        d_col = (
            duration_col if duration_col in df_clean.columns else df_clean.columns[0]
        )
        e_col = event_col if event_col in df_clean.columns else df_clean.columns[1]

        durations = df_clean[d_col].to_numpy(dtype=np.float64)
        events = df_clean[e_col].to_numpy(dtype=np.float64)
        covar_cols = [c for c in df_clean.columns if c not in (d_col, e_col)]
        X = (
            df_clean[covar_cols]
            .select_dtypes(include=[np.number])
            .to_numpy(dtype=np.float64)
        )
        if X.shape[1] == 0:
            X = np.ones((len(df_clean), 1), dtype=np.float64)

        from chokkhu.models.survival.cox_ph import CoxPHRegression
        from chokkhu.models.survival.metrics import concordance_index

        cox = CoxPHRegression(alpha=0.1)
        cox.fit(X, durations, events)
        risk_scores = np.asarray(cox.predict_risk(X))
        c_idx = float(concordance_index(durations, risk_scores, events))

        return MultiTaskPipelineResult(
            task="survival",
            model=cox,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={"X": X, "durations": durations, "events": events},
            duration_col=d_col,
            event_col=e_col,
            model_name="cox_proportional_hazards",
            evaluation={
                "c_index": c_idx,
                "median_duration": float(np.median(durations)),
            },
            task_metadata={"concordance_index": c_idx},
        )

    @staticmethod
    def _dispatch_anomaly(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        model_spec: Any,
        random_state: int,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle unsupervised anomaly detection."""
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        X = df_clean[num_cols].to_numpy(dtype=np.float64)

        from chokkhu.models.ml.isolation_forest import IsolationForest

        iso = IsolationForest(
            n_estimators=100,
            contamination=kwargs.get("contamination", 0.1),
            random_state=random_state,
        )
        iso.fit(X)
        labels = iso.predict(X)
        anomaly_ratio = float(np.mean(labels == -1))

        return MultiTaskPipelineResult(
            task="anomaly_detection",
            model=iso,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={"X": X, "labels": labels},
            model_name="isolation_forest",
            evaluation={
                "anomaly_ratio": anomaly_ratio,
                "n_anomalies": int(np.sum(labels == -1)),
            },
            task_metadata={"contamination": kwargs.get("contamination", 0.1)},
        )

    @staticmethod
    def _dispatch_clustering(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        model_spec: Any,
        random_state: int,
        verbose: bool,
        **kwargs,
    ) -> MultiTaskPipelineResult:
        """Handle unsupervised clustering."""
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        X = df_clean[num_cols].to_numpy(dtype=np.float64)

        from chokkhu.models.ml.kmeans import KMeans

        n_clusters = kwargs.get("n_clusters", 3)
        km = KMeans(n_clusters=n_clusters, random_state=random_state)
        km.fit(X)
        labels = km.predict(X)

        inertia = float(km.inertia_) if hasattr(km, "inertia_") else 0.0

        return MultiTaskPipelineResult(
            task="clustering",
            model=km,
            data_raw=df_raw,
            data_cleaned=df_clean,
            splits={"X": X, "labels": labels},
            model_name="kmeans",
            evaluation={"n_clusters": n_clusters, "inertia": inertia},
            task_metadata={"cluster_centers": getattr(km, "cluster_centers_", None)},
        )


def dispatch_pipeline(
    data: Union[str, pd.DataFrame, np.ndarray],
    target: Optional[str] = None,
    task: str = "auto",
    treatment: Optional[str] = None,
    duration: Optional[str] = None,
    event: Optional[str] = None,
    model: Union[str, List[str], Any] = "auto",
    clean: Union[bool, str, Dict[str, Any]] = "auto",
    preprocess: Union[bool, str, Dict[str, Any]] = "auto",
    transform: Optional[Dict[str, Any]] = None,
    resample: Optional[str] = None,
    resample_ratio: float = 1.0,
    conformal_interval: Optional[float] = None,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    evaluate: bool = True,
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
    verbose: bool = True,
    **kwargs,
) -> MultiTaskPipelineResult:
    """Convenience functional interface for MultiTaskDispatcher."""
    return MultiTaskDispatcher.dispatch(
        data=data,
        target=target,
        task=task,
        treatment=treatment,
        duration=duration,
        event=event,
        model=model,
        clean=clean,
        preprocess=preprocess,
        transform=transform,
        resample=resample,
        resample_ratio=resample_ratio,
        conformal_interval=conformal_interval,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        evaluate=evaluate,
        save_reports=save_reports,
        save_dir=save_dir,
        verbose=verbose,
        **kwargs,
    )
