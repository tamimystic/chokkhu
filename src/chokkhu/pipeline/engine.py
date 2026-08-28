from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from chokkhu.cleaning import clean as clean_fn
from chokkhu.core.logger import Logger
from chokkhu.evaluation.engine import evaluate as eval_fn
from chokkhu.evaluation.metrics import accuracy_score, r2_score
from chokkhu.io.loader import load as load_fn
from chokkhu.models.base import ChokkhuModel
from chokkhu.models.engine import train as train_fn
from chokkhu.preprocessing import PreprocessorState, preprocess as preprocess_fn
from chokkhu.splitting.splitter import train_test_split
from chokkhu.transformation.features import PolynomialFeatures
from chokkhu.transformation.lda import LinearDiscriminantAnalysis
from chokkhu.transformation.pca import PCA, TruncatedSVD
from chokkhu.transformation.resampling import (
    ADASYN,
    SMOTE,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTETomek,
)


class TransformationState:
    """Maintains fitted transformation components (PCA, LDA, Polynomial)
    ensuring ZERO data leakage between train and test/validation sets.
    """

    def __init__(
        self,
        pca: Optional[int] = None,
        pca_variance: Optional[float] = None,
        pca_whiten: bool = False,
        svd: Optional[int] = None,
        lda: Optional[int] = None,
        polynomial: Optional[int] = None,
        interaction_only: bool = False,
        include_bias: bool = False,
    ) -> None:
        self.pca_n = pca
        self.pca_variance = pca_variance
        self.pca_whiten = pca_whiten
        self.svd_n = svd
        self.lda_n = lda
        self.polynomial_degree = polynomial
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.pca_model: Optional[PCA] = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.lda_model: Optional[LinearDiscriminantAnalysis] = None
        self.poly_model: Optional[PolynomialFeatures] = None
        self.num_cols: List[str] = []

    def fit_transform(
        self, X_df: pd.DataFrame, y_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        df = X_df.copy()
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.polynomial_degree is not None and self.polynomial_degree >= 2:
            if self.num_cols:
                self.poly_model = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    interaction_only=self.interaction_only,
                    include_bias=self.include_bias,
                )
                poly_arr = self.poly_model.fit_transform(df[self.num_cols].to_numpy())
                poly_cols = [f"poly_{i}" for i in range(poly_arr.shape[1])]
                poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in self.num_cols]
                df = pd.concat([df[non_num_cols], poly_df], axis=1)
                self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.pca_n is not None or self.pca_variance is not None:
            if self.num_cols:
                self.pca_model = PCA(
                    n_components=self.pca_n,
                    variance_ratio=self.pca_variance,
                    whiten=self.pca_whiten,
                )
                pca_arr = self.pca_model.fit_transform(df[self.num_cols].to_numpy())
                pca_cols = [f"pca_{i}" for i in range(pca_arr.shape[1])]
                pca_df = pd.DataFrame(pca_arr, columns=pca_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in self.num_cols]
                df = pd.concat([df[non_num_cols], pca_df], axis=1)
                self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.svd_n is not None:
            if self.num_cols:
                self.svd_model = TruncatedSVD(n_components=self.svd_n)
                svd_arr = self.svd_model.fit_transform(df[self.num_cols].to_numpy())
                svd_cols = [f"svd_{i}" for i in range(svd_arr.shape[1])]
                svd_df = pd.DataFrame(svd_arr, columns=svd_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in self.num_cols]
                df = pd.concat([df[non_num_cols], svd_df], axis=1)
                self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.lda_n is not None and y_series is not None:
            if self.num_cols:
                self.lda_model = LinearDiscriminantAnalysis(n_components=self.lda_n)
                lda_arr = self.lda_model.fit_transform(
                    df[self.num_cols].to_numpy(), y_series.to_numpy()
                )
                lda_cols = [f"lda_{i}" for i in range(lda_arr.shape[1])]
                lda_df = pd.DataFrame(lda_arr, columns=lda_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in self.num_cols]
                df = pd.concat([df[non_num_cols], lda_df], axis=1)
                self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return df

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        df = X_df.copy()

        if self.poly_model is not None:
            num_cols_poly = [c for c in self.num_cols if c in df.columns]
            if num_cols_poly:
                poly_arr = self.poly_model.transform(df[num_cols_poly].to_numpy())
                poly_cols = [f"poly_{i}" for i in range(poly_arr.shape[1])]
                poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in num_cols_poly]
                df = pd.concat([df[non_num_cols], poly_df], axis=1)

        if self.pca_model is not None:
            cur_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if cur_num:
                pca_arr = self.pca_model.transform(df[cur_num].to_numpy())
                pca_cols = [f"pca_{i}" for i in range(pca_arr.shape[1])]
                pca_df = pd.DataFrame(pca_arr, columns=pca_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in cur_num]
                df = pd.concat([df[non_num_cols], pca_df], axis=1)

        if self.svd_model is not None:
            cur_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if cur_num:
                svd_arr = self.svd_model.transform(df[cur_num].to_numpy())
                svd_cols = [f"svd_{i}" for i in range(svd_arr.shape[1])]
                svd_df = pd.DataFrame(svd_arr, columns=svd_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in cur_num]
                df = pd.concat([df[non_num_cols], svd_df], axis=1)

        if self.lda_model is not None:
            cur_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if cur_num:
                lda_arr = self.lda_model.transform(df[cur_num].to_numpy())
                lda_cols = [f"lda_{i}" for i in range(lda_arr.shape[1])]
                lda_df = pd.DataFrame(lda_arr, columns=lda_cols, index=df.index)
                non_num_cols = [c for c in df.columns if c not in cur_num]
                df = pd.concat([df[non_num_cols], lda_df], axis=1)

        return df


class PipelineResult:
    """Encapsulates the complete results and trained artifact of an End-to-End Chokkhu Pipeline.
    Provides a simple, leak-free .predict() interface for new inference data.
    """

    def __init__(
        self,
        data_raw: pd.DataFrame,
        data_cleaned: pd.DataFrame,
        splits: Dict[str, Any],
        preprocessor_state: PreprocessorState,
        transformation_state: Optional[TransformationState],
        model: ChokkhuModel,
        model_name: str,
        task: str,
        target_col: str,
        evaluation: Dict[str, Any],
        cv_scores: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.data_raw = data_raw
        self.data_cleaned = data_cleaned
        self.splits = splits
        self.preprocessor_state = preprocessor_state
        self.transformation_state = transformation_state
        self.model = model
        self.model_name = model_name
        self.task = task
        self.target_col = target_col
        self.evaluation = evaluation
        self.cv_scores = cv_scores or {}
        self.feature_names = feature_names or []

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.evaluation

    def predict(
        self, new_data: Union[pd.DataFrame, np.ndarray, dict, list, str]
    ) -> np.ndarray:
        """Runs the fitted preprocessing and transformation pipeline and returns model predictions
        on unseen data with ZERO data leakage.
        """
        if isinstance(new_data, str):
            from chokkhu.io import load

            df = load(new_data)
        elif isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, np.ndarray):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            df = new_data.copy()
        else:
            raise TypeError(
                "Unsupported data type for inference. Must be DataFrame, dict, list, ndarray, or file path."
            )

        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        df_proc = self.preprocessor_state.transform(df)

        if self.transformation_state is not None:
            df_proc = self.transformation_state.transform(df_proc)

        X_arr = df_proc.to_numpy(dtype=np.float64)
        return self.model.predict(X_arr)

    def predict_proba(
        self, new_data: Union[pd.DataFrame, np.ndarray, dict, list]
    ) -> np.ndarray:
        """Returns predicted class probabilities if model supports predict_proba."""
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, np.ndarray):
            df = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            df = new_data.copy()
        else:
            raise TypeError("Unsupported data type for inference.")

        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        df_proc = self.preprocessor_state.transform(df)
        if self.transformation_state is not None:
            df_proc = self.transformation_state.transform(df_proc)

        X_arr = df_proc.to_numpy(dtype=np.float64)
        return self.model.predict_proba(X_arr)

    def summary(self) -> str:
        """Returns a formatted summary of the entire pipeline execution."""
        lines = [
            "=" * 60,
            "              CHOKKHU END-TO-END PIPELINE SUMMARY",
            "=" * 60,
            f"  Task Type         : {self.task.upper()}",
            f"  Target Column     : {self.target_col}",
            f"  Original Shape    : {self.data_raw.shape}",
            f"  Cleaned Shape     : {self.data_cleaned.shape}",
            f"  Train Set Size    : {len(self.splits.get('y_train', []))} samples",
            f"  Test Set Size     : {len(self.splits.get('y_test', []))} samples",
            f"  Selected Model    : {self.model_name}",
            "-" * 60,
            "  EVALUATION METRICS (Test Set):",
        ]
        for metric, val in self.evaluation.items():
            if isinstance(val, (int, float, np.floating, np.integer)):
                lines.append(f"    - {metric.capitalize():<18}: {val:.4f}")
            elif isinstance(val, str):
                lines.append(f"    - {metric.capitalize():<18}: {val}")

        if self.cv_scores:
            lines.append("-" * 60)
            lines.append("  MODEL SELECTION CROSS-VALIDATION SCORES:")
            for m_name, score in sorted(
                self.cv_scores.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    - {m_name:<20}: {score:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str) -> str:
        """Saves the entire pipeline result object and preprocessor state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        Logger.info(f"Saved Chokkhu PipelineResult to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> PipelineResult:
        """Loads a saved pipeline result from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline file not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a PipelineResult (got {type(obj)})")
        Logger.info(f"Loaded Chokkhu PipelineResult from: {path}")
        return obj


def pipeline(
    data: Union[str, pd.DataFrame],
    target: str,
    clean: Union[bool, str, Dict[str, Any]] = "auto",
    preprocess: Union[bool, str, Dict[str, Any]] = "auto",
    transform: Optional[Dict[str, Any]] = None,
    resample: Optional[str] = None,
    resample_ratio: float = 1.0,
    smote_k: int = 5,
    model: Union[str, List[str]] = "auto",
    task: str = "auto",
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify: bool = True,
    random_state: int = 42,
    evaluate: bool = True,
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
    verbose: bool = True,
    **kwargs,
) -> PipelineResult:
    """Executes the complete End-to-End Machine Learning Pipeline with ZERO Data Leakage.

    Execution Flow:
    1. Loads raw dataset
    2. Performs data sanitation & cleaning (type fixes, duplicate removal)
    3. Splits data into Train and Test sets BEFORE feature processing
    4. Fits encoders, scalers, and feature selectors STRICTLY on X_train
    5. Transforms X_train, X_test, and X_val using the fitted PreprocessorState
    6. Fits transformations (PCA, LDA, Polynomial) on X_train and applies to test
    7. Applies resampling (e.g. SMOTE) ONLY to training data (X_train, y_train)
    8. Trains model (or performs automated model selection across candidates)
    9. Evaluates model performance on the untouched test set
    10. Returns a leak-free PipelineResult object with .predict() support.
    """
    if verbose:
        Logger.info("Starting Chokkhu End-to-End Pipeline Execution...")

    if isinstance(data, str):
        df_raw = load_fn(data, verbose=False)
    elif isinstance(data, pd.DataFrame):
        df_raw = data.copy()
    else:
        raise TypeError("Data must be a file path string or pandas DataFrame.")

    if target not in df_raw.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset columns: {list(df_raw.columns)}"
        )

    df_clean = df_raw.copy()
    if clean:
        clean_kwargs: Dict[str, Any] = {}
        if isinstance(clean, dict):
            clean_kwargs = clean
        df_clean = clean_fn(
            df_clean,
            missing=clean_kwargs.get("missing", "median"),
            outliers=clean_kwargs.get("outliers", "iqr"),
            duplicates=clean_kwargs.get("duplicates", True),
            fix_data_types=clean_kwargs.get("fix_data_types", True),
            verbose=False,
        )

    df_clean = df_clean.dropna(subset=[target])

    y = df_clean[target]
    X = df_clean.drop(columns=[target])

    if task == "auto":
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            task = "regression"
        else:
            task = "classification"

    use_stratify = stratify if task == "classification" else False

    # Step 3: Split BEFORE Preprocessing & Transformation (CRITICAL: Zero Data Leakage)
    if val_size is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            val_size=val_size,
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

    # Step 4: Preprocessing (Fit on X_train, Transform on Test & Val)
    prep_kwargs: Dict[str, Any] = {}
    if isinstance(preprocess, dict):
        prep_kwargs = preprocess
    elif preprocess in (True, "auto"):
        prep_kwargs = {
            "scale": "standard",
            "encode": "onehot",
            "select_features": None,
        }

    scale_opt = prep_kwargs.get("scale", "standard" if preprocess else None)
    encode_opt = prep_kwargs.get("encode", "onehot" if preprocess else None)
    select_opt = prep_kwargs.get("select_features", None)
    select_k = prep_kwargs.get("select_k", None)

    X_train_proc, preprocessor_state = preprocess_fn(
        X_train,
        target=None,
        scale=scale_opt,
        encode=encode_opt,
        select_features=select_opt,
        select_k=select_k,
        verbose=False,
    )

    X_test_proc = preprocessor_state.transform(X_test)
    X_val_proc = preprocessor_state.transform(X_val) if X_val is not None else None

    # Step 5: Transformations (PCA, LDA, Polynomial)
    transform_state: Optional[TransformationState] = None
    if transform:
        transform_state = TransformationState(
            pca=transform.get("pca", None),
            pca_variance=transform.get("pca_variance", None),
            pca_whiten=transform.get("pca_whiten", False),
            svd=transform.get("svd", None),
            lda=transform.get("lda", None),
            polynomial=transform.get("polynomial", None),
            interaction_only=transform.get("interaction_only", False),
            include_bias=transform.get("include_bias", False),
        )
        X_train_proc = transform_state.fit_transform(X_train_proc, y_series=y_train)
        X_test_proc = transform_state.transform(X_test_proc)
        if X_val_proc is not None:
            X_val_proc = transform_state.transform(X_val_proc)

    # Step 6: Resampling (Applied ONLY to training set, never test!)
    X_train_final = X_train_proc.to_numpy(dtype=np.float64)
    y_train_final = y_train.to_numpy()

    if resample and task == "classification":
        sampler: Any = None
        if resample == "smote":
            sampler = SMOTE(
                k_neighbors=smote_k, ratio=resample_ratio, random_state=random_state
            )
        elif resample == "adasyn":
            sampler = ADASYN(
                k_neighbors=smote_k, ratio=resample_ratio, random_state=random_state
            )
        elif resample == "random_oversample":
            sampler = RandomOverSampler(ratio=resample_ratio, random_state=random_state)
        elif resample == "random_undersample":
            sampler = RandomUnderSampler(
                ratio=resample_ratio, random_state=random_state
            )
        elif resample == "smote_tomek":
            sampler = SMOTETomek(
                k_neighbors=smote_k, ratio=resample_ratio, random_state=random_state
            )

        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train_final, y_train_final)
            X_train_final = np.asarray(X_res, dtype=np.float64)
            y_train_final = np.asarray(y_res)
            if verbose:
                Logger.info(
                    f"Resampled training set from {len(y_train)} to {len(y_train_final)} samples."
                )

    X_test_final = X_test_proc.to_numpy(dtype=np.float64)
    y_test_final = y_test.to_numpy()

    # Step 7: Model Selection and Training
    cv_scores: Dict[str, float] = {}
    chosen_model_name: str = ""

    if model == "auto" or isinstance(model, list):
        if isinstance(model, list):
            candidates = model
        elif task == "classification":
            candidates = [
                "random_forest",
                "logistic_regression",
                "decision_tree",
                "knn",
                "gradient_boosting",
                "naive_bayes",
            ]
        else:
            candidates = [
                "random_forest",
                "linear_regression",
                "ridge",
                "decision_tree",
                "knn",
                "gradient_boosting",
            ]

        if verbose:
            Logger.info(f"Evaluating candidate models: {candidates}")

        val_split_n = int(len(X_train_final) * 0.2)
        X_tr_cand = X_train_final[val_split_n:]
        y_tr_cand = y_train_final[val_split_n:]
        X_va_cand = X_train_final[:val_split_n]
        y_va_cand = y_train_final[:val_split_n]

        best_score = -float("inf")
        best_candidate = candidates[0]

        for cand in candidates:
            try:
                m_obj = train_fn(
                    model=cand,
                    X_train=X_tr_cand,
                    y_train=y_tr_cand,
                    task=task,
                    random_state=random_state,
                    verbose=False,
                )
                preds = m_obj.predict(X_va_cand)
                if task == "classification":
                    score = accuracy_score(y_va_cand, preds)
                else:
                    score = r2_score(y_va_cand, preds)
                cv_scores[cand] = score
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            except Exception as e:
                if verbose:
                    Logger.warning(
                        f"Candidate model '{cand}' failed during evaluation: {e}"
                    )

        chosen_model_name = best_candidate
        if verbose:
            Logger.info(
                f"Best model selected: '{chosen_model_name}' (validation score: {best_score:.4f})"
            )
    else:
        chosen_model_name = model

    pipeline_param_names = {
        "clean_missing",
        "clean_outliers",
        "clean_duplicates",
        "clean_dtypes",
        "scale",
        "encode",
        "select_features",
        "select_k",
        "pca",
        "lda",
        "svd",
        "tsne",
        "resample",
        "resample_ratio",
        "polynomial",
        "test_size",
        "val_size",
        "stratify",
        "evaluate",
        "save_reports",
        "report_dir",
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k not in pipeline_param_names}

    fitted_model = train_fn(
        model=chosen_model_name,
        X_train=X_train_final,
        y_train=y_train_final,
        task=task,
        random_state=random_state,
        verbose=verbose,
        **model_kwargs,
    )

    eval_results: Dict[str, Any] = {}
    if evaluate:
        eval_results = eval_fn(
            model=fitted_model,
            X_test=X_test_final,
            y_test=y_test_final,
            task=task,
            save_reports=save_reports,
            save_dir=save_dir,
        )

    splits_dict = {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": y_train_final,
        "y_test": y_test_final,
    }
    if X_val is not None:
        splits_dict["X_val"] = (
            X_val_proc.to_numpy(dtype=np.float64) if X_val_proc is not None else None
        )
        splits_dict["y_val"] = y_val.to_numpy() if y_val is not None else None

    result = PipelineResult(
        data_raw=df_raw,
        data_cleaned=df_clean,
        splits=splits_dict,
        preprocessor_state=preprocessor_state,
        transformation_state=transform_state,
        model=fitted_model,
        model_name=chosen_model_name,
        task=task,
        target_col=target,
        evaluation=eval_results,
        cv_scores=cv_scores,
        feature_names=list(X_train_proc.columns),
    )

    if verbose:
        Logger.info(
            "Chokkhu Pipeline execution completed successfully with ZERO Data Leakage."
        )

    return result
