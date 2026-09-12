"""SuperLearner Multi-Layer Out-of-Fold Stacking Pipeline in Pure NumPy."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from chokkhu.models.base import ChokkhuModel


class SuperLearner(ChokkhuModel):
    """SuperLearner Multi-Layer Out-Of-Fold (OOF) Stacking Ensemble.

    Constructs cross-validated out-of-fold meta-feature matrices:
    Z_meta = [y_hat_OOF_1, y_hat_OOF_2, ..., y_hat_OOF_M]
    and trains a Level-2 meta-learner (or computes optimal non-negative constrained weights)
    while avoiding any data leakage between training folds and validation holdouts.

    Parameters
    ----------
    estimators : list of estimators or model names
        Base estimators to stack.
    meta_estimator : estimator, optional
        Level-2 meta-learner. If None, computes constrained non-negative least squares weights.
    task : str, default="classification"
        Task type ("classification" or "regression").
    cv : int, default=5
        Number of cross-validation folds for OOF meta-feature generation.
    passthrough : bool, default=False
        If True, concatenates original input features X alongside meta-features Z_meta.
    use_probabilities : bool, default=True
        For classification, whether to pass predicted probabilities instead of hard class labels.
    random_state : int, default=42
        Random seed for fold partitioning.
    """

    def __init__(
        self,
        estimators: List[Any],
        meta_estimator: Optional[Any] = None,
        task: str = "classification",
        cv: int = 5,
        passthrough: bool = False,
        use_probabilities: bool = True,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.task = str(task).lower()
        self.cv = max(2, int(cv))
        self.passthrough = bool(passthrough)
        self.use_probabilities = bool(use_probabilities)
        self.random_state = int(random_state)

        self.fitted_estimators_: List[Any] = []
        self.fitted_meta_estimator_: Optional[Any] = None
        self.weights_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.is_fitted_: bool = False

    def _clone_estimator(self, est: Any) -> Any:
        """Create a fresh instance of the base estimator."""
        try:
            return copy.deepcopy(est)
        except Exception:
            cls = est.__class__
            return cls()

    def _get_oof_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate deterministic K-Fold train/val split indices."""
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        folds = np.array_split(indices, self.cv)
        splits = []
        for i in range(self.cv):
            val_idx = folds[i]
            train_idx = np.setdiff1d(indices, val_idx)
            splits.append((train_idx, val_idx))
        return splits

    def _get_preds(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Helper to extract predictions (probabilities or continuous values) from a model."""
        if (
            self.task == "classification"
            and self.use_probabilities
            and hasattr(model, "predict_proba")
        ):
            try:
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    return probs[:, 1]
                elif probs.ndim == 2 and probs.shape[1] > 2:
                    return probs
                elif probs.ndim == 1:
                    return probs
            except Exception:
                pass

        if hasattr(model, "predict"):
            preds = model.predict(X)
        elif callable(model):
            preds = model(X)
        else:
            raise AttributeError(f"Estimator {model} has no predict method.")

        return np.asarray(preds, dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "SuperLearner":
        """Fit base estimators using K-Fold CV to generate OOF meta-features,
        then fits the meta-estimator and all full base models.
        """
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y)
        n_samples, n_features = X_mat.shape
        self.n_features_in_ = n_features

        if self.task == "classification":
            self.classes_ = np.unique(y_vec)

        splits = self._get_oof_splits(n_samples)
        oof_features_list: List[np.ndarray] = []

        # 1. Out-of-fold feature generation
        for est_proto in self.estimators:
            fold_preds: List[np.ndarray] = []
            val_indices: List[np.ndarray] = []

            for train_idx, val_idx in splits:
                X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
                X_va = X_mat[val_idx]

                fold_model = self._clone_estimator(est_proto)
                if hasattr(fold_model, "fit"):
                    fold_model.fit(X_tr, y_tr)
                elif hasattr(fold_model, "train"):
                    fold_model.train(X_tr, y_tr)

                preds = self._get_preds(fold_model, X_va)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)

                fold_preds.append(preds)
                val_indices.append(val_idx)

            n_cols = fold_preds[0].shape[1]
            est_oof = np.zeros((n_samples, n_cols), dtype=np.float64)
            for v_idx, f_pred in zip(val_indices, fold_preds):
                est_oof[v_idx] = f_pred

            oof_features_list.append(est_oof)

        Z_oof = np.concatenate(oof_features_list, axis=1)

        if self.passthrough:
            Z_oof = np.hstack([X_mat, Z_oof])

        # 2. Fit Meta-Estimator or optimal constrained simplex weights
        if self.meta_estimator is not None:
            self.fitted_meta_estimator_ = self._clone_estimator(self.meta_estimator)
            if hasattr(self.fitted_meta_estimator_, "fit"):
                self.fitted_meta_estimator_.fit(Z_oof, y_vec)
            elif hasattr(self.fitted_meta_estimator_, "train"):
                self.fitted_meta_estimator_.train(Z_oof, y_vec)
        else:
            Z_norm = Z_oof - np.mean(Z_oof, axis=0, keepdims=True)
            if self.classes_ is not None and len(self.classes_) == 2:
                y_numeric = (y_vec == self.classes_[1]).astype(np.float64)
            else:
                y_numeric = y_vec.astype(np.float64)
            y_norm = y_numeric - np.mean(y_numeric)

            reg = 1e-4 * np.eye(Z_norm.shape[1])
            try:
                raw_w = np.linalg.solve(
                    np.dot(Z_norm.T, Z_norm) + reg, np.dot(Z_norm.T, y_norm)
                )
            except np.linalg.LinAlgError:
                raw_w = np.ones(Z_norm.shape[1]) / float(Z_norm.shape[1])

            pos_w = np.maximum(0.0, raw_w)
            w_sum = float(np.sum(pos_w))
            if w_sum > 0:
                self.weights_ = pos_w / w_sum
            else:
                self.weights_ = np.ones(Z_norm.shape[1], dtype=np.float64) / float(
                    Z_norm.shape[1]
                )

        # 3. Fit full base models on entire dataset
        self.fitted_estimators_ = []
        for est_proto in self.estimators:
            full_model = self._clone_estimator(est_proto)
            if hasattr(full_model, "fit"):
                full_model.fit(X_mat, y_vec)
            elif hasattr(full_model, "train"):
                full_model.train(X_mat, y_vec)
            self.fitted_estimators_.append(full_model)

        self.is_fitted_ = True
        return self

    def _construct_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-feature matrix for inference data using fitted base models."""
        X_mat = np.asarray(X, dtype=np.float64)
        meta_cols: List[np.ndarray] = []

        for model in self.fitted_estimators_:
            preds = self._get_preds(model, X_mat)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            meta_cols.append(preds)

        Z = np.concatenate(meta_cols, axis=1)
        if self.passthrough:
            Z = np.hstack([X_mat, Z])
        return Z

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict target using the stacked meta-features."""
        if not self.is_fitted_:
            raise RuntimeError("SuperLearner has not been fitted yet.")

        X_mat = np.asarray(X, dtype=np.float64)
        Z_test = self._construct_meta_features(X_mat)

        if self.fitted_meta_estimator_ is not None:
            if hasattr(self.fitted_meta_estimator_, "predict"):
                return np.asarray(self.fitted_meta_estimator_.predict(Z_test))
            elif callable(self.fitted_meta_estimator_):
                return np.asarray(self.fitted_meta_estimator_(Z_test))

        if self.weights_ is not None:
            continuous_pred = np.dot(Z_test, self.weights_)
            if self.task == "classification" and self.classes_ is not None:
                if len(self.classes_) == 2:
                    return np.where(
                        continuous_pred >= 0.5, self.classes_[1], self.classes_[0]
                    )
                else:
                    return np.asarray(np.round(continuous_pred), dtype=int)
            return continuous_pred

        raise RuntimeError("No meta-estimator or weights available for prediction.")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("SuperLearner has not been fitted yet.")
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        X_mat = np.asarray(X, dtype=np.float64)
        Z_test = self._construct_meta_features(X_mat)

        if self.fitted_meta_estimator_ is not None and hasattr(
            self.fitted_meta_estimator_, "predict_proba"
        ):
            return np.asarray(self.fitted_meta_estimator_.predict_proba(Z_test))

        if self.weights_ is not None:
            p1 = np.clip(np.dot(Z_test, self.weights_), 0.0, 1.0)
            if self.classes_ is not None and len(self.classes_) == 2:
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
            return p1.reshape(-1, 1)

        raise RuntimeError("Meta-estimator does not support predict_proba.")

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            "task": self.task,
            "cv": self.cv,
            "passthrough": self.passthrough,
            "use_probabilities": self.use_probabilities,
            "random_state": self.random_state,
            "n_estimators": len(self.estimators),
            "is_fitted": self.is_fitted_,
        }


# Alias for compatibility
StackingPipeline = SuperLearner
