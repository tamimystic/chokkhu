"""Sovereign Multi-Model Super Learner & Stacking Ensemble in Pure NumPy."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np


class SuperLearner:
    """Super Learner / Stacking Ensemble with Out-Of-Fold (OOF) Meta-Feature Generation."""

    def __init__(
        self,
        estimators: List[Any],
        meta_estimator: Optional[Any] = None,
        task: str = "classification",
        cv: int = 5,
        use_probabilities: bool = True,
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.task = task
        self.cv = cv
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        self.fitted_estimators_: List[Any] = []
        self.fitted_meta_estimator_: Optional[Any] = None
        self.weights_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _get_oof_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create deterministic K-Fold train/val split indices."""
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

    def _clone_estimator(self, est: Any) -> Any:
        """Create a fresh instance of the estimator with same hyperparameters."""
        import copy
        try:
            cloned = copy.deepcopy(est)
            return cloned
        except Exception:
            cls = est.__class__
            return cls()

    def fit(self, X: np.ndarray, y: np.ndarray) -> SuperLearner:
        """Fit base models with cross-validation to construct OOF meta-features and train meta-learner."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n_samples = X.shape[0]
        splits = self._get_oof_splits(n_samples)

        if self.task == "classification":
            self.classes_ = np.unique(y)

        # 1. Generate Out-Of-Fold meta-features
        oof_features_list = []

        for est_proto in self.estimators:
            oof_col = np.zeros((n_samples,), dtype=np.float32)

            for train_idx, val_idx in splits:
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va = X[val_idx]

                fold_model = self._clone_estimator(est_proto)
                if hasattr(fold_model, "fit"):
                    fold_model.fit(X_tr, y_tr)

                preds = None
                if self.task == "classification" and self.use_probabilities and hasattr(fold_model, "predict_proba"):
                    try:
                        probs = fold_model.predict_proba(X_va)
                        if probs.ndim == 2 and probs.shape[1] == 2:
                            preds = probs[:, 1]
                        elif probs.ndim == 2:
                            preds = probs[:, 0]
                        else:
                            preds = probs
                    except Exception:
                        preds = None

                if preds is None:
                    if hasattr(fold_model, "predict"):
                        preds = fold_model.predict(X_va)
                    else:
                        preds = fold_model(X_va)

                oof_col[val_idx] = np.asarray(preds, dtype=np.float32).squeeze()

            oof_features_list.append(oof_col.reshape(-1, 1))

        Z_oof = np.concatenate(oof_features_list, axis=1)

        # 2. Fit Meta-Learner (or optimal non-negative least squares weighting)
        if self.meta_estimator is not None:
            self.fitted_meta_estimator_ = self._clone_estimator(self.meta_estimator)
            if hasattr(self.fitted_meta_estimator_, "fit"):
                self.fitted_meta_estimator_.fit(Z_oof, y)
        else:
            # Sovereign constrained weight blending via non-negative least squares
            Z_norm = Z_oof - np.mean(Z_oof, axis=0, keepdims=True)
            if self.classes_ is not None and len(self.classes_) == 2:
                y_norm = (y == self.classes_[1]).astype(np.float32)
            else:
                y_norm = y.astype(np.float32)
            y_norm = y_norm - np.mean(y_norm)

            # Solve regularized normal equations: (Z^T Z + lambda I)^-1 Z^T y
            reg = 1e-3 * np.eye(Z_norm.shape[1])
            raw_weights = np.linalg.solve(np.dot(Z_norm.T, Z_norm) + reg, np.dot(Z_norm.T, y_norm))
            # Project onto positive probability simplex
            pos_weights = np.maximum(0.0, raw_weights)
            weight_sum: float = float(np.sum(pos_weights))
            if weight_sum > 0:
                self.weights_ = pos_weights / weight_sum
            else:
                self.weights_ = np.ones(Z_norm.shape[1], dtype=np.float32) / float(Z_norm.shape[1])

        # 3. Fit base estimators on full dataset
        self.fitted_estimators_ = []
        for est_proto in self.estimators:
            full_model = self._clone_estimator(est_proto)
            if hasattr(full_model, "fit"):
                full_model.fit(X, y)
            self.fitted_estimators_.append(full_model)

        return self

    def _transform_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from base estimators into test meta-features matrix."""
        X = np.asarray(X, dtype=np.float32)
        cols = []

        for model in self.fitted_estimators_:
            p = None
            if self.task == "classification" and self.use_probabilities and hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)
                    if probs.ndim == 2 and probs.shape[1] == 2:
                        p = probs[:, 1]
                    elif probs.ndim == 2:
                        p = probs[:, 0]
                    else:
                        p = probs
                except Exception:
                    p = None

            if p is None:
                if hasattr(model, "predict"):
                    p = model.predict(X)
                else:
                    p = model(X)
            cols.append(np.asarray(p, dtype=np.float32).reshape(-1, 1))

        return np.concatenate(cols, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target using stacking ensemble."""
        Z_test = self._transform_meta_features(X)

        if self.fitted_meta_estimator_ is not None:
            return self.fitted_meta_estimator_.predict(Z_test)

        if self.weights_ is not None:
            blended = np.dot(Z_test, self.weights_)
            if self.task == "classification" and self.classes_ is not None:
                if len(self.classes_) == 2:
                    return np.where(blended >= 0.5, self.classes_[1], self.classes_[0])
                else:
                    return np.asarray(np.round(blended), dtype=int)
            return blended

        return np.mean(Z_test, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        Z_test = self._transform_meta_features(X)

        if self.fitted_meta_estimator_ is not None and hasattr(self.fitted_meta_estimator_, "predict_proba"):
            probs = self.fitted_meta_estimator_.predict_proba(Z_test)
            if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
                p1 = np.clip(probs.squeeze(), 0.0, 1.0)
                return np.column_stack([1.0 - p1, p1])
            return probs

        if self.weights_ is not None:
            p1 = np.clip(np.dot(Z_test, self.weights_), 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        p1 = np.clip(np.mean(Z_test, axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


# Alias
StackingEnsemble = SuperLearner
