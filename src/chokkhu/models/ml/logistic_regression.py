from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np

from ..base import ChokkhuModel


class LogisticRegression(ChokkhuModel):
    def __init__(
        self,
        penalty: Optional[str] = "l2",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        random_state: Optional[int] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        l1_ratio: Optional[float] = None,
        learning_rate: float = 0.01,
        regularization: Optional[str] = None,
        alpha: Optional[float] = None,
        epochs: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        self.penalty = regularization if regularization is not None else penalty
        self.dual = dual
        self.tol = tolerance if tolerance is not None else tol
        self.C = 1.0 / alpha if alpha is not None and alpha > 0 else float(C)
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = epochs if epochs is not None else max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate

        self.classes_: Optional[np.ndarray] = None
        self.coef_: np.ndarray = np.array([])
        self.intercept_: np.ndarray = np.array([])
        self.n_features_in_: int = 0
        self.n_iter_: int = 0
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-15)

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            w_map = {
                c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)
            }
        elif isinstance(self.class_weight, dict):
            w_map = self.class_weight
        else:
            return np.ones(len(y), dtype=np.float64)
        return np.array([w_map.get(v, 1.0) for v in y], dtype=np.float64)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> LogisticRegression:
        if y is None:
            raise ValueError("y cannot be None for Logistic Regression")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        sample_weights = self._compute_sample_weights(y_arr)

        reg_alpha = 1.0 / max(1e-12, self.C)

        if n_classes == 2:
            # Binary classification
            pos_label = self.classes_[1]
            y_bin = (y_arr == pos_label).astype(np.float64)

            w = np.zeros(n_features, dtype=np.float64)
            b = 0.0

            for it in range(self.max_iter):
                linear = X_arr @ w + (b if self.fit_intercept else 0.0)
                y_pred = self._sigmoid(linear)
                error = (y_pred - y_bin) * sample_weights

                dw = (1.0 / n_samples) * (X_arr.T @ error)
                db = (
                    (1.0 / n_samples) * float(np.sum(error))
                    if self.fit_intercept
                    else 0.0
                )

                if self.penalty == "l2":
                    dw += (reg_alpha / n_samples) * w
                elif self.penalty == "l1":
                    dw += (reg_alpha / n_samples) * np.sign(w)
                elif self.penalty in ("elasticnet", "elastic_net"):
                    ratio = self.l1_ratio if self.l1_ratio is not None else 0.5
                    dw += (reg_alpha / n_samples) * (
                        ratio * np.sign(w) + (1.0 - ratio) * w
                    )

                w -= self.learning_rate * dw
                if self.fit_intercept:
                    b -= self.learning_rate * db

                if float(np.max(np.abs(self.learning_rate * dw))) < self.tol:
                    break

            self.coef_ = w.reshape(1, -1)
            self.intercept_ = np.array([b])
            self.weights = w
            self.bias = b
            self.n_iter_ = it + 1
        else:
            # Multi-class One-vs-Rest (OVR)
            all_coefs = []
            all_intercepts = []
            for c in self.classes_:
                y_c = (y_arr == c).astype(np.float64)
                w = np.zeros(n_features, dtype=np.float64)
                b = 0.0
                for _ in range(self.max_iter):
                    linear = X_arr @ w + (b if self.fit_intercept else 0.0)
                    y_pred = self._sigmoid(linear)
                    error = (y_pred - y_c) * sample_weights
                    dw = (1.0 / n_samples) * (X_arr.T @ error)
                    db = (
                        (1.0 / n_samples) * float(np.sum(error))
                        if self.fit_intercept
                        else 0.0
                    )

                    if self.penalty == "l2":
                        dw += (reg_alpha / n_samples) * w
                    elif self.penalty == "l1":
                        dw += (reg_alpha / n_samples) * np.sign(w)

                    w -= self.learning_rate * dw
                    if self.fit_intercept:
                        b -= self.learning_rate * db
                    if float(np.max(np.abs(self.learning_rate * dw))) < self.tol:
                        break

                all_coefs.append(w)
                all_intercepts.append(b)

            self.coef_ = np.array(all_coefs)
            self.intercept_ = np.array(all_intercepts)
            self.weights = self.coef_[0]
            self.bias = float(self.intercept_[0])
            self.n_iter_ = self.max_iter

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or len(self.coef_) == 0:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        scores = X_arr @ self.coef_.T + self.intercept_
        if self.classes_ is not None and len(self.classes_) == 2:
            return scores.ravel()
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self.classes_ is not None and len(self.classes_) == 2:
            prob_1 = self._sigmoid(scores).reshape(-1, 1)
            return np.hstack([1.0 - prob_1, prob_1])
        else:
            return self._softmax(scores)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.log(np.clip(probs, 1e-15, 1.0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is not None and len(self.classes_) == 2:
            scores = self.decision_function(X)
            preds = (scores >= 0.0).astype(int)
            return self.classes_[preds]
        else:
            probas = self.predict_proba(X)
            idx = np.argmax(probas, axis=1)
            if self.classes_ is not None:
                return self.classes_[idx]
            return idx

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
