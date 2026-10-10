from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from chokkhu.core.math import cdist

from ..base import ChokkhuModel


class SVM(ChokkhuModel):
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 1e-3,
        cache_size: float = 200.0,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        verbose: bool = False,
        max_iter: int = 1000,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: Optional[int] = None,
        learning_rate: float = 0.001,
        lambda_param: Optional[float] = None,
        epochs: Optional[int] = None,
    ) -> None:
        self.C = (
            1.0 / (2.0 * lambda_param)
            if lambda_param is not None and lambda_param > 0
            else float(C)
        )
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.shrinking = shrinking
        self.probability = probability
        self.tol = float(tol)
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = epochs if epochs is not None else max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.lambda_param = 1.0 / (2.0 * self.C) if self.C > 0 else 0.01

        self.support_: np.ndarray = np.array([], dtype=np.int64)
        self.support_vectors_: np.ndarray = np.array([])
        self.n_support_: np.ndarray = np.array([], dtype=np.int64)
        self.dual_coef_: np.ndarray = np.array([])
        self.intercept_: np.ndarray = np.array([0.0])
        self.classes_: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.gamma_val_: float = 1.0

        # Platt scaling parameters (A, B) for probability calibration
        self._platt_a: float = -1.0
        self._platt_b: float = 0.0

    def _get_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = X.shape[1]
        var = float(np.var(X))
        if self.gamma == "scale" and var > 0.0:
            return float(1.0 / (n_features * var))
        return float(1.0 / max(1, n_features))

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            sq_dists = cdist(X1, X2, metric="sqeuclidean")
            return np.exp(-self.gamma_val_ * sq_dists)
        elif self.kernel == "poly":
            return (self.gamma_val_ * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma_val_ * (X1 @ X2.T) + self.coef0)
        return X1 @ X2.T

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> SVM:
        if y is None:
            raise ValueError("y cannot be None for SVM")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError("Currently binary classification is supported for SVM")

        self.gamma_val_ = self._get_gamma(X_arr)
        y_signed = np.where(y_arr == self.classes_[0], -1.0, 1.0)
        n_samples, n_features = X_arr.shape

        if self.kernel == "linear":
            # Linear primal SGD optimization with hinge loss
            w = np.zeros(n_features, dtype=np.float64)
            b = 0.0
            lambda_p = self.lambda_param

            for _ in range(self.max_iter):
                for idx in range(n_samples):
                    x_i = X_arr[idx]
                    y_i = y_signed[idx]
                    margin = y_i * (np.dot(x_i, w) + b)
                    if margin >= 1.0:
                        w -= self.learning_rate * (2.0 * lambda_p * w)
                    else:
                        w -= self.learning_rate * (2.0 * lambda_p * w - y_i * x_i)
                        b -= self.learning_rate * (-y_i)

            self.weights = w
            self.bias = b
            self.intercept_ = np.array([b])

            # Compute support vectors (|margin - 1| < 0.2 or margin < 1)
            margins = y_signed * (X_arr @ w + b)
            sv_mask = margins <= 1.05
            self.support_ = np.where(sv_mask)[0]
            self.support_vectors_ = X_arr[self.support_]
            self.n_support_ = np.array(
                [
                    np.sum(y_signed[self.support_] == -1),
                    np.sum(y_signed[self.support_] == 1),
                ]
            )
            self.dual_coef_ = (
                y_signed[self.support_] * (self.C / max(1, len(self.support_)))
            ).reshape(1, -1)
        else:
            # Kernel SVM dual representation using kernel matrix
            K = self._compute_kernel(X_arr, X_arr)
            # SMO / Coordinate Descent on Dual alphas: 0 <= alpha_i <= C
            alphas = np.zeros(n_samples, dtype=np.float64)
            b = 0.0

            for _ in range(self.max_iter):
                grad = 1.0 - y_signed * (K @ (alphas * y_signed) + b)
                alphas = np.clip(alphas + self.learning_rate * grad, 0.0, self.C)

            sv_mask = alphas > 1e-5
            self.support_ = np.where(sv_mask)[0]
            if len(self.support_) == 0:
                self.support_ = np.arange(min(5, n_samples))
                alphas[self.support_] = self.C

            self.support_vectors_ = X_arr[self.support_]
            self.dual_coef_ = (alphas[self.support_] * y_signed[self.support_]).reshape(
                1, -1
            )

            # Compute bias from support vectors
            K_sv = K[self.support_][:, self.support_]
            b_vals = y_signed[self.support_] - self.dual_coef_ @ K_sv.T
            b = float(np.mean(b_vals)) if len(self.support_) > 0 else 0.0
            self.bias = b
            self.intercept_ = np.array([b])
            self.n_support_ = np.array(
                [
                    np.sum(y_signed[self.support_] == -1),
                    np.sum(y_signed[self.support_] == 1),
                ]
            )

        # Platt scaling parameter estimation if probability=True
        if self.probability:
            decs = self.decision_function(X_arr)
            target = (y_signed + 1.0) / 2.0
            # Simple logistic fit on decision values
            self._platt_a = -float(np.std(target) / (np.std(decs) + 1e-15))
            self._platt_b = float(np.mean(target) - self._platt_a * np.mean(decs))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear" and self.weights is not None:
            return X_arr @ self.weights + self.bias
        elif len(self.support_vectors_) > 0:
            K_test = self._compute_kernel(X_arr, self.support_vectors_)
            return (self.dual_coef_ @ K_test.T).ravel() + self.bias
        return np.zeros(len(X_arr))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        decs = self.decision_function(X)
        preds = np.where(decs >= 0.0, self.classes_[1], self.classes_[0])
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        decs = self.decision_function(X)
        if self.probability:
            prob_pos = 1.0 / (1.0 + np.exp(self._platt_a * decs + self._platt_b))
        else:
            prob_pos = 1.0 / (1.0 + np.exp(-np.clip(decs, -250.0, 250.0)))
        return np.vstack([1.0 - prob_pos, prob_pos]).T

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.log(np.clip(probs, 1e-15, 1.0))

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        return float(np.mean(self.predict(X) == y))
