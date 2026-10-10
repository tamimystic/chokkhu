from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from chokkhu.models.base import ChokkhuModel


class GaussianNB(ChokkhuModel):
    def __init__(
        self,
        priors: Optional[np.ndarray] = None,
        var_smoothing: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.priors = priors
        self.var_smoothing = float(var_smoothing)
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.class_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.theta_: np.ndarray = np.array([], dtype=np.float64)
        self.var_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> GaussianNB:
        if y is None:
            raise ValueError("y cannot be None for GaussianNB")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)
        epsilon = self.var_smoothing * float(np.var(x_arr, axis=0).max())

        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            self.class_prior_[idx] = (
                self.priors[idx]
                if self.priors is not None
                else len(x_cls) / float(n_samples)
            )
            self.theta_[idx] = np.mean(x_cls, axis=0)
            self.var_[idx] = np.var(x_cls, axis=0) + max(epsilon, 1e-9)

        if self.priors is not None:
            self.class_prior_ = self.class_prior_ / np.sum(self.class_prior_)

        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_classes = x_arr.shape[0], len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        for idx in range(n_classes):
            mean, var = self.theta_[idx], self.var_[idx]
            log_lik = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((x_arr - mean) ** 2) / var, axis=1
            )
            log_probs[:, idx] = (
                np.log(np.maximum(self.class_prior_[idx], 1e-12)) + log_lik
            )
        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(shifted)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


class MultinomialNB(ChokkhuModel):
    def __init__(
        self,
        alpha: float = 1.0,
        force_alpha: bool = True,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.class_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultinomialNB:
        if y is None:
            raise ValueError("y cannot be None for MultinomialNB")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            if self.class_prior is not None:
                self.class_prior_[idx] = self.class_prior[idx]
            elif self.fit_prior:
                self.class_prior_[idx] = len(x_cls) / float(n_samples)
            else:
                self.class_prior_[idx] = 1.0 / float(n_classes)

            count_per_feat = np.sum(x_cls, axis=0) + self.alpha
            total_count = float(np.sum(count_per_feat))
            self.feature_log_prob_[idx] = np.log(count_per_feat / total_count)

        self.class_prior_ = self.class_prior_ / np.sum(self.class_prior_)
        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        log_prior = np.log(np.maximum(self.class_prior_, 1e-12))
        return np.dot(x_arr, self.feature_log_prob_.T) + log_prior

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(shifted)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


class BernoulliNB(ChokkhuModel):
    def __init__(
        self,
        alpha: float = 1.0,
        force_alpha: bool = True,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.class_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_log_prob_neg_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BernoulliNB:
        if y is None:
            raise ValueError("y cannot be None for BernoulliNB")
        x_arr = np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            x_arr = (x_arr > self.binarize).astype(np.float64)

        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.feature_log_prob_neg_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            n_cls = len(x_cls)
            if self.class_prior is not None:
                self.class_prior_[idx] = self.class_prior[idx]
            elif self.fit_prior:
                self.class_prior_[idx] = n_cls / float(n_samples)
            else:
                self.class_prior_[idx] = 1.0 / float(n_classes)

            smoothed_fc = np.sum(x_cls, axis=0) + self.alpha
            smoothed_cc = n_cls + 2.0 * self.alpha
            p = smoothed_fc / smoothed_cc
            self.feature_log_prob_[idx] = np.log(p)
            self.feature_log_prob_neg_[idx] = np.log(1.0 - p)

        self.class_prior_ = self.class_prior_ / np.sum(self.class_prior_)
        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            x_arr = (x_arr > self.binarize).astype(np.float64)
        log_prior = np.log(np.maximum(self.class_prior_, 1e-12))
        neg_x = 1.0 - x_arr
        return (
            np.dot(x_arr, self.feature_log_prob_.T)
            + np.dot(neg_x, self.feature_log_prob_neg_.T)
            + log_prior
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(shifted)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


class ComplementNB(ChokkhuModel):
    def __init__(
        self,
        alpha: float = 1.0,
        force_alpha: bool = True,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.feature_weights_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ComplementNB:
        if y is None:
            raise ValueError("y cannot be None for ComplementNB")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        total_feature_counts = np.sum(x_arr, axis=0)
        self.feature_weights_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            comp_counts = (total_feature_counts - np.sum(x_cls, axis=0)) + self.alpha
            p_comp = comp_counts / np.sum(comp_counts)
            w_c = np.log(p_comp)
            self.feature_weights_[idx] = w_c / np.sum(np.abs(w_c)) if self.norm else w_c

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        scores = np.dot(x_arr, self.feature_weights_.T)
        return self.classes_[np.argmin(scores, axis=1)]


class CategoricalNB(ChokkhuModel):
    def __init__(
        self,
        alpha: float = 1.0,
        force_alpha: bool = True,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        min_categories: Optional[Union[int, list]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.class_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.category_count_: List[List[np.ndarray]] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> CategoricalNB:
        if y is None:
            raise ValueError("y cannot be None for CategoricalNB")
        x_arr = np.asarray(X, dtype=np.int64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            self.class_prior_[idx] = (
                len(x_cls) / float(n_samples) if self.fit_prior else 1.0 / n_classes
            )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.int64)
        n_samples = x_arr.shape[0]
        return np.full(
            n_samples, self.classes_[np.argmax(self.class_prior_)], dtype=np.int64
        )


class LDAClassifier(ChokkhuModel):
    def __init__(
        self,
        solver: str = "svd",
        shrinkage: Any = None,
        priors: Optional[np.ndarray] = None,
        n_components: Optional[int] = None,
        store_covariance: bool = False,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.priors_: np.ndarray = np.array([], dtype=np.float64)
        self.means_: np.ndarray = np.array([], dtype=np.float64)
        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> LDAClassifier:
        if y is None:
            raise ValueError("y cannot be None for LinearDiscriminantAnalysis")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)

        pooled_cov = np.zeros((n_features, n_features), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            self.priors_[idx] = (
                self.priors[idx]
                if self.priors is not None
                else len(x_cls) / float(n_samples)
            )
            mean_cls = np.mean(x_cls, axis=0)
            self.means_[idx] = mean_cls
            diff = x_cls - mean_cls
            pooled_cov += np.dot(diff.T, diff)

        pooled_cov /= max(float(n_samples - n_classes), 1.0)
        pooled_cov += 1e-6 * np.eye(n_features)

        try:
            cov_inv = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(pooled_cov)

        self.coef_ = np.dot(self.means_, cov_inv)
        self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1) + np.log(
            np.maximum(self.priors_, 1e-12)
        )
        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        return np.dot(x_arr, self.coef_.T) + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class QDAClassifier(ChokkhuModel):
    def __init__(
        self,
        priors: Optional[np.ndarray] = None,
        reg_param: float = 1e-4,
        store_covariance: bool = False,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.priors = priors
        self.reg_param = float(reg_param)
        self.store_covariance = store_covariance
        self.tol = tol
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.priors_: np.ndarray = np.array([], dtype=np.float64)
        self.means_: np.ndarray = np.array([], dtype=np.float64)
        self.covariances_: list[np.ndarray] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> QDAClassifier:
        if y is None:
            raise ValueError("y cannot be None for QuadraticDiscriminantAnalysis")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = x_arr.shape

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.covariances_ = []

        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            n_cls = len(x_cls)
            self.priors_[idx] = (
                self.priors[idx]
                if self.priors is not None
                else n_cls / float(n_samples)
            )
            mean_cls = np.mean(x_cls, axis=0)
            self.means_[idx] = mean_cls
            diff = x_cls - mean_cls
            cov_cls = np.dot(diff.T, diff) / max(float(n_cls - 1), 1.0)
            cov_cls += self.reg_param * np.eye(n_features)
            self.covariances_.append(cov_cls)

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_classes = x_arr.shape[0], len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            mean, cov = self.means_[idx], self.covariances_[idx]
            sign, logdet = np.linalg.slogdet(cov)
            cov_inv = np.linalg.pinv(cov)
            diff = x_arr - mean
            quad = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            scores[:, idx] = (
                -0.5 * logdet
                - 0.5 * quad
                + np.log(np.maximum(self.priors_[idx], 1e-12))
            )

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


LinearDiscriminantAnalysis = LDAClassifier
QuadraticDiscriminantAnalysis = QDAClassifier
