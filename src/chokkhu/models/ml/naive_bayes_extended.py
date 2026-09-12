"""Probabilistic & Bayesian Classifiers in Pure NumPy.

References:
- Zhang (2004): "The Optimality of Naive Bayes" (FLAIRS).
- Rennie et al. (2003): "Tackling the Poor Assumptions of Naive Bayes Text Classifiers" (ICML).
- Fisher (1936): "The Use of Multiple Measurements in Taxonomic Problems" (Annals of Eugenics).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from chokkhu.models.base import ChokkhuModel


class GaussianNB(ChokkhuModel):
    """Gaussian Naive Bayes for continuous feature vectors."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        super().__init__()
        self.var_smoothing = float(var_smoothing)

        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.class_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.theta_: np.ndarray = np.array([], dtype=np.float64)  # class means
        self.var_: np.ndarray = np.array([], dtype=np.float64)  # class variances
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
            self.class_prior_[idx] = len(x_cls) / float(n_samples)
            self.theta_[idx] = np.mean(x_cls, axis=0)
            self.var_[idx] = np.var(x_cls, axis=0) + max(epsilon, 1e-9)

        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            mean = self.theta_[idx]
            var = self.var_[idx]
            # Gaussian log-likelihood
            log_lik = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((x_arr - mean) ** 2) / var, axis=1
            )
            log_probs[:, idx] = np.log(self.class_prior_[idx]) + log_lik

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
    """Multinomial Naive Bayes with Laplace/Lidstone smoothing for discrete counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

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
            self.class_prior_[idx] = len(x_cls) / float(n_samples)
            count_per_feat = np.sum(x_cls, axis=0) + self.alpha
            total_count: float = float(np.sum(count_per_feat))
            self.feature_log_prob_[idx] = np.log(count_per_feat / total_count)

        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        log_prior = np.log(self.class_prior_)
        return np.dot(x_arr, self.feature_log_prob_.T) + log_prior

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


class BernoulliNB(ChokkhuModel):
    """Bernoulli Naive Bayes for binary feature vectors."""

    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.binarize = binarize

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
            self.class_prior_[idx] = n_cls / float(n_samples)
            smoothed_fc = np.sum(x_cls, axis=0) + self.alpha
            smoothed_cc = n_cls + 2.0 * self.alpha
            p = smoothed_fc / smoothed_cc
            self.feature_log_prob_[idx] = np.log(p)
            self.feature_log_prob_neg_[idx] = np.log(1.0 - p)

        self.is_fitted = True
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            x_arr = (x_arr > self.binarize).astype(np.float64)

        log_prior = np.log(self.class_prior_)
        neg_x = 1.0 - x_arr
        return (
            np.dot(x_arr, self.feature_log_prob_.T)
            + np.dot(neg_x, self.feature_log_prob_neg_.T)
            + log_prior
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


class ComplementNB(ChokkhuModel):
    """Complement Naive Bayes designed for imbalanced text datasets."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

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
            # Complement counts: sum across all samples NOT in class cls
            comp_counts = (total_feature_counts - np.sum(x_cls, axis=0)) + self.alpha
            p_comp = comp_counts / np.sum(comp_counts)
            w_c = np.log(p_comp)
            # Normalize weights
            self.feature_weights_[idx] = w_c / np.sum(np.abs(w_c))

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        # Class with minimum penalty is selected
        scores = np.dot(x_arr, self.feature_weights_.T)
        return self.classes_[np.argmin(scores, axis=1)]


class LDAClassifier(ChokkhuModel):
    """Fisher Linear Discriminant Analysis Classifier with pooled covariance."""

    def __init__(self, solver: str = "svd") -> None:
        super().__init__()
        self.solver = solver

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

        # Compute class means and pooled covariance matrix
        pooled_cov = np.zeros((n_features, n_features), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            x_cls = x_arr[y_arr == cls]
            self.priors_[idx] = len(x_cls) / float(n_samples)
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

        self.coef_ = np.dot(self.means_, cov_inv)  # [C, D]
        self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1) + np.log(
            self.priors_
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
    """Quadratic Discriminant Analysis with per-class covariance matrices."""

    def __init__(self, reg_param: float = 1e-4) -> None:
        super().__init__()
        self.reg_param = float(reg_param)

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
            self.priors_[idx] = n_cls / float(n_samples)
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
        n_samples = x_arr.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            mean = self.means_[idx]
            cov = self.covariances_[idx]
            sign, logdet = np.linalg.slogdet(cov)
            cov_inv = np.linalg.pinv(cov)

            diff = x_arr - mean
            quad = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            scores[:, idx] = -0.5 * logdet - 0.5 * quad + np.log(self.priors_[idx])

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


LinearDiscriminantAnalysis = LDAClassifier
QuadraticDiscriminantAnalysis = QDAClassifier
