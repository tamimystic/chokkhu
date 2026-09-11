"""Double / Debiased Machine Learning (DML) & R-Learner for Causal Inference.

Formulated from first principles in pure NumPy and SciPy, implementing
Chernozhukov et al. (Econometrica 2018) Double/Debiased Machine Learning with
Neyman-orthogonal scores and K-fold cross-fitting, alongside Nie & Wager (Biometrika 2021)
R-Learner for heterogeneous treatment effect (CATE) estimation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, Optional


class _RidgeRegressor:
    """Internal pure NumPy Ridge Regressor for nuisance function estimation."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "_RidgeRegressor":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        N, D = X_arr.shape

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            w_sqrt = np.sqrt(np.maximum(w, 1e-12))
            X_w = X_arr * w_sqrt[:, None]
            y_w = y_arr * w_sqrt
        else:
            X_w = X_arr
            y_w = y_arr

        # Center for intercept
        x_mean = np.mean(X_w, axis=0)
        y_mean = float(np.mean(y_w))
        X_c = X_w - x_mean
        y_c = y_w - y_mean

        A = np.dot(X_c.T, X_c) + self.alpha * np.eye(D)
        b = np.dot(X_c.T, y_c)
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.coef_) + self.intercept_


class DoubleMLPLR:
    r"""Double/Debiased Machine Learning for Partially Linear Regression (PLR).

    Estimates average treatment effect :math:`\theta_0` in the structural equation:

    .. math::
        Y = D \theta_0 + g(X) + U, \quad \mathbb{E}[U | X, D] = 0
        D = m(X) + V, \quad \mathbb{E}[V | X] = 0

    using Neyman-orthogonal score residualization :math:`\tilde{Y} = Y - \hat{g}(X)`,
    :math:`\tilde{D} = D - \hat{m}(X)` and K-fold sample-splitting cross-fitting.

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-fitting folds :math:`K`.
    alpha : float, default=1.0
        L2 regularization parameter for nuisance function estimators.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, n_folds: int = 5, alpha: float = 1.0, seed: int = 42) -> None:
        self.n_folds = int(n_folds)
        self.alpha = float(alpha)
        self.seed = int(seed)

        self.coef_: float = 0.0
        self.se_: float = 0.0
        self.t_stat_: float = 0.0
        self.p_val_: float = 0.0
        self.ci_lower_: float = 0.0
        self.ci_upper_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> "DoubleMLPLR":
        r"""Fit Double ML with cross-fitting.

        Parameters
        ----------
        X : np.ndarray, shape (N, P)
            Confounder feature matrix.
        y : np.ndarray, shape (N,)
            Observed continuous target outcome.
        d : np.ndarray, shape (N,)
            Treatment assignment variable.

        Returns
        -------
        self : DoubleMLPLR
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        d_arr = np.asarray(d, dtype=np.float64).ravel()

        N, _ = X_arr.shape
        rng = np.random.RandomState(self.seed)

        # Create random K-fold splits
        indices = rng.permutation(N)
        fold_sizes: np.ndarray = np.full(self.n_folds, N // self.n_folds, dtype=int)
        fold_sizes[: N % self.n_folds] += 1
        folds = []
        current = 0
        for fold_size in fold_sizes:
            folds.append(indices[current : current + fold_size])
            current += fold_size

        y_res = np.zeros(N, dtype=np.float64)
        d_res = np.zeros(N, dtype=np.float64)

        # Cross-fitting loop
        for k in range(self.n_folds):
            test_idx = folds[k]
            train_idx = np.setdiff1d(indices, test_idx)

            X_tr, y_tr, d_tr = X_arr[train_idx], y_arr[train_idx], d_arr[train_idx]
            X_te, y_te, d_te = X_arr[test_idx], y_arr[test_idx], d_arr[test_idx]

            # Model g(X) for Y
            model_g = _RidgeRegressor(alpha=self.alpha).fit(X_tr, y_tr)
            y_pred = model_g.predict(X_te)
            y_res[test_idx] = y_te - y_pred

            # Model m(X) for D
            model_m = _RidgeRegressor(alpha=self.alpha).fit(X_tr, d_tr)
            d_pred = model_m.predict(X_te)
            d_res[test_idx] = d_te - d_pred

        # Neyman-orthogonal score estimator: theta_hat = <d_res, y_res> / <d_res, d_res>
        d_res_sq = np.mean(d_res**2)
        if d_res_sq < 1e-12:
            raise ValueError(
                "Treatment variation insufficient after orthogonalization (d_res_sq ~ 0)"
            )

        theta_hat = float(np.mean(d_res * y_res) / d_res_sq)
        self.coef_ = theta_hat

        # Asymptotic variance estimator
        psi = (y_res - d_res * theta_hat) * d_res
        J = d_res_sq
        var_hat = np.mean(psi**2) / (J**2)
        self.se_ = float(np.sqrt(max(1e-12, var_hat / N)))

        self.t_stat_ = float(self.coef_ / self.se_)
        self.p_val_ = float(2.0 * stats.norm.sf(np.abs(self.t_stat_)))
        self.ci_lower_ = float(self.coef_ - 1.96 * self.se_)
        self.ci_upper_ = float(self.coef_ + 1.96 * self.se_)

        return self

    def summary(self) -> Dict[str, float]:
        """Return diagnostic metrics dictionary."""
        return {
            "theta": self.coef_,
            "se": self.se_,
            "t_statistic": self.t_stat_,
            "p_value": self.p_val_,
            "ci_95_lower": self.ci_lower_,
            "ci_95_upper": self.ci_upper_,
        }


class RLearner:
    r"""R-Learner for Heterogeneous Treatment Effect (CATE) Estimation.

    Solves the Robinson residualized loss objective:

    .. math::
        \min_\tau \frac{1}{N} \sum_{i=1}^N \left( (Y_i - \hat{m}(X_i)) - (D_i - \hat{e}(X_i)) \tau(X_i) \right)^2

    to estimate heterogeneous treatment effects :math:`\tau(X) = \mathbb{E}[Y(1) - Y(0) | X]`.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength for base and second-stage models.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.tau_model: Optional[_RidgeRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> "RLearner":
        r"""Fit the R-Learner CATE model.

        Parameters
        ----------
        X : np.ndarray, shape (N, P)
        y : np.ndarray, shape (N,)
        d : np.ndarray, shape (N,)

        Returns
        -------
        self : RLearner
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        d_arr = np.asarray(d, dtype=np.float64).ravel()

        N, _ = X_arr.shape

        # Step 1: Fit outcome model m(X) = E[Y|X] and propensity model e(X) = E[D|X]
        model_m = _RidgeRegressor(alpha=self.alpha).fit(X_arr, y_arr)
        model_e = _RidgeRegressor(alpha=self.alpha).fit(X_arr, d_arr)

        y_res = y_arr - model_m.predict(X_arr)
        d_res = d_arr - model_e.predict(X_arr)

        # Step 2: Minimize weighted residualized loss:
        # L(tau) = sum_i d_res^2 * (y_res / d_res - tau(X_i))^2
        # Target: y_tilde = y_res / d_res (with weights w = d_res^2)
        # Equivalent to regressing y_res on d_res * X
        X_mod = X_arr * d_res[:, None]
        y_mod = y_res

        self.tau_model = _RidgeRegressor(alpha=self.alpha).fit(X_mod, y_mod)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        r"""Predict Conditional Average Treatment Effect (CATE) :math:`\tau(X)`.

        Parameters
        ----------
        X : np.ndarray, shape (N, P)

        Returns
        -------
        cate : np.ndarray, shape (N,)
        """
        if self.tau_model is None:
            raise ValueError("Model must be fitted before predicting CATE")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.tau_model.coef_) + self.tau_model.intercept_
