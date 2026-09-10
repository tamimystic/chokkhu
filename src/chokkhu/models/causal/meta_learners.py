"""Meta-Learners for Heterogeneous Treatment Effect (HTE / CATE) Estimation.

Pure NumPy implementations of:
- SLearner: Single Model Learner mu(X, T)
- TLearner: Two Independent Models mu_1(X), mu_0(X)
- XLearner: Crossover Imputation Learner (Kunzel et al., 2019)
"""

from typing import Optional
import numpy as np


class _LinearRegressionSolver:
    """Internal OLS / Ridge solver."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearRegressionSolver":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X_arr.shape

        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))

        X_c = X_arr - X_mean
        y_c = y_arr - y_mean

        A = np.dot(X_c.T, X_c) + self.alpha * np.eye(n_features, dtype=np.float32)
        b = np.dot(X_c.T, y_c)
        self.coef = np.linalg.solve(A, b)
        self.intercept = y_mean - float(np.dot(X_mean, self.coef))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Solver must be fitted before predict.")
        X_arr = np.asarray(X, dtype=np.float32)
        return np.dot(X_arr, self.coef) + self.intercept


class SLearner:
    r"""S-Learner (Single Model Meta-Learner).

    Estimates treatment effects using a single joint model :math:`\mu(X, T)`:

    .. math::
        \hat{	au}(X) = \mu(X, 1) - \mu(X, 0)

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization weight for ridge regression solver.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = _LinearRegressionSolver(alpha=alpha)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        """Fit joint model on concatenated features [X, T]."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.float32).reshape(-1, 1)
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        X_joint = np.concatenate([X_arr, T_arr], axis=1)
        self.model.fit(X_joint, Y_arr)
        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        r"""Predict Conditional Average Treatment Effect :math:`\tau(X)`."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples = len(X_arr)

        ones: np.ndarray = np.ones((n_samples, 1), dtype=np.float32)
        zeros: np.ndarray = np.zeros((n_samples, 1), dtype=np.float32)

        mu1 = self.model.predict(np.concatenate([X_arr, ones], axis=1))
        mu0 = self.model.predict(np.concatenate([X_arr, zeros], axis=1))
        return mu1 - mu0

    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate sample Average Treatment Effect."""
        return float(np.mean(self.predict_cate(X)))


class TLearner:
    r"""T-Learner (Two Models Meta-Learner).

    Estimates treatment effects by fitting separate response models for treated and control groups:

    .. math::
        \hat{	au}(X) = \mu_1(X) - \mu_0(X)

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization weight.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model_1 = _LinearRegressionSolver(alpha=alpha)
        self.model_0 = _LinearRegressionSolver(alpha=alpha)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        """Fit independent models on treated (T=1) and control (T=0) cohorts."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.int32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        treated_idx = np.where(T_arr == 1)[0]
        control_idx = np.where(T_arr == 0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            raise ValueError(
                "Both treated (T=1) and control (T=0) samples are required."
            )

        self.model_1.fit(X_arr[treated_idx], Y_arr[treated_idx])
        self.model_0.fit(X_arr[control_idx], Y_arr[control_idx])
        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        r"""Predict CATE :math:`\hat{\tau}(X) = \mu_1(X) - \mu_0(X)`."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model_1.predict(X_arr) - self.model_0.predict(X_arr)

    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate sample Average Treatment Effect."""
        return float(np.mean(self.predict_cate(X)))


class XLearner:
    r"""X-Learner (Crossover Meta-Learner - Künzel et al. 2019).

    Designed for unbalanced treatment assignments by imputing counterfactuals
    and fitting second-stage effect estimators:

    1. Fit :math:`\mu_1(X)` on treated and :math:`\mu_0(X)` on control.
    2. Impute counterfactual treatment effects:
       :math:`D_{1, i} = Y_{1, i} - \mu_0(X_{1, i})` and :math:`D_{0, j} = \mu_1(X_{0, j}) - Y_{0, j}`.
    3. Fit :math:`	au_1(X)` on :math:`(X_1, D_1)` and :math:`	au_0(X)` on :math:`(X_0, D_0)`.
    4. Combine via propensity weighting: :math:`\hat{	au}(X) = e(X) 	au_0(X) + (1 - e(X)) 	au_1(X)`.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization weight.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(self, alpha: float = 1.0, seed: Optional[int] = 42) -> None:
        self.alpha = alpha
        self.seed = seed

        self.mu1 = _LinearRegressionSolver(alpha=alpha)
        self.mu0 = _LinearRegressionSolver(alpha=alpha)
        self.tau1 = _LinearRegressionSolver(alpha=alpha)
        self.tau0 = _LinearRegressionSolver(alpha=alpha)

        self.prop_w: Optional[np.ndarray] = None
        self.prop_b: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        """Fit X-Learner through two-stage counterfactual imputation and weighting."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.float32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        treated_idx = np.where(T_arr == 1.0)[0]
        control_idx = np.where(T_arr == 0.0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            raise ValueError("Both treated and control units are required.")

        X1, Y1 = X_arr[treated_idx], Y_arr[treated_idx]
        X0, Y0 = X_arr[control_idx], Y_arr[control_idx]

        # Stage 1: Fit base models
        self.mu1.fit(X1, Y1)
        self.mu0.fit(X0, Y0)

        # Stage 2: Impute counterfactual effects
        D1 = Y1 - self.mu0.predict(X1)
        D0 = self.mu1.predict(X0) - Y0

        # Fit effect models
        self.tau1.fit(X1, D1)
        self.tau0.fit(X0, D0)

        # Fit propensity model
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X_arr.shape
        self.prop_w = (rng.randn(n_features) * 0.01).astype(np.float32)
        self.prop_b = 0.0

        for _ in range(100):
            logits = np.dot(X_arr, self.prop_w) + self.prop_b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            errors = probs - T_arr
            grad_w = np.dot(X_arr.T, errors) / n_samples + 0.01 * self.prop_w
            grad_b = float(np.mean(errors))
            self.prop_w -= 0.05 * grad_w
            self.prop_b -= 0.05 * grad_b

        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        r"""Predict CATE :math:`\hat{\tau}(X) = e(X) \tau_0(X) + (1 - e(X)) \tau_1(X)`."""
        if not self.is_fitted or self.prop_w is None:
            raise RuntimeError("Model must be fitted before predict_cate.")

        X_arr = np.asarray(X, dtype=np.float32)
        logits = np.dot(X_arr, self.prop_w) + self.prop_b
        e_x = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0))), 0.01, 0.99)

        pred_tau0 = self.tau0.predict(X_arr)
        pred_tau1 = self.tau1.predict(X_arr)

        return e_x * pred_tau0 + (1.0 - e_x) * pred_tau1

    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate sample Average Treatment Effect."""
        return float(np.mean(self.predict_cate(X)))
