"""Doubly Robust Estimation & Conditional Average Treatment Effect (CATE).

Pure NumPy implementation of the Doubly Robust Learner (DR-Learner).
Guarantees unbiased estimation of treatment effects if EITHER the outcome models
OR the propensity score model is correctly specified.
"""

from typing import Optional, Tuple
import numpy as np


class _LinearRegressionSolver:
    """Internal OLS / Ridge regression solver for outcome modeling."""

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


class DoublyRobustLearner:
    r"""Doubly Robust Estimator for ATE and CATE :math:`	au(X)`.

        Combines outcome regression models :math:`\mu_1(X), \mu_0(X)` and propensity scores :math:`e(X)`:

        .. math::
            \Gamma_i = \mu_1(X_i) - \mu_0(X_i) +
    rac{T_i (Y_i - \mu_1(X_i))}{e(X_i)} -
    rac{(1 - T_i)(Y_i - \mu_0(X_i))}{1 - e(X_i)}
            \hat{	au}_{	ext{ATE}} =
    rac{1}{N} \sum_{i=1}^N \Gamma_i

        Parameters
        ----------
        clip_range : Tuple[float, float], default=(0.01, 0.99)
            Propensity score clipping bounds.
        alpha : float, default=1.0
            L2 regularization weight for outcome and CATE regression solvers.
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        clip_range: Tuple[float, float] = (0.01, 0.99),
        alpha: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.clip_range = clip_range
        self.alpha = alpha
        self.seed = seed

        self.mu1_model = _LinearRegressionSolver(alpha=alpha)
        self.mu0_model = _LinearRegressionSolver(alpha=alpha)
        self.cate_model = _LinearRegressionSolver(alpha=alpha)

        # Propensity weights
        self.prop_w: Optional[np.ndarray] = None
        self.prop_b: float = 0.0

        self.pseudo_outcomes: Optional[np.ndarray] = None
        self.ate_: Optional[float] = None
        self.is_fitted: bool = False

    def _fit_propensity(self, X: np.ndarray, T: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X.shape
        self.prop_w = (rng.randn(n_features) * 0.01).astype(np.float32)
        self.prop_b = 0.0

        for _ in range(100):
            logits = np.dot(X, self.prop_w) + self.prop_b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            errors = probs - T
            grad_w = np.dot(X.T, errors) / n_samples + 0.01 * self.prop_w
            grad_b = float(np.mean(errors))
            self.prop_w -= 0.05 * grad_w
            self.prop_b -= 0.05 * grad_b

    def _predict_propensity(self, X: np.ndarray) -> np.ndarray:
        assert self.prop_w is not None
        logits = np.dot(X, self.prop_w) + self.prop_b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return np.clip(probs, self.clip_range[0], self.clip_range[1])

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "DoublyRobustLearner":
        """Fit doubly robust model on covariates X, treatment T, and outcome Y."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.float32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        treated_idx = np.where(T_arr == 1.0)[0]
        control_idx = np.where(T_arr == 0.0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            raise ValueError("Both treated and control units are required.")

        # 1. Fit outcome regressions
        self.mu1_model.fit(X_arr[treated_idx], Y_arr[treated_idx])
        self.mu0_model.fit(X_arr[control_idx], Y_arr[control_idx])

        # 2. Fit propensity scores
        self._fit_propensity(X_arr, T_arr)
        e_x = self._predict_propensity(X_arr)

        # 3. Compute predicted counterfactuals
        mu1_pred = self.mu1_model.predict(X_arr)
        mu0_pred = self.mu0_model.predict(X_arr)

        # 4. Compute Doubly Robust Pseudo-Outcomes Gamma_i
        w_t = T_arr / e_x
        w_c = (1.0 - T_arr) / (1.0 - e_x)
        self.pseudo_outcomes = (
            mu1_pred - mu0_pred + w_t * (Y_arr - mu1_pred) - w_c * (Y_arr - mu0_pred)
        )

        self.ate_ = float(np.mean(self.pseudo_outcomes))

        # 5. Regress pseudo-outcomes onto X to model CATE tau(X)
        self.cate_model.fit(X_arr, self.pseudo_outcomes)

        self.is_fitted = True
        return self

    def estimate_ate(self) -> float:
        """Return Doubly Robust Average Treatment Effect."""
        if not self.is_fitted or self.ate_ is None:
            raise RuntimeError("Model must be fitted before calling estimate_ate.")
        return self.ate_

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        r"""Predict Conditional Average Treatment Effect :math:`\hat{\tau}(X)`."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_cate.")
        return self.cate_model.predict(X)
