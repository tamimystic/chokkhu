"""Potential Outcomes Framework, Propensity Scores & Weighting Estimators.

Pure NumPy implementations of:
- PropensityModel: Logistic regression propensity score estimator e(X) = P(T=1|X)
- PropensityScoreMatching (PSM): Nearest-neighbor caliper matching for ATE and ATT
- InverseProbabilityWeighting (IPW): Horvitz-Thompson and Hajek stabilized estimators
"""

from typing import List, Optional, Tuple
import numpy as np


class PropensityModel:
    r"""Logistic Regression Propensity Score Estimator :math:`e(X) = P(T=1|X)`.

    Parameters
    ----------
    lr : float, default=0.05
        Learning rate for gradient descent.
    n_epochs : int, default=100
        Number of optimization epochs.
    reg : float, default=0.01
        L2 regularization weight.
    seed : Optional[int], default=42
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        lr: float = 0.05,
        n_epochs: int = 100,
        reg: float = 0.01,
        seed: Optional[int] = 42,
    ) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.seed = seed

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray) -> "PropensityModel":
        """Fit propensity model predicting treatment assignment T from covariates X."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.float32).ravel()

        if len(X_arr) != len(T_arr):
            raise ValueError("X and T must have identical number of samples.")

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.seed)

        self.weights = (rng.randn(n_features) * 0.01).astype(np.float32)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            logits = np.dot(X_arr, self.weights) + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

            errors = probs - T_arr  # (N,)
            grad_w = np.dot(X_arr.T, errors) / n_samples + self.reg * self.weights
            grad_b = float(np.mean(errors))

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        r"""Predict propensity scores :math:`e(X) \in [\epsilon, 1-\epsilon]`."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X_arr = np.asarray(X, dtype=np.float32)
        logits = np.dot(X_arr, self.weights) + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return np.clip(probs, eps, 1.0 - eps)


class PropensityScoreMatching:
    r"""Propensity Score Matching (PSM) for ATE & ATT Estimation.

        Matches treated units to control units based on propensity score logits:

        .. math::
                    ext{logit}(e(X)) = \ln
    rac{e(X)}{1 - e(X)}

        Parameters
        ----------
        caliper : Optional[float], default=0.2
            Maximum allowed distance in standard deviations of the logit propensity score.
        replace : bool, default=False
            Whether to perform matching with replacement.
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        caliper: Optional[float] = 0.2,
        replace: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        self.caliper = caliper
        self.replace = replace
        self.seed = seed

        self.propensity_model = PropensityModel(seed=seed)
        self.matched_pairs: List[Tuple[int, int]] = []
        self.ate_: Optional[float] = None
        self.att_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "PropensityScoreMatching":
        """Fit propensity scores, match treated and control samples, and estimate treatment effects."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.int32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()

        if len(X_arr) != len(T_arr) or len(X_arr) != len(Y_arr):
            raise ValueError("X, T, and Y must have matching dimensions.")

        self.propensity_model.fit(X_arr, T_arr)
        pscores = self.propensity_model.predict_proba(X_arr)
        logits = np.log(pscores / (1.0 - pscores))

        treated_idx = np.where(T_arr == 1)[0]
        control_idx = np.where(T_arr == 0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            raise ValueError(
                "Both treated (T=1) and control (T=0) samples are required."
            )

        logit_std = float(np.std(logits))
        max_dist = (
            (self.caliper * logit_std) if self.caliper is not None else float("inf")
        )

        used_controls = set()
        matched_treated: List[int] = []
        matched_control: List[int] = []

        rng = np.random.RandomState(self.seed)
        shuffled_treated = rng.permutation(treated_idx)

        for t_idx in shuffled_treated:
            t_logit = logits[t_idx]
            available_controls = [
                c for c in control_idx if self.replace or c not in used_controls
            ]
            if not available_controls:
                break

            control_logits = logits[available_controls]
            dists = np.abs(control_logits - t_logit)
            min_pos = int(np.argmin(dists))
            best_dist = dists[min_pos]
            best_c_idx = available_controls[min_pos]

            if best_dist <= max_dist:
                matched_treated.append(t_idx)
                matched_control.append(best_c_idx)
                if not self.replace:
                    used_controls.add(best_c_idx)

        self.matched_pairs = list(zip(matched_treated, matched_control))

        if not self.matched_pairs:
            self.att_ = 0.0
            self.ate_ = 0.0
        else:
            diffs = Y_arr[matched_treated] - Y_arr[matched_control]
            self.att_ = float(np.mean(diffs))
            self.ate_ = self.att_

        self.is_fitted = True
        return self

    def estimate_att(self) -> float:
        """Return estimated Average Treatment Effect on the Treated (ATT)."""
        if not self.is_fitted or self.att_ is None:
            raise RuntimeError("Model must be fitted before calling estimate_att.")
        return self.att_

    def estimate_ate(self) -> float:
        """Return estimated Average Treatment Effect (ATE)."""
        if not self.is_fitted or self.ate_ is None:
            raise RuntimeError("Model must be fitted before calling estimate_ate.")
        return self.ate_


class InverseProbabilityWeighting:
    r"""Inverse Probability Weighting (IPW) Estimator for ATE.

        Implements both standard Horvitz-Thompson and Hajek stabilized estimators:

        .. math::
            \hat{	au}_{	ext{HT}} =
    rac{1}{N} \sum_{i=1}^N \left[
    rac{T_i Y_i}{e(X_i)} -
    rac{(1 - T_i)Y_i}{1 - e(X_i)}
    ight]
            \hat{	au}_{	ext{Hajek}} =
    rac{\sum_i
    rac{T_i Y_i}{e(X_i)}}{\sum_i
    rac{T_i}{e(X_i)}} -
    rac{\sum_i
    rac{(1-T_i)Y_i}{1-e(X_i)}}{\sum_i
    rac{1-T_i}{1-e(X_i)}}

        Parameters
        ----------
        stabilized : bool, default=True
            Whether to use Hajek stabilized weights (sum to 1 normalization).
        clip_range : Tuple[float, float], default=(0.01, 0.99)
            Lower and upper bounds for clipping propensity scores.
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        stabilized: bool = True,
        clip_range: Tuple[float, float] = (0.01, 0.99),
        seed: Optional[int] = 42,
    ) -> None:
        self.stabilized = stabilized
        self.clip_range = clip_range
        self.seed = seed

        self.propensity_model = PropensityModel(seed=seed)
        self.ate_: Optional[float] = None
        self.std_err_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "InverseProbabilityWeighting":
        """Fit propensity scores and calculate IPW ATE estimate."""
        X_arr = np.asarray(X, dtype=np.float32)
        T_arr = np.asarray(T, dtype=np.float32).ravel()
        Y_arr = np.asarray(Y, dtype=np.float32).ravel()
        n_samples = len(X_arr)

        self.propensity_model.fit(X_arr, T_arr)
        raw_ps = self.propensity_model.predict_proba(X_arr)
        e_x = np.clip(raw_ps, self.clip_range[0], self.clip_range[1])

        w_t = T_arr / e_x
        w_c = (1.0 - T_arr) / (1.0 - e_x)

        if self.stabilized:
            sum_wt: float = float(np.sum(w_t))
            sum_wc: float = float(np.sum(w_c))
            y1_hat = float(np.sum(w_t * Y_arr) / sum_wt) if sum_wt > 0 else 0.0
            y0_hat = float(np.sum(w_c * Y_arr) / sum_wc) if sum_wc > 0 else 0.0
            self.ate_ = float(y1_hat - y0_hat)
        else:
            diffs = (w_t * Y_arr) - (w_c * Y_arr)
            self.ate_ = float(np.mean(diffs))

        # Influence function variance approximation
        influence = (
            (w_t * Y_arr)
            - (w_c * Y_arr)
            - (self.ate_ if self.ate_ is not None else 0.0)
        )
        self.std_err_ = float(np.std(influence) / np.sqrt(n_samples))

        self.is_fitted = True
        return self

    def estimate_ate(self) -> float:
        """Return estimated Average Treatment Effect (ATE)."""
        if not self.is_fitted or self.ate_ is None:
            raise RuntimeError("Model must be fitted before calling estimate_ate.")
        return self.ate_

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute asymptotic 1-alpha confidence interval for ATE."""
        if not self.is_fitted or self.ate_ is None or self.std_err_ is None:
            raise RuntimeError(
                "Model must be fitted before computing confidence intervals."
            )

        z = 1.96 if abs(alpha - 0.05) < 1e-4 else 2.576
        margin = z * self.std_err_
        return (float(self.ate_ - margin), float(self.ate_ + margin))
