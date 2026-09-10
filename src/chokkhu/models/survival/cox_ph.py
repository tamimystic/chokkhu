"""Cox Proportional Hazards Regression (Semi-Parametric Survival Analysis).

Pure NumPy implementation of Cox PH model with:
- Breslow and Efron tie approximation methods
- Regularized Newton-Raphson maximum partial likelihood optimization
- Baseline hazard, baseline cumulative hazard, and baseline survival estimation
- Hazard ratios, standard errors, z-scores, Wald test p-values, and 95% confidence intervals
- Individual survival curve prediction S(t|X) = S_0(t)^exp(X beta)
"""

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np


class CoxPHRegression:
    r"""Cox Proportional Hazards Semi-Parametric Regression Model.

    Models hazard as:

    .. math::
        \lambda(t | X) = \lambda_0(t) \exp(X \beta)

    Parameters
    ----------
    alpha : float, default=0.01
        L2 ridge penalty parameter to guarantee numerical stability.
    tie_method : str, default="efron"
        Method for handling tied event times ("efron" or "breslow").
    max_iter : int, default=100
        Maximum Newton-Raphson iterations.
    tol : float, default=1e-6
        Convergence tolerance on parameter step norm.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        tie_method: str = "efron",
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        if tie_method not in ("efron", "breslow"):
            raise ValueError(
                f"tie_method must be 'efron' or 'breslow', got '{tie_method}'"
            )

        self.alpha = alpha
        self.tie_method = tie_method
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.standard_errors_: Optional[np.ndarray] = None
        self.hazard_ratios_: Optional[np.ndarray] = None
        self.hazard_ratios_ci_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.baseline_timeline_: Optional[np.ndarray] = None
        self.baseline_hazard_: Optional[np.ndarray] = None
        self.baseline_cumulative_hazard_: Optional[np.ndarray] = None
        self.baseline_survival_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> "CoxPHRegression":
        """Fit Cox PH model to covariate matrix X, durations T, and event indicators E."""
        X_arr = np.asarray(X, dtype=np.float64)
        T_arr = np.asarray(durations, dtype=np.float64).ravel()
        if events is None:
            E_arr = np.ones_like(T_arr, dtype=np.int32)
        else:
            E_arr = np.asarray(events, dtype=np.int32).ravel()

        if len(X_arr) != len(T_arr) or len(T_arr) != len(E_arr):
            raise ValueError(
                "X, durations, and events must have the same number of samples."
            )

        n_samples, n_features = X_arr.shape
        if n_samples == 0:
            raise ValueError("Input data cannot be empty.")
        if np.sum(E_arr == 1) == 0:
            raise ValueError(
                "At least one observed event (event=1) is required to fit Cox PH."
            )

        # Center X for numeric conditioning
        self._x_mean = np.mean(X_arr, axis=0)
        X_c = X_arr - self._x_mean

        # Group by distinct event times
        unique_event_times = np.sort(np.unique(T_arr[E_arr == 1]))

        # Initialize beta = zeros
        beta = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.max_iter):
            xb = np.dot(X_c, beta)
            # Clip for numerical stability
            xb = np.clip(xb, -30.0, 30.0)
            exp_xb = np.exp(xb)

            grad = np.zeros(n_features, dtype=np.float64)
            hess = np.zeros((n_features, n_features), dtype=np.float64)
            ll = 0.0

            for t in unique_event_times:
                risk_mask = T_arr >= t
                event_mask = (T_arr == t) & (E_arr == 1)

                d_m = float(np.sum(event_mask))
                if d_m == 0:
                    continue

                X_risk = X_c[risk_mask]
                exp_risk = exp_xb[risk_mask]
                X_event = X_c[event_mask]
                exp_event = exp_xb[event_mask]

                S0 = float(np.sum(exp_risk))
                S1 = np.dot(exp_risk, X_risk)  # (p,)
                S2 = np.dot(X_risk.T, X_risk * exp_risk[:, None])  # (p, p)

                sum_X_event = np.sum(X_event, axis=0)

                if self.tie_method == "breslow" or d_m == 1:
                    ll += float(
                        np.dot(sum_X_event, beta) - d_m * math.log(max(S0, 1e-12))
                    )
                    grad += sum_X_event - d_m * (S1 / S0)
                    hess -= d_m * ((S2 / S0) - np.outer(S1, S1) / (S0 * S0))
                else:  # Efron
                    B0 = float(np.sum(exp_event))
                    B1 = np.dot(exp_event, X_event)
                    B2 = np.dot(X_event.T, X_event * exp_event[:, None])

                    for r in range(int(d_m)):
                        frac = float(r) / d_m
                        S0_r = max(S0 - frac * B0, 1e-12)
                        S1_r = S1 - frac * B1
                        S2_r = S2 - frac * B2

                        ll += float(np.dot(sum_X_event, beta) / d_m - math.log(S0_r))
                        grad += (sum_X_event / d_m) - (S1_r / S0_r)
                        hess -= (S2_r / S0_r) - np.outer(S1_r, S1_r) / (S0_r * S0_r)

            # Add L2 penalty
            grad -= self.alpha * beta
            hess -= self.alpha * np.eye(n_features)
            ll -= 0.5 * self.alpha * float(np.dot(beta, beta))

            # Newton step
            try:
                delta = np.linalg.solve(-hess, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(-hess) @ grad

            beta += delta

            if np.max(np.abs(delta)) < self.tol:
                break

        self.coef_ = beta
        self.log_likelihood_ = ll

        # Standard errors from inverse Fisher information matrix
        try:
            cov_mat = np.linalg.inv(-hess)
        except np.linalg.LinAlgError:
            cov_mat = np.linalg.pinv(-hess)

        se = np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
        z_scores = beta / se
        p_vals = np.array([float(math.erfc(abs(z) / math.sqrt(2.0))) for z in z_scores])

        hr = np.exp(beta)
        hr_lower = np.exp(beta - 1.959963984540054 * se)
        hr_upper = np.exp(beta + 1.959963984540054 * se)

        self.standard_errors_ = se
        self.hazard_ratios_ = hr
        self.hazard_ratios_ci_ = np.column_stack([hr_lower, hr_upper])
        self.p_values_ = p_vals

        # Baseline hazard & survival calculation (Breslow estimator)
        xb_final = np.dot(X_c, beta)
        exp_xb_final = np.exp(np.clip(xb_final, -30.0, 30.0))

        timeline_list = (
            [0.0] + list(unique_event_times)
            if unique_event_times[0] > 0.0
            else list(unique_event_times)
        )
        timeline = np.array(timeline_list, dtype=np.float64)

        base_hazard: np.ndarray = np.zeros(len(timeline), dtype=np.float64)
        base_cum_hazard: np.ndarray = np.zeros(len(timeline), dtype=np.float64)
        cum_h = 0.0

        for i, t in enumerate(timeline):
            if t == 0.0 and (
                len(unique_event_times) == 0 or unique_event_times[0] > 0.0
            ):
                continue
            d_m = float(np.sum((T_arr == t) & (E_arr == 1)))
            denom = float(np.sum(exp_xb_final[T_arr >= t]))
            h_m = (d_m / denom) if denom > 0 else 0.0
            base_hazard[i] = h_m
            cum_h += h_m
            base_cum_hazard[i] = cum_h

        self.baseline_timeline_ = timeline
        self.baseline_hazard_ = base_hazard
        self.baseline_cumulative_hazard_ = base_cum_hazard
        self.baseline_survival_ = np.exp(-base_cum_hazard)

        self.is_fitted = True
        return self

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        """Predict individual partial hazard ratio exp(X beta)."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError(
                "Model must be fitted before calling predict_partial_hazard."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        X_c = X_arr - self._x_mean
        return np.exp(np.clip(np.dot(X_c, self.coef_), -30.0, 30.0))

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict linear risk score X beta."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict_risk.")
        X_arr = np.asarray(X, dtype=np.float64)
        X_c = X_arr - self._x_mean
        return np.dot(X_c, self.coef_)

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[Union[float, np.ndarray, List[float]]] = None,
    ) -> np.ndarray:
        """Predict survival function S(t | X) = S_0(t)^exp(X beta) for given individuals."""
        if (
            not self.is_fitted
            or self.baseline_survival_ is None
            or self.baseline_timeline_ is None
        ):
            raise RuntimeError(
                "Model must be fitted before calling predict_survival_function."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        hazard_mult = self.predict_partial_hazard(X_arr)  # (N,)

        if times is None:
            base_s = self.baseline_survival_  # (T,)
            # Return (N, T) matrix
            return np.power(base_s[None, :], hazard_mult[:, None])
        else:
            t_query = np.asarray(times, dtype=np.float64)
            scalar_time = t_query.ndim == 0
            t_query = np.atleast_1d(t_query)

            indices = (
                np.searchsorted(self.baseline_timeline_, t_query, side="right") - 1
            )
            indices = np.clip(indices, 0, len(self.baseline_survival_) - 1)
            base_s = self.baseline_survival_[indices]
            base_s[t_query < 0.0] = 1.0

            # (N, len(t_query))
            res = np.power(base_s[None, :], hazard_mult[:, None])
            return res[:, 0] if scalar_time and res.shape[1] == 1 else res

    def summary(self) -> Dict[str, Any]:
        """Return structured summary table of model estimates."""
        if (
            not self.is_fitted
            or self.coef_ is None
            or self.hazard_ratios_ is None
            or self.standard_errors_ is None
            or self.p_values_ is None
            or self.hazard_ratios_ci_ is None
        ):
            raise RuntimeError("Model must be fitted before calling summary.")
        return {
            "coef": self.coef_,
            "hazard_ratio": self.hazard_ratios_,
            "std_err": self.standard_errors_,
            "p_value": self.p_values_,
            "hazard_ratio_95_ci_lower": self.hazard_ratios_ci_[:, 0],
            "hazard_ratio_95_ci_upper": self.hazard_ratios_ci_[:, 1],
            "log_likelihood": self.log_likelihood_,
        }
