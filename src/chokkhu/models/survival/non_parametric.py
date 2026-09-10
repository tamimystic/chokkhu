"""Non-parametric Survival Analysis: Kaplan-Meier & Nelson-Aalen Estimators.

Pure NumPy implementations of:
- KaplanMeierFitter: Survival function S(t) = P(T > t) with Greenwood confidence intervals
- NelsonAalenFitter: Cumulative hazard H(t) with Aalen variance
- logrank_test: Non-parametric hypothesis testing for comparing survival curves
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class KaplanMeierFitter:
    r"""Kaplan-Meier Non-Parametric Survival Curve Estimator.

        Estimates the survival function:

        .. math::
            \hat{S}(t) = \prod_{t_i \le t} \left(1 -
    rac{d_i}{n_i}
    ight)

        with Greenwood's variance formula:

        .. math::
            \widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals (default 95% CI).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.timeline: Optional[np.ndarray] = None
        self.survival_probabilities: Optional[np.ndarray] = None
        self.standard_errors: Optional[np.ndarray] = None
        self.confidence_interval_: Optional[np.ndarray] = None
        self.median_survival_time_: Optional[float] = None
        self.event_table_: Optional[Dict[str, np.ndarray]] = None
        self.is_fitted: bool = False

    def fit(
        self,
        durations: np.ndarray,
        event_observed: Optional[np.ndarray] = None,
    ) -> "KaplanMeierFitter":
        """Fit Kaplan-Meier survival curve.

        Parameters
        ----------
        durations : array-like of shape (n_samples,)
            Observed time-to-event or censoring durations.
        event_observed : array-like of shape (n_samples,), optional
            Binary indicator (1 for observed event/death, 0 for censored).
            If None, all durations are assumed to be observed events.
        """
        dur = np.asarray(durations, dtype=np.float64).ravel()
        if event_observed is None:
            evt = np.ones_like(dur, dtype=np.int32)
        else:
            evt = np.asarray(event_observed, dtype=np.int32).ravel()

        if len(dur) != len(evt):
            raise ValueError("durations and event_observed must have identical length.")
        if np.any(dur < 0):
            raise ValueError("Durations must be non-negative.")

        n_samples = len(dur)
        if n_samples == 0:
            raise ValueError("Input durations cannot be empty.")

        order = np.lexsort((-evt, dur))
        dur_sorted = dur[order]
        evt_sorted = evt[order]

        unique_times = np.unique(dur_sorted)
        if unique_times[0] > 0.0:
            timeline_list = [0.0] + list(unique_times)
        else:
            timeline_list = list(unique_times)

        timeline = np.array(timeline_list, dtype=np.float64)
        n_times = len(timeline)

        at_risk: np.ndarray = np.zeros(n_times, dtype=np.float64)
        events: np.ndarray = np.zeros(n_times, dtype=np.float64)
        censored: np.ndarray = np.zeros(n_times, dtype=np.float64)

        current_n = float(n_samples)
        for i, t in enumerate(timeline):
            if t == 0.0 and (len(unique_times) == 0 or unique_times[0] > 0.0):
                at_risk[i] = current_n
                events[i] = 0.0
                censored[i] = 0.0
                continue

            mask = dur_sorted == t
            d_i = float(np.sum(evt_sorted[mask] == 1))
            c_i = float(np.sum(evt_sorted[mask] == 0))

            at_risk[i] = current_n
            events[i] = d_i
            censored[i] = c_i

            current_n -= d_i + c_i

        surv_prob: np.ndarray = np.ones(n_times, dtype=np.float64)
        greenwood_sum: np.ndarray = np.zeros(n_times, dtype=np.float64)
        cum_sum = 0.0
        current_s = 1.0

        for i in range(n_times):
            n_i = float(at_risk[i])
            d_i = float(events[i])
            if n_i > 0 and d_i > 0:
                current_s *= 1.0 - (d_i / n_i)
                if n_i > d_i:
                    cum_sum += d_i / (n_i * (n_i - d_i))
            surv_prob[i] = current_s
            greenwood_sum[i] = cum_sum

        se = surv_prob * np.sqrt(greenwood_sum)
        z_crit = 1.959963984540054
        if self.alpha != 0.05:
            z_crit = float(np.sqrt(2.0) * _erfinv(1.0 - self.alpha))

        ci_lower: np.ndarray = np.zeros(n_times, dtype=np.float64)
        ci_upper: np.ndarray = np.zeros(n_times, dtype=np.float64)

        for i in range(n_times):
            s = surv_prob[i]
            if s <= 0.0:
                ci_lower[i] = 0.0
                ci_upper[i] = 0.0
            elif s >= 1.0:
                ci_lower[i] = 1.0
                ci_upper[i] = 1.0
            else:
                log_s = math.log(s)
                sigma = float(np.sqrt(greenwood_sum[i]))
                if sigma > 0 and abs(log_s) > 1e-9:
                    theta = (z_crit * sigma) / abs(log_s)
                    ci_lower[i] = max(0.0, float(s ** math.exp(theta)))
                    ci_upper[i] = min(1.0, float(s ** math.exp(-theta)))
                else:
                    ci_lower[i] = max(0.0, float(s - z_crit * se[i]))
                    ci_upper[i] = min(1.0, float(s + z_crit * se[i]))

        median_time: Optional[float] = None
        for i in range(n_times):
            if surv_prob[i] <= 0.5:
                median_time = float(timeline[i])
                break

        self.timeline = timeline
        self.survival_probabilities = surv_prob
        self.standard_errors = se
        self.confidence_interval_ = np.column_stack([ci_lower, ci_upper])
        self.median_survival_time_ = median_time
        self.event_table_ = {
            "timeline": timeline,
            "at_risk": at_risk,
            "events": events,
            "censored": censored,
            "survival_probability": surv_prob,
        }
        self.is_fitted = True
        return self

    def predict(self, times: Union[float, np.ndarray, List[float]]) -> np.ndarray:
        """Predict survival probability S(t) at arbitrary queried time points."""
        if (
            not self.is_fitted
            or self.timeline is None
            or self.survival_probabilities is None
        ):
            raise RuntimeError("Model must be fitted before calling predict.")

        t_query = np.asarray(times, dtype=np.float64)
        scalar_input = t_query.ndim == 0
        t_query = np.atleast_1d(t_query)

        indices = np.searchsorted(self.timeline, t_query, side="right") - 1
        indices = np.clip(indices, 0, len(self.survival_probabilities) - 1)
        preds = self.survival_probabilities[indices]
        preds[t_query < 0.0] = 1.0

        return preds[0] if scalar_input else preds


class NelsonAalenFitter:
    r"""Nelson-Aalen Non-Parametric Cumulative Hazard Estimator.

    Estimates cumulative hazard:

    .. math::
        \hat{H}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for confidence intervals.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.timeline: Optional[np.ndarray] = None
        self.cumulative_hazard_: Optional[np.ndarray] = None
        self.standard_errors: Optional[np.ndarray] = None
        self.confidence_interval_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        durations: np.ndarray,
        event_observed: Optional[np.ndarray] = None,
    ) -> "NelsonAalenFitter":
        """Fit Nelson-Aalen cumulative hazard estimator."""
        dur = np.asarray(durations, dtype=np.float64).ravel()
        if event_observed is None:
            evt = np.ones_like(dur, dtype=np.int32)
        else:
            evt = np.asarray(event_observed, dtype=np.int32).ravel()

        if len(dur) != len(evt):
            raise ValueError("durations and event_observed must have identical length.")

        n_samples = len(dur)
        if n_samples == 0:
            raise ValueError("Input durations cannot be empty.")

        order = np.lexsort((-evt, dur))
        dur_sorted = dur[order]
        evt_sorted = evt[order]

        unique_times = np.unique(dur_sorted)
        timeline_list = (
            [0.0] + list(unique_times) if unique_times[0] > 0.0 else list(unique_times)
        )
        timeline = np.array(timeline_list, dtype=np.float64)
        n_times = len(timeline)

        cum_hazard: np.ndarray = np.zeros(n_times, dtype=np.float64)
        var_hazard: np.ndarray = np.zeros(n_times, dtype=np.float64)

        current_n = float(n_samples)
        cum_h = 0.0
        cum_var = 0.0

        for i, t in enumerate(timeline):
            if t == 0.0 and (len(unique_times) == 0 or unique_times[0] > 0.0):
                continue

            mask = dur_sorted == t
            d_i = float(np.sum(evt_sorted[mask] == 1))
            c_i = float(np.sum(evt_sorted[mask] == 0))

            if current_n > 0 and d_i > 0:
                cum_h += d_i / current_n
                cum_var += d_i / (current_n * current_n)

            cum_hazard[i] = cum_h
            var_hazard[i] = cum_var
            current_n -= d_i + c_i

        se = np.sqrt(var_hazard)
        z_crit = 1.959963984540054
        ci_lower = np.maximum(0.0, cum_hazard - z_crit * se)
        ci_upper = cum_hazard + z_crit * se

        self.timeline = timeline
        self.cumulative_hazard_ = cum_hazard
        self.standard_errors = se
        self.confidence_interval_ = np.column_stack([ci_lower, ci_upper])
        self.is_fitted = True
        return self

    def predict(self, times: Union[float, np.ndarray, List[float]]) -> np.ndarray:
        """Predict cumulative hazard H(t) at given time points."""
        if (
            not self.is_fitted
            or self.timeline is None
            or self.cumulative_hazard_ is None
        ):
            raise RuntimeError("Model must be fitted before calling predict.")

        t_query = np.asarray(times, dtype=np.float64)
        scalar_input = t_query.ndim == 0
        t_query = np.atleast_1d(t_query)

        indices = np.searchsorted(self.timeline, t_query, side="right") - 1
        indices = np.clip(indices, 0, len(self.cumulative_hazard_) - 1)
        preds = self.cumulative_hazard_[indices]
        preds[t_query < 0.0] = 0.0

        return preds[0] if scalar_input else preds


def logrank_test(
    durations_A: np.ndarray,
    events_A: np.ndarray,
    durations_B: np.ndarray,
    events_B: np.ndarray,
) -> Tuple[float, float]:
    r"""Two-Sample Log-Rank Hypothesis Test for Survival Distributions.

    Tests :math:`H_0: S_A(t) = S_B(t)` against :math:`H_1: S_A(t) \neq S_B(t)`.

    Parameters
    ----------
    durations_A : array-like
        Observed times for group A.
    events_A : array-like
        Binary event indicators for group A.
    durations_B : array-like
        Observed times for group B.
    events_B : array-like
        Binary event indicators for group B.

    Returns
    -------
    test_statistic : float
        Chi-square test statistic with 1 degree of freedom.
    p_value : float
        Two-sided asymptotic p-value.
    """
    t_a = np.asarray(durations_A, dtype=np.float64).ravel()
    e_a = np.asarray(events_A, dtype=np.int32).ravel()
    t_b = np.asarray(durations_B, dtype=np.float64).ravel()
    e_b = np.asarray(events_B, dtype=np.int32).ravel()

    all_event_times = np.unique(np.concatenate([t_a[e_a == 1], t_b[e_b == 1]]))
    if len(all_event_times) == 0:
        return 0.0, 1.0

    total_o = 0.0
    total_e = 0.0
    total_v = 0.0

    for t in all_event_times:
        n_a = float(np.sum(t_a >= t))
        n_b = float(np.sum(t_b >= t))
        n_total = n_a + n_b

        d_a = float(np.sum((t_a == t) & (e_a == 1)))
        d_b = float(np.sum((t_b == t) & (e_b == 1)))
        d_total = d_a + d_b

        if n_total <= 1 or d_total == 0:
            continue

        e_val = n_a * (d_total / n_total)
        v_val = (n_a * n_b * d_total * (n_total - d_total)) / (
            n_total * n_total * (n_total - 1.0)
        )

        total_o += d_a
        total_e += e_val
        total_v += v_val

    if total_v <= 1e-12:
        return 0.0, 1.0

    z = (total_o - total_e) / math.sqrt(total_v)
    chi2_stat = float(z * z)
    p_val = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return chi2_stat, p_val


def _erfinv(y: float) -> float:
    """Approximation of inverse error function for CI critical values."""
    a = 0.147
    y = float(np.clip(y, -0.999999, 0.999999))
    log_term = math.log(1.0 - y * y)
    term1 = 2.0 / (math.pi * a) + log_term / 2.0
    inner = term1 * term1 - (log_term / a)
    sign = 1.0 if y >= 0 else -1.0
    return float(sign * math.sqrt(max(0.0, math.sqrt(inner) - term1)))
