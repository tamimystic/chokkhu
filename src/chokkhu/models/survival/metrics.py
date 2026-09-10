"""Evaluation Metrics for Time-to-Event and Survival Modeling.

Pure NumPy implementations of:
- concordance_index: Harrell's C-index for censored survival data with tie handling
- brier_score_loss: Time-dependent Brier score with Inverse Probability Censoring Weighting (IPCW)
- integrated_brier_score: Integrated Brier score across timeline
"""

from typing import Callable, List, Optional, Union
import numpy as np
from .non_parametric import KaplanMeierFitter


def concordance_index(
    durations: np.ndarray,
    risk_scores: np.ndarray,
    events: Optional[np.ndarray] = None,
) -> float:
    r"""Compute Harrell's Concordance Index (C-index) for Censored Survival Outcomes.

    Evaluates whether individuals with higher predicted risk fail earlier:

    .. math::
        C = \frac{\sum_{i,j} \mathbb{I}(T_i < T_j) \mathbb{I}(\hat{r}_i > \hat{r}_j) E_i +
        0.5 \sum_{i,j} \mathbb{I}(T_i < T_j) \mathbb{I}(\hat{r}_i = \hat{r}_j) E_i}{\sum_{i,j} \mathbb{I}(T_i < T_j) E_i}

    Parameters
    ----------
    durations : array-like of shape (n_samples,)
        Observed durations or time-to-event values.
    risk_scores : array-like of shape (n_samples,)
        Predicted risk scores (higher score implies higher risk / shorter expected time).
    events : array-like of shape (n_samples,), optional
        Binary event indicators (1 for observed event, 0 for censored).
        If None, all observations are assumed uncensored.

    Returns
    -------
    c_index : float
        Concordance index in range [0.0, 1.0]. (1.0 = perfect ranking, 0.5 = random).
    """
    t_arr = np.asarray(durations, dtype=np.float64).ravel()
    r_arr = np.asarray(risk_scores, dtype=np.float64).ravel()
    if events is None:
        e_arr = np.ones_like(t_arr, dtype=np.int32)
    else:
        e_arr = np.asarray(events, dtype=np.int32).ravel()

    if len(t_arr) != len(r_arr) or len(t_arr) != len(e_arr):
        raise ValueError(
            "durations, risk_scores, and events must have the same length."
        )

    n = len(t_arr)
    if n < 2:
        return 1.0

    concordant = 0.0
    total_pairs = 0.0

    for i in range(n):
        if e_arr[i] == 0:
            continue  # i must experience the event to define ordering

        for j in range(n):
            if i == j:
                continue

            if t_arr[i] < t_arr[j]:
                # i failed strictly before j
                total_pairs += 1.0
                if r_arr[i] > r_arr[j]:
                    concordant += 1.0
                elif r_arr[i] == r_arr[j]:
                    concordant += 0.5
            elif t_arr[i] == t_arr[j] and e_arr[j] == 0:
                # i failed at t, j was censored at t (i experienced event before j's true failure)
                total_pairs += 1.0
                if r_arr[i] > r_arr[j]:
                    concordant += 1.0
                elif r_arr[i] == r_arr[j]:
                    concordant += 0.5

    if total_pairs == 0.0:
        return 0.5

    return float(concordant / total_pairs)


def brier_score_loss(
    durations_train: np.ndarray,
    events_train: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    survival_probs: np.ndarray,
    eval_time: float,
) -> float:
    r"""Compute IPCW Time-Dependent Brier Score at evaluation time :math:`t`.

    .. math::
        \text{BS}(t) = \frac{1}{N} \sum_{i=1}^N \left( \frac{(0 - S(t|X_i))^2}{G(T_i)} \mathbb{I}(T_i \le t, E_i=1) +
        \frac{(1 - S(t|X_i))^2}{G(t)} \mathbb{I}(T_i > t) \right)

    where :math:`G(t) = P(C > t)` is the Kaplan-Meier censoring survival distribution.
    """
    t_train = np.asarray(durations_train, dtype=np.float64).ravel()
    e_train = np.asarray(events_train, dtype=np.int32).ravel()
    t_test = np.asarray(durations_test, dtype=np.float64).ravel()
    e_test = np.asarray(events_test, dtype=np.int32).ravel()
    s_probs = np.asarray(survival_probs, dtype=np.float64).ravel()

    n_test = len(t_test)
    if n_test == 0:
        return 0.0

    # Fit KM on censoring distribution (swapping event indicator: 1-e)
    censoring_km = KaplanMeierFitter()
    censoring_km.fit(t_train, 1 - e_train)

    g_eval = float(censoring_km.predict(eval_time))
    g_eval = max(g_eval, 1e-4)

    total_loss = 0.0
    for i in range(n_test):
        ti = t_test[i]
        ei = e_test[i]
        si = s_probs[i]

        if ti <= eval_time and ei == 1:
            g_ti = float(censoring_km.predict(ti))
            g_ti = max(g_ti, 1e-4)
            total_loss += (0.0 - si) ** 2 / g_ti
        elif ti > eval_time:
            total_loss += (1.0 - si) ** 2 / g_eval

    return float(total_loss / n_test)


def integrated_brier_score(
    durations_train: np.ndarray,
    events_train: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    survival_predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X_test: np.ndarray,
    eval_times: Optional[Union[List[float], np.ndarray]] = None,
) -> float:
    """Compute Integrated Brier Score (IBS) integrated across multiple time points."""
    if eval_times is None:
        t_test = np.asarray(durations_test, dtype=np.float64)
        eval_times = np.linspace(np.min(t_test), np.max(t_test), 20)

    times = np.asarray(eval_times, dtype=np.float64)
    if len(times) < 2:
        s_prob = survival_predict_fn(X_test, times)
        return brier_score_loss(
            durations_train,
            events_train,
            durations_test,
            events_test,
            s_prob,
            float(times[0]),
        )

    brier_scores = []
    for t in times:
        s_prob = survival_predict_fn(X_test, np.array([t]))
        bs = brier_score_loss(
            durations_train,
            events_train,
            durations_test,
            events_test,
            s_prob.ravel(),
            float(t),
        )
        brier_scores.append(bs)

    bs_arr = np.array(brier_scores, dtype=np.float64)
    # Trapezoidal integration
    auc: float = float(
        np.sum((bs_arr[:-1] + bs_arr[1:]) * 0.5 * (times[1:] - times[:-1]))
    )
    time_span = float(times[-1] - times[0])
    return float(auc / time_span) if time_span > 0 else float(bs_arr[0])
