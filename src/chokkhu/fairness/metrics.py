"""Algorithmic Fairness Metrics & Disparity Diagnostics.

Pure NumPy implementations of:
- demographic_parity_difference / demographic_parity_ratio (Statistical Parity)
- equalized_odds_difference / equal_opportunity_difference
- disparate_impact_ratio (80% Rule)
- theil_index (Generalized Entropy Index of Inequality)
- fairness_report: Comprehensive diagnostic dictionary
"""

from typing import Dict, Optional, Union
import numpy as np


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    r"""Compute Demographic Parity Difference (Statistical Parity Gap).

    .. math::
        \Delta_{\text{DP}} = \max_a P(\hat{Y}=1 | A=a) - \min_a P(\hat{Y}=1 | A=a)

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Binary predicted labels :math:`\hat{Y} \in \{0, 1\}`.
    sensitive_features : array-like of shape (n_samples,)
        Protected attribute/group indicator :math:`A`.

    Returns
    -------
    diff : float
        Demographic parity difference in range [0.0, 1.0]. (0.0 = perfect demographic parity).
    """
    preds = np.asarray(y_pred, dtype=np.int32).ravel()
    sens = np.asarray(sensitive_features).ravel()

    if len(preds) != len(sens):
        raise ValueError("y_pred and sensitive_features must have identical length.")

    groups = np.unique(sens)
    if len(groups) < 2:
        return 0.0

    selection_rates: list[float] = []
    for g in groups:
        mask = sens == g
        if np.sum(mask) > 0:
            rate = float(np.mean(preds[mask] == 1))
            selection_rates.append(rate)

    if not selection_rates:
        return 0.0

    return float(max(selection_rates) - min(selection_rates))


def demographic_parity_ratio(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    r"""Compute Demographic Parity Ratio (Disparate Impact / 80% Rule Ratio).

    .. math::
        \text{DPR} = \frac{\min_a P(\hat{Y}=1 | A=a)}{\max_a P(\hat{Y}=1 | A=a)}

    Returns
    -------
    ratio : float
        Ratio in range [0.0, 1.0]. (1.0 = perfect parity, >= 0.8 satisfies EEOC Four-Fifths rule).
    """
    preds = np.asarray(y_pred, dtype=np.int32).ravel()
    sens = np.asarray(sensitive_features).ravel()

    groups = np.unique(sens)
    if len(groups) < 2:
        return 1.0

    rates = [
        float(np.mean(preds[sens == g] == 1)) for g in groups if np.sum(sens == g) > 0
    ]
    max_rate = max(rates) if rates else 0.0
    min_rate = min(rates) if rates else 0.0

    if max_rate <= 1e-12:
        return 1.0 if min_rate <= 1e-12 else 0.0

    return float(min_rate / max_rate)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    r"""Compute Equal Opportunity Difference (True Positive Rate Gap).

    .. math::
        \Delta_{\text{EOpp}} = \max_a \text{TPR}_a - \min_a \text{TPR}_a
    """
    y_t = np.asarray(y_true, dtype=np.int32).ravel()
    y_p = np.asarray(y_pred, dtype=np.int32).ravel()
    sens = np.asarray(sensitive_features).ravel()

    groups = np.unique(sens)
    if len(groups) < 2:
        return 0.0

    tprs: list[float] = []
    for g in groups:
        mask = (sens == g) & (y_t == 1)
        if np.sum(mask) > 0:
            tpr = float(np.mean(y_p[mask] == 1))
            tprs.append(tpr)

    if len(tprs) < 2:
        return 0.0

    return float(max(tprs) - min(tprs))


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    r"""Compute Equalized Odds Difference:

    .. math::
        \Delta_{\text{EOdds}} = \max \left( |\Delta \text{TPR}|, |\Delta \text{FPR}| \right)
    """
    y_t = np.asarray(y_true, dtype=np.int32).ravel()
    y_p = np.asarray(y_pred, dtype=np.int32).ravel()
    sens = np.asarray(sensitive_features).ravel()

    groups = np.unique(sens)
    if len(groups) < 2:
        return 0.0

    tprs: list[float] = []
    fprs: list[float] = []
    for g in groups:
        mask_pos = (sens == g) & (y_t == 1)
        mask_neg = (sens == g) & (y_t == 0)

        if np.sum(mask_pos) > 0:
            tprs.append(float(np.mean(y_p[mask_pos] == 1)))
        if np.sum(mask_neg) > 0:
            fprs.append(float(np.mean(y_p[mask_neg] == 1)))

    diff_tpr = float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    diff_fpr = float(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0

    return float(max(diff_tpr, diff_fpr))


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    """Calculate disparate impact ratio."""
    return demographic_parity_ratio(y_pred, sensitive_features)


def theil_index(
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> float:
    r"""Compute Generalized Entropy / Theil Inequality Index.

    Measures inequality in benefit/error distribution:

    .. math::
        T = \frac{1}{N} \sum_{i=1}^N \frac{b_i}{\mu_b} \ln \left( \frac{b_i}{\mu_b} \right)
    """
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true is not None:
        t = np.asarray(y_true, dtype=np.float64).ravel()
        # Benefit = 1 - error = 1 - |y_true - y_pred|
        benefits = np.maximum(1.0 - np.abs(t - p) + 1e-4, 1e-4)
    else:
        benefits = np.maximum(p + 1e-4, 1e-4)

    mu = float(np.mean(benefits))
    if mu <= 1e-12:
        return 0.0

    ratios = benefits / mu
    theil = float(np.mean(ratios * np.log(ratios)))
    return max(0.0, theil)


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Dict[str, Union[float, bool]]:
    """Generate a comprehensive multi-metric fairness audit report."""
    di_ratio = disparate_impact_ratio(y_pred, sensitive_features)
    return {
        "demographic_parity_difference": demographic_parity_difference(
            y_pred, sensitive_features
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_pred, sensitive_features
        ),
        "equal_opportunity_difference": equal_opportunity_difference(
            y_true, y_pred, sensitive_features
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features
        ),
        "disparate_impact_ratio": di_ratio,
        "theil_index": theil_index(y_pred, y_true),
        "four_fifths_rule_passed": bool(di_ratio >= 0.8),
    }
