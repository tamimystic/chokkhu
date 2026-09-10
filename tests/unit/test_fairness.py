"""Unit tests for Algorithmic Fairness & Bias Mitigation Module."""

import numpy as np
import pytest

from chokkhu.fairness import (
    ReweighingTransformer,
    DisparateImpactRemover,
    ThresholdOptimizer,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    disparate_impact_ratio,
    theil_index,
    fairness_report,
)


def test_demographic_parity_difference() -> None:
    # Group 0: 80% positive selection (8/10), Group 1: 20% positive selection (2/10)
    sens = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)

    diff = demographic_parity_difference(y_pred, sens)
    assert pytest.approx(diff, 1e-4) == 0.6

    # Fair scenario: 50% for both groups
    y_fair = np.array([1, 0] * 5 + [1, 0] * 5)
    diff_fair = demographic_parity_difference(y_fair, sens)
    assert pytest.approx(diff_fair, 1e-4) == 0.0


def test_demographic_parity_ratio() -> None:
    # Group 0: 80% selection, Group 1: 40% selection -> ratio = 40/80 = 0.5 (< 0.8 fails 4/5ths rule)
    sens = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 4 + [0] * 6)

    ratio = demographic_parity_ratio(y_pred, sens)
    assert pytest.approx(ratio, 1e-4) == 0.5
    assert pytest.approx(disparate_impact_ratio(y_pred, sens), 1e-4) == 0.5


def test_equal_opportunity_and_equalized_odds() -> None:
    # 20 samples: 10 group 0, 10 group 1
    sens = np.array([0] * 10 + [1] * 10)
    y_true = np.array([1] * 5 + [0] * 5 + [1] * 5 + [0] * 5)
    # Group 0: all 5 positives predicted 1 (TPR = 1.0), 1 negative predicted 1 (FPR = 0.2)
    # Group 1: 3 positives predicted 1 (TPR = 0.6), 2 negatives predicted 1 (FPR = 0.4)
    y_pred = np.array([1] * 5 + [1, 0, 0, 0, 0] + [1] * 3 + [0, 0] + [1, 1, 0, 0, 0])

    eopp = equal_opportunity_difference(y_true, y_pred, sens)
    assert pytest.approx(eopp, 1e-4) == 0.4  # |1.0 - 0.6| = 0.4

    eodds = equalized_odds_difference(y_true, y_pred, sens)
    # TPR diff = 0.4, FPR diff = |0.2 - 0.4| = 0.2 -> max is 0.4
    assert pytest.approx(eodds, 1e-4) == 0.4


def test_theil_index() -> None:
    # Perfectly equal benefits
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    y_true = np.array([1, 1, 1, 1])
    t_index = theil_index(y_pred, y_true)
    assert pytest.approx(t_index, 1e-4) == 0.0

    # Unequal benefits
    y_pred_unequal = np.array([0.9, 0.1, 0.9, 0.1])
    t_unequal = theil_index(y_pred_unequal, y_true)
    assert t_unequal > 0.0


def test_fairness_report() -> None:
    sens = np.array([0] * 10 + [1] * 10)
    y_true = np.array([1] * 5 + [0] * 5 + [1] * 5 + [0] * 5)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 4 + [0] * 6)

    report = fairness_report(y_true, y_pred, sens)
    assert "demographic_parity_difference" in report
    assert "demographic_parity_ratio" in report
    assert "equal_opportunity_difference" in report
    assert "equalized_odds_difference" in report
    assert "theil_index" in report
    assert "four_fifths_rule_passed" in report
    assert isinstance(report["four_fifths_rule_passed"], bool)


def test_reweighing_transformer() -> None:
    # Biased dataset: Group 0 is mostly positive (8/10), Group 1 is mostly negative (2/10 positive)
    sens = np.array([0] * 10 + [1] * 10)
    y = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)

    reweighter = ReweighingTransformer()
    weights = reweighter.fit_transform(y, sens)

    assert len(weights) == 20
    assert np.all(weights > 0.0)

    # In group 0 (favored), positive instances should be downweighted (w < 1)
    assert weights[0] < 1.0
    # In group 1 (deprived), positive instances should be upweighted (w > 1)
    assert weights[10] > 1.0

    # Total weighted probability of Y=1 given A=0 should equal total weighted probability of Y=1 given A=1
    mask0 = sens == 0
    mask1 = sens == 1
    weighted_p0 = float(np.sum(weights[mask0 & (y == 1)])) / float(
        np.sum(weights[mask0])
    )
    weighted_p1 = float(np.sum(weights[mask1 & (y == 1)])) / float(
        np.sum(weights[mask1])
    )
    assert pytest.approx(weighted_p0, 1e-4) == weighted_p1


def test_disparate_impact_remover() -> None:
    np.random.seed(42)
    # Group 0: drawn from N(10, 2), Group 1: drawn from N(0, 2)
    X0 = np.random.normal(10.0, 1.0, size=50)
    X1 = np.random.normal(0.0, 1.0, size=50)
    X = np.concatenate([X0, X1])
    sens = np.array([0] * 50 + [1] * 50)

    remover = DisparateImpactRemover(repair_level=1.0)
    X_repaired = remover.fit_transform(X, sens)

    assert X_repaired.shape == X.shape
    # Means after full repair should be much closer
    mean_diff_before = abs(float(np.mean(X[:50])) - float(np.mean(X[50:])))
    mean_diff_after = abs(
        float(np.mean(X_repaired[:50])) - float(np.mean(X_repaired[50:]))
    )
    assert mean_diff_after < mean_diff_before
    assert mean_diff_after < 1.0

    # 2D feature matrix test
    X_2d = np.column_stack([X, X * 2.0])
    X_2d_repaired = remover.fit_transform(X_2d, sens)
    assert X_2d_repaired.shape == (100, 2)


def test_threshold_optimizer() -> None:
    np.random.seed(42)
    sens = np.array([0] * 50 + [1] * 50)
    # Group 0 predicted probabilities higher overall than Group 1
    y_proba = np.concatenate(
        [np.random.uniform(0.4, 0.9, size=50), np.random.uniform(0.1, 0.6, size=50)]
    )
    y_true = np.concatenate(
        [np.random.binomial(1, 0.7, size=50), np.random.binomial(1, 0.7, size=50)]
    )

    # Raw fixed threshold (0.5) has high DP difference
    raw_preds = (y_proba >= 0.5).astype(int)
    raw_dp = demographic_parity_difference(raw_preds, sens)

    opt = ThresholdOptimizer(constraint="demographic_parity", grid_size=50)
    opt.fit(y_true, y_proba, sens)
    fair_preds = opt.predict(y_proba, sens)

    fair_dp = demographic_parity_difference(fair_preds, sens)
    assert fair_dp <= raw_dp
    assert len(opt.group_thresholds_) == 2


def test_fairness_edge_cases() -> None:
    # Single group returns default parity
    sens_single: np.ndarray = np.zeros(10, dtype=int)
    y_pred: np.ndarray = np.ones(10, dtype=int)
    assert demographic_parity_difference(y_pred, sens_single) == 0.0
    assert demographic_parity_ratio(y_pred, sens_single) == 1.0

    # Invalid length
    with pytest.raises(ValueError):
        demographic_parity_difference(np.array([1, 0]), np.array([0]))

    # Invalid repair level
    with pytest.raises(ValueError):
        DisparateImpactRemover(repair_level=1.5)

    # Invalid constraint
    with pytest.raises(ValueError):
        ThresholdOptimizer(constraint="invalid_constraint")
