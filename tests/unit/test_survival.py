"""Unit tests for Sovereign Survival Analysis & Reliability Modeling (Milestone 4)."""

import numpy as np
import pytest
from chokkhu.models.survival import (
    KaplanMeierFitter,
    NelsonAalenFitter,
    logrank_test,
    CoxPHRegression,
    DeepSurv,
    concordance_index,
    brier_score_loss,
    integrated_brier_score,
)


def test_kaplan_meier_basic():
    """Test KaplanMeierFitter curve estimation, confidence bounds, and predictions."""
    durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    events = np.array([1, 0, 1, 1, 0, 1, 0, 1])

    km = KaplanMeierFitter(alpha=0.05)
    km.fit(durations, events)

    assert km.is_fitted
    assert km.timeline is not None
    assert km.survival_probabilities is not None
    assert km.confidence_interval_ is not None
    assert km.timeline[0] == 0.0
    assert km.survival_probabilities[0] == 1.0

    # Survival probabilities should be monotonically non-increasing
    assert np.all(np.diff(km.survival_probabilities) <= 1e-9)

    # Predictions
    p_0 = km.predict(0.0)
    p_3 = km.predict(3.0)
    p_10 = km.predict(10.0)
    assert p_0 == 1.0
    assert 0.0 <= p_10 <= p_3 <= 1.0

    # Confidence intervals validity
    ci = km.confidence_interval_
    assert np.all(ci[:, 0] <= km.survival_probabilities + 1e-9)
    assert np.all(km.survival_probabilities <= ci[:, 1] + 1e-9)


def test_nelson_aalen_basic():
    """Test NelsonAalenFitter cumulative hazard estimator."""
    durations = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    events = np.array([1, 1, 0, 1, 1, 0])

    na = NelsonAalenFitter(alpha=0.05)
    na.fit(durations, events)

    assert na.is_fitted
    assert na.cumulative_hazard_ is not None
    # Cumulative hazard should be monotonically non-decreasing
    assert np.all(np.diff(na.cumulative_hazard_) >= -1e-9)

    # Predictions
    h_0 = na.predict(0.0)
    h_7 = na.predict(7.0)
    assert h_0 == 0.0
    assert h_7 >= 0.0


def test_logrank_test():
    """Test Two-Sample Log-Rank hypothesis test."""
    rng = np.random.RandomState(42)
    # Group A: shorter survival
    dur_a = rng.exponential(scale=5.0, size=50)
    evt_a = np.ones(50, dtype=int)

    # Group B: longer survival
    dur_b = rng.exponential(scale=25.0, size=50)
    evt_b = np.ones(50, dtype=int)

    chi2, p_val = logrank_test(dur_a, evt_a, dur_b, evt_b)
    assert chi2 > 0.0
    assert p_val < 0.01  # Significant difference

    # Test identical groups
    chi2_same, p_val_same = logrank_test(dur_a, evt_a, dur_a, evt_a)
    assert chi2_same == 0.0
    assert p_val_same == 1.0


def test_cox_ph_regression_efron():
    """Test CoxPHRegression with Efron ties on synthetic survival data."""
    rng = np.random.RandomState(42)
    n = 100
    p = 3
    X = rng.randn(n, p)
    true_beta = np.array([0.8, -0.6, 0.0])

    # Generate survival times from proportional hazards model
    hazard = np.exp(np.dot(X, true_beta))
    durations = rng.exponential(scale=1.0 / hazard)
    events = (rng.rand(n) > 0.1).astype(int)  # 10% censored

    model = CoxPHRegression(alpha=0.01, tie_method="efron", max_iter=50)
    model.fit(X, durations, events)

    assert model.is_fitted
    assert model.coef_ is not None
    assert len(model.coef_) == p

    # Check recovery of sign of strongest coefficients
    assert model.coef_[0] > 0.0
    assert model.coef_[1] < 0.0

    # Predictions
    partial_hazard = model.predict_partial_hazard(X[:5])
    risk = model.predict_risk(X[:5])
    assert len(partial_hazard) == 5
    assert len(risk) == 5
    assert np.all(partial_hazard > 0.0)

    surv_funcs = model.predict_survival_function(X[:5], times=[1.0, 2.0, 3.0])
    assert surv_funcs.shape == (5, 3)
    assert np.all((surv_funcs >= 0.0) & (surv_funcs <= 1.0))

    summary = model.summary()
    assert "hazard_ratio" in summary
    assert "p_value" in summary
    assert len(summary["hazard_ratio"]) == p


def test_cox_ph_regression_breslow():
    """Test CoxPHRegression with Breslow ties."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 2)
    durations = rng.exponential(scale=2.0, size=40)
    events = np.ones(40, dtype=int)

    model = CoxPHRegression(alpha=0.1, tie_method="breslow")
    model.fit(X, durations, events)

    assert model.is_fitted
    assert model.hazard_ratios_ is not None
    assert np.all(model.hazard_ratios_ > 0.0)


def test_deep_surv():
    """Test DeepSurv deep Cox neural network training and inference."""
    rng = np.random.RandomState(42)
    n = 80
    p = 4
    X = rng.randn(n, p)
    true_beta = np.array([1.0, -1.0, 0.5, 0.0])
    risk_true = np.dot(X, true_beta)
    durations = rng.exponential(scale=1.0 / np.exp(np.clip(risk_true, -2.0, 2.0)))
    events = np.ones(n, dtype=int)

    ds = DeepSurv(
        hidden_layers=[16, 8], activation="relu", lr=0.01, n_epochs=50, seed=42
    )
    ds.fit(X, durations, events)

    assert ds.is_fitted
    assert len(ds.loss_history_) == 50
    # Loss should decrease from initial epochs to final
    assert ds.loss_history_[-1] < ds.loss_history_[0]

    # Predict risk & partial hazard
    pred_risk = ds.predict_risk(X[:10])
    pred_hazard = ds.predict_partial_hazard(X[:10])
    assert len(pred_risk) == 10
    assert np.all(pred_hazard > 0.0)

    # Predict survival curves
    surv_curves = ds.predict_survival_function(X[:5], times=[0.5, 1.0, 2.0])
    assert surv_curves.shape == (5, 3)
    assert np.all((surv_curves >= 0.0) & (surv_curves <= 1.0))


def test_concordance_index():
    """Test Harrell's C-index metric with various ranking scenarios."""
    durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 1, 1, 1])

    # Perfect ranking: high risk -> small duration
    risk_perfect = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    c_perf = concordance_index(durations, risk_perfect, events)
    assert c_perf == pytest.approx(1.0, abs=1e-6)

    # Inverse ranking: high risk -> large duration
    risk_inverse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c_inv = concordance_index(durations, risk_inverse, events)
    assert c_inv == pytest.approx(0.0, abs=1e-6)

    # Tied predictions: should give 0.5
    risk_tied = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    c_tied = concordance_index(durations, risk_tied, events)
    assert c_tied == pytest.approx(0.5, abs=1e-6)


def test_brier_score_loss_and_ibs():
    """Test IPCW Brier score and Integrated Brier Score calculation."""
    rng = np.random.RandomState(42)
    dur_train = rng.exponential(scale=5.0, size=50)
    evt_train = (rng.rand(50) > 0.2).astype(int)

    dur_test = rng.exponential(scale=5.0, size=30)
    evt_test = (rng.rand(30) > 0.2).astype(int)

    # Survival probabilities at t=3.0
    surv_probs = np.clip(np.exp(-dur_test / 5.0), 0.0, 1.0)
    bs = brier_score_loss(
        dur_train, evt_train, dur_test, evt_test, surv_probs, eval_time=3.0
    )
    assert 0.0 <= bs <= 1.0

    # Integrated Brier Score
    X_test = rng.randn(30, 2)

    def dummy_predict_fn(x, times):
        t_arr = np.asarray(times)
        return np.ones((len(x), len(t_arr))) * 0.7

    ibs = integrated_brier_score(
        dur_train,
        evt_train,
        dur_test,
        evt_test,
        dummy_predict_fn,
        X_test,
        eval_times=[1.0, 2.0, 3.0],
    )
    assert 0.0 <= ibs <= 1.0


def test_survival_edge_cases():
    """Test error handling and validation."""
    km = KaplanMeierFitter()
    with pytest.raises(RuntimeError):
        km.predict([1.0, 2.0])

    cox = CoxPHRegression()
    with pytest.raises(RuntimeError):
        cox.predict_risk(np.zeros((5, 2)))

    ds = DeepSurv()
    with pytest.raises(RuntimeError):
        ds.predict_risk(np.zeros((5, 2)))

    with pytest.raises(ValueError):
        CoxPHRegression(tie_method="invalid_tie")

    with pytest.raises(ValueError):
        km.fit(np.array([-1.0, 2.0]))
