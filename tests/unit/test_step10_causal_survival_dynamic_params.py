from __future__ import annotations

import numpy as np

from chokkhu.models.causal import (
    DoublyRobustLearner,
    InverseProbabilityWeighting,
    PropensityModel,
    PropensityScoreMatching,
    SLearner,
    TLearner,
    XLearner,
)
from chokkhu.models.survival import (
    CoxPHRegression,
    DeepSurv,
    KaplanMeierFitter,
    NelsonAalenFitter,
)


def _get_causal_data(n: int = 80, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3).astype(np.float32)
    logits = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    T = (rng.rand(n) < (1.0 / (1.0 + np.exp(-logits)))).astype(np.int32)
    Y = (1.0 + 1.5 * X[:, 0] + 2.0 * T + rng.randn(n).astype(np.float32) * 0.2).astype(
        np.float32
    )
    return X, T, Y


def _get_survival_data(n: int = 60, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    durations = rng.exponential(scale=10.0, size=n) + 0.5
    events = rng.binomial(1, 0.7, size=n)
    return X, durations, events


def test_causal_potential_outcomes_dynamic_params() -> None:
    X, T, Y = _get_causal_data()
    prop = PropensityModel(lr=0.02, n_epochs=30, reg=0.05, custom_param="ignored")
    assert prop.get_params()["lr"] == 0.02
    prop.set_params(lr=0.01)
    assert prop.lr == 0.01
    prop.fit(X, T)
    preds = prop.predict(X)
    assert len(preds) == len(X) and set(np.unique(preds)).issubset({0, 1})

    psm = PropensityScoreMatching(caliper=0.25, replace=True, extra_kw=123)
    assert psm.get_params()["replace"] is True
    psm.fit(X, T, Y)
    ate_val = psm.predict(X)
    assert isinstance(ate_val, float)

    ipw = InverseProbabilityWeighting(stabilized=True, clip_range=(0.05, 0.95))
    assert ipw.get_params()["stabilized"] is True
    ipw.fit(X, T, Y)
    assert isinstance(ipw.predict(X), float)


def test_meta_learners_and_doubly_robust_dynamic_params() -> None:
    X, T, Y = _get_causal_data()
    for LearnerClass in (SLearner, TLearner, XLearner, DoublyRobustLearner):
        model = LearnerClass(alpha=0.5, custom_arg=999)
        assert model.get_params()["alpha"] == 0.5
        model.set_params(alpha=1.5)
        assert model.alpha == 1.5
        model.fit(X, T, Y)
        assert model.is_fitted
        cate_preds = model.predict(X)
        assert len(cate_preds) == len(X)
        assert np.isfinite(cate_preds).all()


def test_survival_models_dynamic_params() -> None:
    X, durations, events = _get_survival_data()

    km = KaplanMeierFitter(alpha=0.1, custom_kw="abc")
    assert km.get_params()["alpha"] == 0.1
    km.set_params(alpha=0.05)
    km.fit(durations, events)
    assert km.is_fitted
    p = km.predict(5.0)
    assert 0.0 <= p <= 1.0

    na = NelsonAalenFitter(alpha=0.1)
    assert na.get_params()["alpha"] == 0.1
    na.fit(durations, events)
    assert na.is_fitted
    h = na.predict(5.0)
    assert h >= 0.0

    cox = CoxPHRegression(alpha=0.05, tie_method="breslow", max_iter=50)
    assert cox.get_params()["tie_method"] == "breslow"
    cox.fit(X, durations, events)
    assert cox.is_fitted
    risks = cox.predict(X)
    assert len(risks) == len(X)

    ds = DeepSurv(hidden_layers=[16, 8], activation="tanh", n_epochs=20)
    assert ds.get_params()["activation"] == "tanh"
    ds.fit(X, durations, events)
    assert ds.is_fitted
    ds_risks = ds.predict(X)
    assert len(ds_risks) == len(X)
