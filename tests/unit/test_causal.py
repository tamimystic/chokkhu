"""Comprehensive Unit Tests for Chokkhu Causal Inference & Uplift Modeling.

Tests:
- PropensityModel (Logistic propensity estimation)
- PropensityScoreMatching (PSM nearest-neighbor with caliper)
- InverseProbabilityWeighting (IPW Horvitz-Thompson & Hajek)
- DoublyRobustLearner (DR-Learner for ATE & CATE)
- SLearner (Single-model meta-learner)
- TLearner (Two-model meta-learner)
- XLearner (Crossover imputation meta-learner)
- TwoModelUplift (Differential conversion classifier)
- ClassTransformationUplift (Lai's generalized target transform)
- Uplift Metrics (qini_curve, qini_score, cumulative_gain_curve, uplift_at_k)
"""

import numpy as np

from chokkhu.models.causal import (
    ClassTransformationUplift,
    DoublyRobustLearner,
    InverseProbabilityWeighting,
    PropensityModel,
    PropensityScoreMatching,
    SLearner,
    TLearner,
    TwoModelUplift,
    XLearner,
    cumulative_gain_curve,
    qini_curve,
    qini_score,
    uplift_at_k,
)


def _generate_synthetic_causal_data(
    n_samples: int = 200, true_ate: float = 2.5, seed: int = 42
):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 4).astype(np.float32)
    # Propensity: higher X0 increases probability of treatment
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    prob_t = 1.0 / (1.0 + np.exp(-logits))
    T = (rng.rand(n_samples) < prob_t).astype(np.int32)
    # Outcome with confounding and true effect
    Y = (
        1.5
        + 2.0 * X[:, 0]
        + 1.0 * X[:, 2]
        + true_ate * T
        + rng.randn(n_samples).astype(np.float32) * 0.5
    )
    return X, T, Y


def test_propensity_model():
    X, T, _ = _generate_synthetic_causal_data(n_samples=100)
    model = PropensityModel(lr=0.05, n_epochs=50, seed=42)
    model.fit(X, T)

    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_propensity_score_matching():
    X, T, Y = _generate_synthetic_causal_data(n_samples=250, true_ate=3.0)
    psm = PropensityScoreMatching(caliper=0.5, seed=42)
    psm.fit(X, T, Y)

    ate = psm.estimate_ate()
    att = psm.estimate_att()
    assert isinstance(ate, float)
    assert isinstance(att, float)
    assert 1.0 <= ate <= 5.0  # captures positive true effect 3.0


def test_inverse_probability_weighting():
    X, T, Y = _generate_synthetic_causal_data(n_samples=250, true_ate=2.5)
    ipw = InverseProbabilityWeighting(stabilized=True, seed=42)
    ipw.fit(X, T, Y)

    ate = ipw.estimate_ate()
    assert isinstance(ate, float)
    assert 1.0 <= ate <= 4.0

    ci_low, ci_high = ipw.confidence_interval()
    assert ci_low < ate < ci_high


def test_doubly_robust_learner():
    X, T, Y = _generate_synthetic_causal_data(n_samples=200, true_ate=2.0)
    dr = DoublyRobustLearner(alpha=1.0, seed=42)
    dr.fit(X, T, Y)

    ate = dr.estimate_ate()
    assert isinstance(ate, float)
    assert 0.5 <= ate <= 3.5

    cate = dr.predict_cate(X[:10])
    assert cate.shape == (10,)


def test_s_learner():
    X, T, Y = _generate_synthetic_causal_data(n_samples=150, true_ate=2.0)
    s_learner = SLearner(alpha=1.0)
    s_learner.fit(X, T, Y)

    cate = s_learner.predict_cate(X[:10])
    assert cate.shape == (10,)
    ate = s_learner.estimate_ate(X)
    assert isinstance(ate, float)
    assert 0.5 <= ate <= 3.5


def test_t_learner():
    X, T, Y = _generate_synthetic_causal_data(n_samples=150, true_ate=2.5)
    t_learner = TLearner(alpha=1.0)
    t_learner.fit(X, T, Y)

    cate = t_learner.predict_cate(X[:10])
    assert cate.shape == (10,)
    ate = t_learner.estimate_ate(X)
    assert 1.0 <= ate <= 4.0


def test_x_learner():
    # Heterogeneous treatment effect: tau(X) = 1.0 + 2.0 * X0
    rng = np.random.RandomState(42)
    N = 300
    X = rng.randn(N, 3).astype(np.float32)
    T = (rng.rand(N) < 0.3).astype(np.int32)  # imbalanced
    tau_true = 1.0 + 2.0 * X[:, 0]
    Y = 2.0 + 1.5 * X[:, 1] + tau_true * T + rng.randn(N).astype(np.float32) * 0.3

    x_learner = XLearner(alpha=1.0, seed=42)
    x_learner.fit(X, T, Y)

    cate = x_learner.predict_cate(X)
    assert cate.shape == (N,)
    # Check positive correlation with true treatment effects
    corr = float(np.corrcoef(cate, tau_true)[0, 1])
    assert corr > 0.5


def test_two_model_uplift():
    rng = np.random.RandomState(42)
    N = 150
    X = rng.randn(N, 4).astype(np.float32)
    T = rng.randint(0, 2, size=N).astype(np.int32)
    # Binary conversion
    logits = 0.5 * X[:, 0] + 0.8 * T
    probs = 1.0 / (1.0 + np.exp(-logits))
    Y = (rng.rand(N) < probs).astype(np.float32)

    uplift_model = TwoModelUplift(lr=0.05, n_epochs=50, seed=42)
    uplift_model.fit(X, T, Y)

    uplifts = uplift_model.predict_uplift(X)
    assert uplifts.shape == (N,)
    assert np.all((uplifts >= -1.0) & (uplifts <= 1.0))


def test_class_transformation_uplift():
    rng = np.random.RandomState(42)
    N = 150
    X = rng.randn(N, 4).astype(np.float32)
    T = rng.randint(0, 2, size=N).astype(np.int32)
    Y = (rng.rand(N) < 0.5).astype(np.float32)

    ctu = ClassTransformationUplift(lr=0.05, n_epochs=50, seed=42)
    ctu.fit(X, T, Y)

    uplifts = ctu.predict_uplift(X)
    assert uplifts.shape == (N,)
    assert np.all((uplifts >= -1.0) & (uplifts <= 1.0))


def test_uplift_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    treatment = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    uplift_preds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    fracs, qini = qini_curve(y_true, treatment, uplift_preds, n_bins=5)
    assert len(fracs) == 6
    assert len(qini) == 6

    score = qini_score(y_true, treatment, uplift_preds)
    assert isinstance(score, float)

    g_x, g_y = cumulative_gain_curve(y_true, treatment, uplift_preds, n_bins=5)
    assert len(g_x) == 6

    u_20 = uplift_at_k(y_true, treatment, uplift_preds, k=0.5)
    assert isinstance(u_20, float)
