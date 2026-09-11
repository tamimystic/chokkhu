"""Comprehensive Unit Tests for Milestone 26 (v1.5.6).

Covers:
1. Sequential Monte Carlo Particle Filtering (ParticleFilter, RaoBlackwellizedParticleFilter)
2. Double Machine Learning (DoubleMLPLR, RLearner)
3. Sparse Gaussian Process Regression (SparseGaussianProcessRegression, VariationalSparseGP)
4. Graph Contrastive Representation Learning (GraphCL, GRACE)
5. Inductive Neurosymbolic Program Synthesizer (ProgramSynthesizer, DSLGrammar)
"""

import numpy as np

from chokkhu.models.timeseries.particle_filter import (
    ParticleFilter,
    RaoBlackwellizedParticleFilter,
)
from chokkhu.models.causal.double_ml import (
    DoubleMLPLR,
    RLearner,
)
from chokkhu.models.sciml.sparse_gp import (
    SparseGaussianProcessRegression,
    VariationalSparseGP,
)
from chokkhu.models.gnn.contrastive import (
    GraphCL,
    GRACE,
)
from chokkhu.models.neurosymbolic.synthesizer import (
    ProgramSynthesizer,
)


# =====================================================================
# 1. Sequential Monte Carlo & Particle Filtering Tests
# =====================================================================


def test_particle_filter_tracking_and_resampling():
    """Test SIR particle filter state tracking and all resampling schemes."""

    # Non-linear 1D motion model
    def f(x, u=None):
        return 0.5 * x + 2.0 * np.cos(x)

    def h(x):
        return np.array([x[0] ** 2])

    for method in ["systematic", "stratified", "residual", "multinomial"]:
        pf = ParticleFilter(
            dim_x=1,
            dim_z=1,
            n_particles=150,
            f=f,
            h=h,
            resample_method=method,
            seed=42,
        )
        pf.initialize_particles(mean=np.array([1.0]), cov=np.array([[0.5]]))

        true_state = 1.0
        for _ in range(5):
            true_state = 0.5 * true_state + 2.0 * np.cos(true_state)
            meas = np.array([true_state**2 + 0.1 * np.random.randn()])

            pf.predict()
            est_mean, est_cov = pf.update(meas)

            assert est_mean.shape == (1,)
            assert est_cov.shape == (1, 1)
            assert np.all(np.isfinite(est_mean))
            assert np.all(np.isfinite(est_cov))


def test_raoblackwellized_particle_filter():
    """Test Rao-Blackwellized Particle Filter with mixed non-linear/linear states."""
    rbpf = RaoBlackwellizedParticleFilter(
        dim_nl=1,
        dim_lin=2,
        dim_z=2,
        n_particles=50,
        seed=42,
    )
    z = np.array([1.0, 0.5])
    mean_nl, mean_lin = rbpf.step(z)

    assert mean_nl.shape == (1,)
    assert mean_lin.shape == (2,)
    assert np.all(np.isfinite(mean_nl))
    assert np.all(np.isfinite(mean_lin))


# =====================================================================
# 2. Double Machine Learning (DoubleMLPLR & RLearner) Tests
# =====================================================================


def test_double_ml_plr():
    """Test DoubleMLPLR with K-fold cross-fitting and Neyman orthogonal scores."""
    rng = np.random.RandomState(42)
    N = 400
    P = 4

    X = rng.randn(N, P)
    # Non-linear nuisance functions
    g_X = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    m_X = 1.0 / (1.0 + np.exp(-X[:, 0] - X[:, 2]))

    # Treatment D and Outcome Y with true treatment effect theta = 2.5
    d = m_X + 0.5 * rng.randn(N)
    y = 2.5 * d + g_X + 0.5 * rng.randn(N)

    dml = DoubleMLPLR(n_folds=3, alpha=1.0, seed=42)
    dml.fit(X, y, d)

    assert np.isclose(dml.coef_, 2.5, atol=0.45)
    assert dml.se_ > 0.0
    assert dml.ci_lower_ < dml.coef_ < dml.ci_upper_

    summary = dml.summary()
    assert "theta" in summary
    assert "p_value" in summary


def test_r_learner_cate():
    """Test R-Learner for heterogeneous treatment effect estimation."""
    rng = np.random.RandomState(42)
    N = 300
    P = 3

    X = rng.randn(N, P)
    # True heterogeneous treatment effect: tau(X) = 1.5 + 2.0 * X[:, 0]
    true_tau = 1.5 + 2.0 * X[:, 0]

    d = (rng.rand(N) > 0.5).astype(float)
    y = 1.0 + d * true_tau + 0.5 * X[:, 1] + 0.3 * rng.randn(N)

    r_learner = RLearner(alpha=0.5, seed=42)
    r_learner.fit(X, y, d)

    pred_tau = r_learner.predict_cate(X[:10])
    assert len(pred_tau) == 10
    assert np.all(np.isfinite(pred_tau))


# =====================================================================
# 3. Sparse Gaussian Process Regression Tests
# =====================================================================


def test_sparse_gp_regression():
    """Test SparseGaussianProcessRegression with inducing points and posterior intervals."""
    rng = np.random.RandomState(42)
    N = 80
    X = np.sort(rng.uniform(-3.0, 3.0, size=(N, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(N)

    sgp = SparseGaussianProcessRegression(
        n_inducing=12,
        length_scale=1.0,
        variance=1.0,
        noise_variance=0.01,
        seed=42,
    )
    sgp.fit(X, y)

    X_test = np.linspace(-3.0, 3.0, 20).reshape(-1, 1)
    mean, std = sgp.predict(X_test, return_std=True)

    assert mean.shape == (20,)
    assert std.shape == (20,)
    assert np.all(std > 0.0)

    # Check fit quality
    mse = np.mean((mean - np.sin(X_test[:, 0])) ** 2)
    assert mse < 0.25

    # Test alias
    vsgp = VariationalSparseGP(n_inducing=10)
    vsgp.fit(X, y)
    mean_v = vsgp.predict(X_test, return_std=False)
    assert mean_v.shape == (20,)


# =====================================================================
# 4. Graph Contrastive Learning (GraphCL & GRACE) Tests
# =====================================================================


def test_graphcl_contrastive_embeddings():
    """Test GraphCL and GRACE self-supervised graph contrastive representation learning."""
    rng = np.random.RandomState(42)
    N = 16
    in_dim = 6
    out_dim = 8

    # Random connected graph adjacency
    A = (rng.rand(N, N) > 0.7).astype(np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    X = rng.randn(N, in_dim)

    model = GraphCL(
        in_dim=in_dim,
        hidden_dim=12,
        out_dim=out_dim,
        temperature=0.5,
        drop_edge_rate=0.2,
        mask_feat_rate=0.2,
        seed=42,
    )
    model.fit(X, A, epochs=8)

    embeds = model.transform(X, A)
    assert embeds.shape == (N, out_dim)
    assert np.all(np.isfinite(embeds))

    # Test alias
    grace = GRACE(in_dim=in_dim, hidden_dim=12, out_dim=out_dim)
    grace.fit(X, A, epochs=2)
    grace_embeds = grace.transform(X, A)
    assert grace_embeds.shape == (N, out_dim)


# =====================================================================
# 5. Neurosymbolic Program Synthesizer Tests
# =====================================================================


def test_program_synthesizer_arithmetic():
    """Test inductive synthesis of arithmetic expression from I/O pairs."""
    # Target function: y = 2 * x + 1 (inc(double(x)))
    examples = [(1, 3), (2, 5), (3, 7), (4, 9)]

    synth = ProgramSynthesizer(max_depth=3)
    synth.fit(examples)

    assert synth.predict(5) == 11
    assert synth.predict(10) == 21
    assert len(synth.synthesized_code) > 0


def test_program_synthesizer_string():
    """Test inductive synthesis of string operations from I/O pairs."""
    # Target function: y = reverse(x)
    examples = [("abc", "cba"), ("hello", "olleh"), ("world", "dlrow")]

    synth = ProgramSynthesizer(max_depth=2)
    synth.fit(examples)

    assert synth.predict("chokkhu") == "uhkkohc"
