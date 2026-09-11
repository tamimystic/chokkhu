"""Comprehensive Unit Tests for Milestone 25 (v1.5.5).

Covers:
1. Fourier Neural Operator (FNO-2D: SpectralConv2d, FourierNeuralOperator2D)
2. Instrumental Variables Regression Suite (TwoStageLeastSquares, InstrumentalGMM)
3. Learning-to-Rank (LambdaMART, ListNet, NDCG@k)
4. Score-Based Energy Modeling (ScoreMatchingEBM, AnnealedLangevinDynamics)
5. Topological Data Analysis Vectorization (PersistenceLandscape, PersistenceImage)
"""

import numpy as np

from chokkhu.models.sciml.fno import SpectralConv2d, FourierNeuralOperator2D
from chokkhu.models.causal.instrumental import TwoStageLeastSquares, InstrumentalGMM
from chokkhu.ranking.lambdamart import LambdaMART, ListNet, ndcg_at_k
from chokkhu.models.energy.score_matching import ScoreMatchingEBM, AnnealedLangevinDynamics
from chokkhu.tda.vectorization import PersistenceLandscape, PersistenceImage


# =====================================================================
# 1. Fourier Neural Operator (FNO-2D) Tests
# =====================================================================

def test_spectral_conv2d_forward():
    """Test 2D Spectral Convolution layer forward pass and shapes."""
    conv = SpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6, seed=42)
    x = np.random.RandomState(42).randn(2, 16, 16, 4).astype(np.float32)
    out = conv.forward(x)

    assert out.shape == (2, 16, 16, 8)
    assert np.all(np.isfinite(out))


def test_fno2d_forward_and_resolution_invariance():
    """Test FNO-2D forward pass across varying spatial resolutions."""
    fno = FourierNeuralOperator2D(
        in_channels=1,
        out_channels=1,
        modes1=8,
        modes2=8,
        hidden_dim=16,
        num_layers=2,
        include_grid=True,
        seed=42,
    )

    # Resolution 1: 16x16
    x1 = np.random.RandomState(42).randn(2, 16, 16, 1).astype(np.float32)
    out1 = fno.forward(x1)
    assert out1.shape == (2, 16, 16, 1)
    assert np.all(np.isfinite(out1))

    # Resolution 2: 24x24 (zero-shot resolution scaling)
    x2 = np.random.RandomState(42).randn(1, 24, 24, 1).astype(np.float32)
    out2 = fno.forward(x2)
    assert out2.shape == (1, 24, 24, 1)
    assert np.all(np.isfinite(out2))


def test_fno2d_fit():
    """Test FNO-2D training loop."""
    rng = np.random.RandomState(42)
    X = rng.randn(8, 12, 12, 1).astype(np.float32)
    Y = np.sin(X) + 0.1 * rng.randn(8, 12, 12, 1).astype(np.float32)

    fno = FourierNeuralOperator2D(
        in_channels=1,
        out_channels=1,
        modes1=4,
        modes2=4,
        hidden_dim=8,
        num_layers=2,
        seed=42,
    )
    fno.fit(X, Y, epochs=2, batch_size=4)
    pred = fno.forward(X)
    assert pred.shape == Y.shape


# =====================================================================
# 2. Instrumental Variables Regression (2SLS & GMM) Tests
# =====================================================================

def test_twostage_least_squares_endogeneity():
    """Test 2SLS consistency under endogenous omitted variable confounding."""
    rng = np.random.RandomState(42)
    N = 500

    # Exogenous instruments Z
    z1 = rng.randn(N)
    z2 = rng.randn(N)
    Z = np.column_stack([z1, z2])

    # Unobserved confounder u correlated with endogenous regressor x and error
    u = rng.randn(N)
    x = 1.2 * z1 + 0.9 * z2 + 1.5 * u + 0.2 * rng.randn(N)
    y = 2.5 + 3.0 * x + 2.0 * u + 0.5 * rng.randn(N)

    # 2SLS Estimation
    iv = TwoStageLeastSquares(fit_intercept=True, robust=True)
    iv.fit(x, y, Z)

    # Check that estimated slope is close to true causal effect (3.0)
    assert np.isclose(iv.coef_[0], 3.0, atol=0.45)
    assert np.isclose(iv.intercept_, 2.5, atol=0.6)

    # Check diagnostic metrics
    summary = iv.summary()
    assert summary["first_stage_F"] > 10.0  # Strong instruments
    assert summary["sargan_statistic"] is not None
    assert summary["sargan_p_value"] is not None

    # Test prediction
    preds = iv.predict(x[:5])
    assert preds.shape == (5,)


def test_instrumental_gmm():
    """Test 2-step optimal Instrumental GMM estimator."""
    rng = np.random.RandomState(42)
    N = 400

    z1 = rng.randn(N)
    z2 = rng.randn(N)
    z3 = rng.randn(N)
    Z = np.column_stack([z1, z2, z3])

    u = rng.randn(N)
    x = 0.8 * z1 + 0.7 * z2 + 0.5 * z3 + 1.0 * u
    y = 1.0 + 2.0 * x + 1.5 * u

    gmm = InstrumentalGMM(fit_intercept=True)
    gmm.fit(x, y, Z)

    assert np.isclose(gmm.coef_[0], 2.0, atol=0.45)
    assert gmm.j_stat_ >= 0.0
    assert 0.0 <= gmm.j_p_val_ <= 1.0


# =====================================================================
# 3. Learning-to-Rank (LambdaMART & ListNet) Tests
# =====================================================================

def test_ndcg_at_k():
    """Test NDCG@k metric computation."""
    rel_perfect = np.array([3, 2, 1, 0])
    assert np.isclose(ndcg_at_k(rel_perfect, k=4), 1.0)

    rel_worst = np.array([0, 1, 2, 3])
    assert ndcg_at_k(rel_worst, k=4) < 1.0


def test_lambdamart_ranking():
    """Test LambdaMART training and ranking quality."""
    rng = np.random.RandomState(42)

    # 4 queries, 6 documents each = 24 samples
    N = 24
    query_ids = np.repeat(np.arange(4), 6)
    # Features: relevant documents have higher feature values
    relevance = rng.choice([0, 1, 2, 3], size=N)
    X = relevance[:, None] * 1.5 + rng.randn(N, 5) * 0.5

    model = LambdaMART(n_estimators=10, learning_rate=0.1, max_depth=2, seed=42)
    model.fit(X, relevance, query_ids)

    scores = model.predict(X)
    assert len(scores) == N

    # Check NDCG on query 0
    q0_mask = (query_ids == 0)
    q0_rel = relevance[q0_mask]
    q0_scores = scores[q0_mask]
    q0_sorted_rel = q0_rel[np.argsort(q0_scores)[::-1]]

    assert ndcg_at_k(q0_sorted_rel, k=6) >= 0.75


def test_listnet_ranking():
    """Test ListNet listwise cross-entropy ranking."""
    rng = np.random.RandomState(42)
    N = 30
    query_ids = np.repeat(np.arange(5), 6)
    relevance = rng.choice([0, 1, 2, 3], size=N)
    X = relevance[:, None] * 2.0 + rng.randn(N, 4) * 0.5

    listnet = ListNet(lr=0.05, epochs=40, seed=42)
    listnet.fit(X, relevance, query_ids)

    preds = listnet.predict(X)
    assert len(preds) == N
    assert np.all(np.isfinite(preds))


# =====================================================================
# 4. Energy-Based Score Matching Tests
# =====================================================================

def test_score_matching_ebm_and_langevin():
    """Test ScoreMatchingEBM with DSM and Annealed Langevin Dynamics."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2) * 0.5  # Zero-mean Gaussian cluster

    ebm = ScoreMatchingEBM(input_dim=2, hidden_dim=32, method="dsm", sigma=0.1, lr=1e-2, seed=42)
    ebm.fit(X, epochs=15, batch_size=25)

    # Score vector at (0, 0) should be small
    s_origin = ebm.score(np.array([0.0, 0.0]))
    assert s_origin.shape == (2,)

    # Score vector at (3, 3) should point inwards towards origin (negative components)
    s_far = ebm.score(np.array([3.0, 3.0]))
    assert s_far[0] < 0.0 or s_far[1] < 0.0

    # Langevin sampling
    sampler = AnnealedLangevinDynamics(ebm, n_steps_per_sigma=5, n_sigmas=4, seed=42)
    samples = sampler.sample(n_samples=6)
    assert samples.shape == (6, 2)
    assert np.all(np.isfinite(samples))


# =====================================================================
# 5. Topological Data Analysis Vectorization Tests
# =====================================================================

def test_persistence_landscape_vectorization():
    """Test PersistenceLandscape vectorization of persistence diagrams."""
    diagram = np.array([
        [0.0, 1.0],
        [0.2, 0.8],
        [0.3, 0.6],
        [0.5, 0.7],
    ])

    pl = PersistenceLandscape(n_landscapes=3, n_bins=50, t_min=0.0, t_max=1.0)
    single_lands = pl.transform_single(diagram)

    assert single_lands.shape == (3, 50)
    # Monotonicity property: lambda_1(t) >= lambda_2(t) >= lambda_3(t)
    assert np.all(single_lands[0] >= single_lands[1] - 1e-12)
    assert np.all(single_lands[1] >= single_lands[2] - 1e-12)

    # Batch transform
    batch_features = pl.transform([diagram, diagram])
    assert batch_features.shape == (2, 3 * 50)


def test_persistence_image_vectorization():
    """Test PersistenceImage 2D density surface generation."""
    diagram = np.array([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.2, 0.5],
    ])

    pi = PersistenceImage(pixels=(15, 15), sigma=0.15, weight_power=1.0)
    img = pi.transform_single(diagram)

    assert img.shape == (15, 15)
    assert np.all(img >= 0.0)
    assert np.sum(img) > 0.0

    # Batch transform
    batch_imgs = pi.transform([diagram, diagram, diagram])
    assert batch_imgs.shape == (3, 15 * 15)
