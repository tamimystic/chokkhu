"""Unit tests for Milestone 22: SE(3) Geometric DL, FTRL Online, KSG Info, EBM & Model Trees.

Tests covering:
- SphericalHarmonics & SE3EquivariantConv
- FollowTheRegularizedLeader (FTRL-Proximal)
- HedgeAlgorithm
- KraskovMutualInformation
- MultivariateKDE
- EnergyBasedModel & SGLD
- SlicedScoreMatching
- M5ModelTree
- RuleFitRegressor & RuleFitClassifier
"""

import numpy as np

from chokkhu import (
    EnergyBasedModel,
    FollowTheRegularizedLeader,
    HedgeAlgorithm,
    KraskovMutualInformation,
    M5ModelTree,
    MultivariateKDE,
    RuleFitClassifier,
    RuleFitRegressor,
    SE3EquivariantConv,
    SlicedScoreMatching,
    SphericalHarmonics,
)


def test_spherical_harmonics_and_se3_conv():
    # 1. Spherical Harmonics Evaluations
    vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    Y_l0 = SphericalHarmonics.compute(vecs, max_l=0)
    assert Y_l0.shape == (3, 1)
    assert np.allclose(Y_l0[:, 0], 0.5 * np.sqrt(1.0 / np.pi))

    Y_l2 = SphericalHarmonics.compute(vecs, max_l=2)
    assert Y_l2.shape == (3, 9)

    # 2. SE(3) Equivariant Steerable Point Convolution
    conv = SE3EquivariantConv(
        in_dim=4,
        out_dim=8,
        max_l=1,
        num_radial_bases=4,
        cutoff_radius=5.0,
        seed=42,
    )

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    node_feats = np.ones((4, 4))

    out_feats = conv.forward(pos, node_feats)
    assert out_feats.shape == (4, 8)
    assert not np.any(np.isnan(out_feats))


def test_ftrl_proximal_online_learning():
    # 1. Binary classification stream
    ftrl = FollowTheRegularizedLeader(
        alpha=0.2, beta=1.0, lambda1=0.01, lambda2=0.1, loss="logistic"
    )

    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    # Simple linear decision boundary
    y = (X[:, 0] * 2.0 - X[:, 1] > 0.0).astype(float)

    for i in range(len(X)):
        loss = ftrl.update_one(X[i], y[i])
        assert isinstance(loss, float)

    preds = ftrl.predict(X)
    assert preds.shape == (100,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    weights = ftrl.get_weights()
    assert weights.shape == (5,)

    # 2. Linear regression stream
    ftrl_reg = FollowTheRegularizedLeader(
        alpha=0.1, beta=1.0, lambda1=0.0, lambda2=0.01, loss="squared"
    )
    y_reg = X[:, 0] * 3.0 + 1.0
    ftrl_reg.fit(X, y_reg, epochs=2)
    reg_preds = ftrl_reg.predict(X)
    assert reg_preds.shape == (100,)


def test_hedge_multi_expert_regret():
    num_experts = 4
    hedge = HedgeAlgorithm(num_experts=num_experts, time_horizon=50)

    # Expert 0 is consistently best (loss 0.1), Expert 3 is worst (loss 0.8)
    rng = np.random.RandomState(42)
    for _ in range(50):
        losses = np.array([0.1, 0.4, 0.6, 0.8]) + rng.randn(4) * 0.02
        losses = np.clip(losses, 0.0, 1.0)
        round_loss, regret = hedge.update(losses)
        assert round_loss >= 0.0

    final_dist = hedge.predict_distribution()
    assert final_dist.shape == (4,)
    # Expert 0 should have highest weight
    assert np.argmax(final_dist) == 0


def test_kraskov_mutual_information():
    ksg = KraskovMutualInformation(k=3)
    rng = np.random.RandomState(42)

    # 1. Independent Gaussians -> MI near 0
    X_indep = rng.randn(150, 1)
    Y_indep = rng.randn(150, 1)
    mi_indep = ksg.estimate(X_indep, Y_indep)
    assert mi_indep < 0.2

    # 2. Strongly dependent nonlinear relationship -> MI significantly positive
    X_dep = rng.randn(150, 1)
    Y_dep = np.sin(X_dep * 2.0) + 0.1 * rng.randn(150, 1)
    mi_dep = ksg.estimate(X_dep, Y_dep)
    assert mi_dep > 0.4


def test_multivariate_kde():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)

    kde = MultivariateKDE(bandwidth="silverman", kernel="gaussian")
    kde.fit(X)

    log_densities = kde.score_samples(X[:10])
    assert log_densities.shape == (10,)
    assert not np.any(np.isnan(log_densities))

    samples = kde.sample(n_samples=25, seed=42)
    assert samples.shape == (25, 2)


def test_energy_based_model_and_sgld():
    ebm = EnergyBasedModel(in_dim=3, hidden_dim=16, seed=42)

    x_test = np.array([[1.0, 0.5, -0.5], [0.0, 0.0, 0.0]])
    energies = ebm.energy(x_test)
    assert energies.shape == (2,)

    grad = ebm.energy_gradient(x_test)
    assert grad.shape == (2, 3)

    # SGLD sampling
    samples = ebm.sample_sgld(n_samples=10, num_steps=15, step_size=0.01)
    assert samples.shape == (10, 3)
    assert not np.any(np.isnan(samples))


def test_sliced_score_matching():
    # Score function for standard Gaussian N(0, I): s(x) = -x
    def true_gaussian_score(x: np.ndarray) -> np.ndarray:
        return -x

    ssm = SlicedScoreMatching(score_fn=true_gaussian_score, num_projections=4)
    rng = np.random.RandomState(42)
    X = rng.randn(50, 2)

    loss = ssm.compute_loss(X)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_m5_model_tree():
    rng = np.random.RandomState(42)
    X = rng.uniform(-2.0, 2.0, size=(100, 2))
    # Piecewise linear target
    y = np.where(
        X[:, 0] <= 0.0, 2.0 * X[:, 0] + 1.0, -3.0 * X[:, 0] + 1.0
    ) + 0.05 * rng.randn(100)

    tree = M5ModelTree(max_depth=4, min_samples_split=5, smoothing=5.0)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))
    # Measure R^2 correlation
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - preds) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    assert r2 > 0.85


def test_rulefit_regressor_and_classifier():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 4)
    y_reg = X[:, 0] * 2.0 + (X[:, 1] > 0.5) * 3.0 + 0.1 * rng.randn(80)

    # 1. RuleFit Regressor
    rf_reg = RuleFitRegressor(num_trees=5, max_depth=2, alpha=0.01, seed=42)
    rf_reg.fit(X, y_reg)
    preds_reg = rf_reg.predict(X)
    assert preds_reg.shape == (80,)

    # 2. RuleFit Classifier
    y_cls = (y_reg > np.median(y_reg)).astype(int)
    rf_cls = RuleFitClassifier(num_trees=5, max_depth=2, alpha=0.01, seed=42)
    rf_cls.fit(X, y_cls)
    probs = rf_cls.predict_proba(X)
    assert probs.shape == (80, 2)
    preds_cls = rf_cls.predict(X)
    assert preds_cls.shape == (80,)
    assert set(np.unique(preds_cls)).issubset({0, 1})
