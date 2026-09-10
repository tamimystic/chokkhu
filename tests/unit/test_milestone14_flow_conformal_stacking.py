"""Unit Tests for Milestone 14: Flow Matching, Normalizing Flows, LoRA, Conformal Prediction, Matrix Profile & Stacking."""

import numpy as np

from chokkhu.models.generative import (
    FlowMatching,
    RectifiedFlow,
    VelocityMLP,
    RealNVP,
    AffineCouplingLayer,
    LoRALinear,
    LoRAAdapter,
)
from chokkhu.models.timeseries import (
    ConformalPredictor,
    conformal_interval,
    MatrixProfile,
    find_motifs,
    find_discords,
)
from chokkhu.automl import (
    BOHB,
    SuperLearner,
    StackingEnsemble,
)
from chokkhu.models.ml import (
    LinearRegression,
    LogisticRegression,
    RandomForest,
    KNN,
)


def test_velocity_mlp_and_flow_matching():
    """Test FlowMatching velocity field learning and ODE numerical integration."""
    assert RectifiedFlow is FlowMatching
    v_mlp = VelocityMLP(input_dim=4, hidden_dims=[16])
    assert v_mlp.input_dim == 4

    np.random.seed(42)
    x = np.random.randn(20, 4).astype(np.float32)

    fm = FlowMatching(input_dim=4, hidden_dims=[32, 32], lr=0.01, random_state=42)
    loss, xt, t, ut = fm.compute_loss(x)
    assert isinstance(loss, float)
    assert xt.shape == (20, 4)
    assert ut.shape == (20, 4)

    # Train step
    loss_val = fm.train_step(x)
    assert isinstance(loss_val, float)
    assert not np.isnan(loss_val)

    # Sample using Euler, Midpoint, and RK4 ODE solvers
    samples_euler = fm.sample(num_samples=5, steps=5, method="euler")
    samples_mid = fm.sample(num_samples=5, steps=5, method="midpoint")
    samples_rk4 = fm.sample(num_samples=5, steps=5, method="rk4")

    assert samples_euler.shape == (5, 4)
    assert samples_mid.shape == (5, 4)
    assert samples_rk4.shape == (5, 4)
    assert not np.any(np.isnan(samples_rk4))


def test_real_nvp_normalizing_flows():
    """Test RealNVP invertible forward, inverse reconstruction, and log probability."""
    layer = AffineCouplingLayer(dim=6, hidden_dim=16)
    assert layer.dim == 6

    np.random.seed(42)
    x = np.random.randn(15, 6).astype(np.float32)

    flow = RealNVP(dim=6, num_layers=4, hidden_dim=32, random_state=42)
    z, log_det = flow.forward(x)
    assert z.shape == (15, 6)
    assert log_det.shape == (15,)

    # Test exact invertibility: x == inverse(forward(x))
    x_rec = flow.inverse(z)
    np.testing.assert_allclose(x, x_rec, rtol=1e-4, atol=1e-4)

    # Test log-likelihood computation
    log_probs = flow.log_prob(x)
    assert log_probs.shape == (15,)
    assert not np.any(np.isnan(log_probs))

    # Test sampling from prior
    samples = flow.sample(num_samples=8)
    assert samples.shape == (8, 6)
    assert not np.any(np.isnan(samples))


def test_lora_linear_and_adapter():
    """Test Low-Rank Adaptation (LoRA) layer, weight merging, unmerging, and counting."""
    np.random.seed(42)
    x = np.random.randn(8, 16).astype(np.float32)

    lora = LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=16.0, lora_dropout=0.1, random_state=42)
    out_unmerged = lora.forward(x, training=False)
    assert out_unmerged.shape == (8, 8)

    # Trainable parameters check
    params = lora.get_trainable_params()
    assert len(params) == 2
    assert params[0].shape == (16, 4)
    assert params[1].shape == (4, 8)

    # Test merging
    lora.merge()
    assert lora.merged is True
    out_merged = lora.forward(x, training=False)
    np.testing.assert_allclose(out_unmerged, out_merged, rtol=1e-4, atol=1e-4)

    # Test unmerging
    lora.unmerge()
    assert lora.merged is False
    out_restored = lora.forward(x, training=False)
    np.testing.assert_allclose(out_unmerged, out_restored, rtol=1e-4, atol=1e-4)

    # Parameter counting
    stats = LoRAAdapter.count_parameters([lora])
    assert stats["trainable_params"] == (16 * 4 + 4 * 8)
    assert stats["trainable_percent"] > 0


def test_conformal_predictor():
    """Test ConformalPredictor calibration, prediction intervals, and empirical coverage."""
    np.random.seed(42)
    X_train = np.random.randn(80, 3)
    y_train = 2.0 * X_train[:, 0] - 1.5 * X_train[:, 1] + np.random.randn(80) * 0.1

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    X_cal = np.random.randn(40, 3)
    y_cal = 2.0 * X_cal[:, 0] - 1.5 * X_cal[:, 1] + np.random.randn(40) * 0.1

    cp = ConformalPredictor(base_estimator=reg, alpha=0.1)
    q_hat = cp.calibrate(X_cal, y_cal)
    assert q_hat > 0

    X_test = np.random.randn(30, 3)
    y_test = 2.0 * X_test[:, 0] - 1.5 * X_test[:, 1] + np.random.randn(30) * 0.1

    preds, lower, upper = cp.predict_interval(X_test)
    assert preds.shape == (30,)
    assert lower.shape == (30,)
    assert upper.shape == (30,)
    assert np.all(upper >= lower)

    # Coverage metrics
    cov_metrics = cp.evaluate_coverage(X_test, y_test)
    assert cov_metrics["empirical_coverage"] >= 0.75

    # 1-Line helper
    l_bound, u_bound = conformal_interval(y_cal, reg.predict(X_cal), reg.predict(X_test), alpha=0.1)
    assert len(l_bound) == 30


def test_matrix_profile_motifs_and_discords():
    """Test fast MatrixProfile computation, motif pair extraction, and discord detection."""
    np.random.seed(42)
    # Generate time series with repeating motif and an injected spike discord
    t = np.linspace(0, 8 * np.pi, 200)
    ts = np.sin(t) + np.random.randn(200) * 0.05
    # Inject an anomaly spike at index 150
    ts[150:155] += 5.0

    mp = MatrixProfile(window_size=15, exclusion_zone=0.5)
    dist_prof, p_idx = mp.compute(ts)
    assert len(dist_prof) == 200 - 15 + 1
    assert len(p_idx) == 200 - 15 + 1
    assert not np.any(np.isnan(dist_prof))

    # Motifs & Discords
    motifs = mp.find_motifs(top_k=2)
    discords = mp.find_discords(top_k=2)

    assert len(motifs) == 2
    assert len(discords) == 2
    assert discords[0]["distance"] > motifs[0]["distance"]

    # 1-Line helpers
    motifs_h = find_motifs(ts, window_size=15, top_k=1)
    discords_h = find_discords(ts, window_size=15, top_k=1)
    assert len(motifs_h) == 1
    assert len(discords_h) == 1


def test_bohb_hyperparameter_tuner():
    """Test BOHB multi-fidelity successive halving optimization."""
    def toy_eval(config: dict, budget: float) -> float:
        x, y = config["x"], config["y"]
        # Maximize negative quadratic + budget scaling
        score = -(x - 1.0) ** 2 - (y + 2.0) ** 2 + float(budget) * 0.01
        return float(score)

    bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
    bohb = BOHB(
        eval_func=toy_eval,
        param_bounds=bounds,
        min_budget=1.0,
        max_budget=9.0,
        eta=3,
        random_state=42,
    )
    best_config, best_score = bohb.optimize()
    assert isinstance(best_config, dict)
    assert "x" in best_config and "y" in best_config
    assert len(bohb.history) > 0


def test_super_learner_stacking_classifier():
    """Test SuperLearner stacking ensemble for classification."""
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    base_models = [
        LogisticRegression(),
        RandomForest(n_estimators=10, max_depth=3, random_state=42),
        KNN(n_neighbors=3),
    ]
    meta_model = LogisticRegression()

    stacker = SuperLearner(
        estimators=base_models,
        meta_estimator=meta_model,
        task="classification",
        cv=3,
        random_state=42,
    )
    stacker.fit(X, y)

    preds = stacker.predict(X[:10])
    probs = stacker.predict_proba(X[:10])
    assert preds.shape == (10,)
    assert probs.shape == (10, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_super_learner_stacking_regressor():
    """Test SuperLearner stacking ensemble for regression with constrained weight blending."""
    np.random.seed(42)
    X = np.random.randn(80, 4).astype(np.float32)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(80) * 0.1

    base_models = [
        LinearRegression(),
        RandomForest(n_estimators=10, max_depth=3, task="regression", random_state=42),
        KNN(n_neighbors=3, task="regression"),
    ]

    stacker = StackingEnsemble(
        estimators=base_models,
        meta_estimator=None,  # Tests non-negative weight blending
        task="regression",
        cv=3,
        random_state=42,
    )
    stacker.fit(X, y)

    preds = stacker.predict(X[:10])
    assert preds.shape == (10,)
    assert not np.any(np.isnan(preds))
