"""Unit tests for Frontier 4.20: Consistency Models, Optimal Transport Flow Matching & Reflow."""

import numpy as np
from chokkhu.models.generative.consistency import (
    ConsistencyModel,
    OptimalTransportFlowMatching,
    ReflowMatching,
)


def test_consistency_boundary_condition():
    """Verify exact boundary condition f_theta(x, eps) == x for arbitrary random weights and inputs."""
    model = ConsistencyModel(
        input_dim=4, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5
    )

    rng = np.random.default_rng(42)
    x = rng.standard_normal((10, 4)).astype(np.float32)

    # At t = sigma_min = 0.002, c_skip = 1.0 and c_out = 0.0
    c_sk = model.c_skip(0.002)
    c_o = model.c_out(0.002)
    np.testing.assert_allclose(c_sk, 1.0, atol=1e-6)
    np.testing.assert_allclose(c_o, 0.0, atol=1e-6)

    output = model.forward(x, t=0.002)
    np.testing.assert_allclose(output, x, atol=1e-5)


def test_consistency_ema_update():
    """Verify target network parameters update via Exponential Moving Average (EMA)."""
    model = ConsistencyModel(input_dim=3, ema_decay=0.9, random_state=42)

    initial_target_w0 = model.target_net.weights[0].copy()
    initial_online_w0 = model.net.weights[0].copy()
    np.testing.assert_allclose(initial_target_w0, initial_online_w0)

    # Perform a training step
    x_dummy = np.random.randn(8, 3).astype(np.float32)
    loss = model.train_step(x_dummy)
    assert loss > 0.0

    # Online weights changed
    assert not np.allclose(model.net.weights[0], initial_online_w0)
    # Target weights updated via EMA
    expected_target = 0.9 * initial_target_w0 + 0.1 * model.net.weights[0]
    np.testing.assert_allclose(model.target_net.weights[0], expected_target, atol=1e-5)


def test_consistency_1step_and_multistep_sampling():
    """Verify 1-step direct and multi-step iterative consistency generation."""
    model = ConsistencyModel(
        input_dim=5, sigma_min=0.002, sigma_max=10.0, random_state=42
    )

    # 1-step sampling
    samples_1step = model.sample(num_samples=12, steps=1)
    assert samples_1step.shape == (12, 5)
    assert not np.isnan(samples_1step).any()

    # Multi-step sampling (e.g. 5 steps)
    samples_multistep = model.sample(num_samples=8, steps=5)
    assert samples_multistep.shape == (8, 5)
    assert not np.isnan(samples_multistep).any()


def test_optimal_transport_bipartite_assignment():
    """Verify Earth Mover / Hungarian bipartite assignment minimizes pair-wise transport cost."""
    ot_flow = OptimalTransportFlowMatching(input_dim=2, random_state=42)

    x0 = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    x1 = np.array([[10.1, 9.9], [0.1, 0.1]], dtype=np.float32)

    # Without OT, pairing distance is:
    # dist([0,0], [10.1, 9.9])^2 + dist([10,10], [0.1, 0.1])^2 = 200 + 196 = ~396
    # With OT, pairing aligns [0,0]->[0.1,0.1] and [10,10]->[10.1,9.9] = ~0.04
    x0_ot, x1_ot = ot_flow.compute_ot_pairing(x0, x1)

    np.testing.assert_allclose(x0_ot[0], [0.0, 0.0])
    np.testing.assert_allclose(x1_ot[0], [0.1, 0.1], atol=1e-5)
    np.testing.assert_allclose(x0_ot[1], [10.0, 10.0])
    np.testing.assert_allclose(x1_ot[1], [10.1, 9.9], atol=1e-5)


def test_ot_flow_matching_training_and_ode_sampling():
    """Verify OT-CFM training steps and ODE solvers (Euler, Midpoint, RK4)."""
    ot_flow = OptimalTransportFlowMatching(input_dim=3, lr=0.01, random_state=42)

    x_data = np.random.randn(16, 3).astype(np.float32)
    loss = ot_flow.train_step(x_data)
    assert loss > 0.0

    # Test ODE sampling with all 3 solvers
    s_euler = ot_flow.sample(num_samples=6, steps=10, method="euler")
    assert s_euler.shape == (6, 3)
    assert not np.isnan(s_euler).any()

    s_midpoint = ot_flow.sample(num_samples=6, steps=10, method="midpoint")
    assert s_midpoint.shape == (6, 3)
    assert not np.isnan(s_midpoint).any()

    s_rk4 = ot_flow.sample(num_samples=6, steps=10, method="rk4")
    assert s_rk4.shape == (6, 3)
    assert not np.isnan(s_rk4).any()


def test_reflow_matching_straightening_and_fast_sampling():
    """Verify Reflow paired dataset generation and ultra-fast 1-step / 2-step sampling."""
    base_flow = OptimalTransportFlowMatching(input_dim=2, random_state=42)
    reflow = ReflowMatching(base_flow=base_flow, lr=0.01, random_state=42)

    # Generate paired dataset from base flow
    x0_pairs, x1_pairs = reflow.generate_paired_dataset(num_pairs=20, ode_steps=10)
    assert x0_pairs.shape == (20, 2)
    assert x1_pairs.shape == (20, 2)

    # Train reflow step
    loss = reflow.train_reflow_step(x0_pairs, x1_pairs)
    assert loss > 0.0

    # Fast 1-step sampling
    fast_samples = reflow.sample_fast(num_samples=10, steps=1)
    assert fast_samples.shape == (10, 2)
    assert not np.isnan(fast_samples).any()
