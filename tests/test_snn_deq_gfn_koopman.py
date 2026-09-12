"""Unit tests for SNN (4.6), DEQ (4.6), GFlowNets (4.7), and Dynamic Mode Decomposition (Phase 22)."""

from __future__ import annotations

import numpy as np
from chokkhu.models.dl.spiking import SpikingNeuralNetwork
from chokkhu.models.dl.deq import DeepEquilibriumModel
from chokkhu.models.generative.gflownet import TrajectoryBalanceGFlowNet
from chokkhu.models.sciml.koopman import DynamicModeDecomposition, ExtendedDMD


def test_spiking_neural_network_forward_and_bptt():
    """Verify SNN integrates membrane potential, emits spikes, and trains via surrogate gradients."""
    rng = np.random.default_rng(42)
    N = 80
    X = rng.standard_normal((N, 4))
    y = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)

    snn = SpikingNeuralNetwork(
        layer_sizes=[4, 16, 2],
        num_classes=2,
        time_steps=8,
        tau_m=10.0,
        v_threshold=0.5,
        learning_rate=0.02,
        random_state=42,
    )

    history = snn.fit(X, y, epochs=15, batch_size=20)
    assert len(history) == 15
    assert history[-1] < history[0]

    rates, spikes, volts = snn.forward(X[:5], time_steps=8)
    assert rates.shape == (5, 2)
    assert len(spikes) == 2
    assert spikes[0].shape == (8, 5, 16)

    preds = snn.predict(X[:10])
    assert preds.shape == (10,)


def test_deep_equilibrium_model_fixed_point_and_ift():
    """Verify DEQ solves fixed-point equilibrium and updates parameters via IFT."""
    rng = np.random.default_rng(42)
    N = 60
    X = rng.standard_normal((N, 4))
    y = (X[:, 0] > 0.0).astype(np.int64)

    deq = DeepEquilibriumModel(
        input_dim=4,
        hidden_dim=16,
        num_classes=2,
        max_iter=30,
        tol=1e-4,
        learning_rate=0.01,
        random_state=42,
    )

    history = deq.fit(X, y, epochs=12, batch_size=20)
    assert len(history) == 12

    probs, z_star = deq.forward(X[:5])
    assert probs.shape == (5, 2)
    assert z_star.shape == (5, 16)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_gflownet_trajectory_balance_optimization():
    """Verify GFlowNet optimizes Trajectory Balance objective to sample high-reward objects."""

    # Simple binary reward: reward count of active bits
    def reward_fn(states: np.ndarray) -> np.ndarray:
        return np.sum(states, axis=1) + 0.1

    gfn = TrajectoryBalanceGFlowNet(
        state_dim=6,
        action_dim=6,
        hidden_dim=24,
        horizon=4,
        learning_rate=0.01,
        random_state=42,
    )

    losses = gfn.fit(reward_fn, num_iterations=40, batch_size=16)
    assert len(losses) == 40

    samples = gfn.sample(num_samples=20)
    assert samples.shape == (20, 6)
    assert np.all((samples == 0.0) | (samples == 1.0))


def test_dynamic_mode_decomposition_exact_dmd():
    """Verify Exact DMD extracts modes and reconstructs dynamical spatiotemporal signals."""
    # Synthetic 2-mode spatio-temporal dynamics: x(r, t) = f1(r) * exp(i*w1*t) + f2(r) * exp(i*w2*t)
    t = np.linspace(0, 4 * np.pi, 50)
    x = np.linspace(-5, 5, 20)
    T, X = np.meshgrid(t, x)

    f1 = (1.0 / np.cosh(X)) * np.exp(1.5j * T)
    f2 = (1.0 / np.cosh(X - 2)) * np.exp(2.5j * T)
    data = f1 + f2  # [20, 50]

    dmd = DynamicModeDecomposition(rank=2, dt=t[1] - t[0])
    dmd.fit(data)

    assert dmd.modes is not None
    assert dmd.modes.shape == (20, 2)
    assert dmd.eigenvalues is not None
    assert len(dmd.eigenvalues) == 2

    recon = dmd.reconstruct(num_steps=50)
    assert recon.shape == (20, 50)
    # Reconstruction should accurately mirror the real part of input data
    np.testing.assert_allclose(recon, np.real(data), atol=1e-1)


def test_extended_dmd_koopman_forward_step():
    """Verify Extended DMD Koopman operator predicts non-linear step transitions."""
    # Non-linear 2D oscillator: dx/dt = -0.1*x - y, dy/dt = x - 0.1*y - 0.05*x^2
    steps = 80
    traj = np.zeros((2, steps), dtype=np.float64)
    traj[:, 0] = [1.0, 0.5]
    dt = 0.05

    for step in range(steps - 1):
        x_val, y_val = traj[0, step], traj[1, step]
        dx = -0.1 * x_val - y_val
        dy = x_val - 0.1 * y_val - 0.05 * (x_val**2)
        traj[0, step + 1] = x_val + dx * dt
        traj[1, step + 1] = y_val + dy * dt

    edmd = ExtendedDMD(polynomial_degree=2)
    edmd.fit(traj)

    assert edmd.K is not None
    # Predict next step from initial condition
    pred_next = edmd.predict_step(traj[:, 0])
    assert pred_next.shape == (2,)
    np.testing.assert_allclose(pred_next, traj[:, 1], atol=1e-2)
