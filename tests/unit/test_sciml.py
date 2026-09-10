"""Unit tests for Scientific Machine Learning, PINNs & Symbolic Modeling (Milestone 6)."""

import numpy as np
import pytest
from chokkhu.models.sciml import (
    PINN,
    BurgersPINN,
    HeatPINN,
    WavePINN,
    HarmonicOscillatorPINN,
    SymbolicRegressor,
    NeuralODE,
)


def test_pinn_forward_and_derivatives():
    """Test base PINN forward pass and finite-difference spatial/temporal derivatives."""
    pinn = PINN(in_dim=2, out_dim=1, hidden_layers=[16, 16], activation="tanh", seed=42)
    tx = np.random.randn(8, 2).astype(np.float32)

    u = pinn.forward(tx)
    assert u.shape == (8, 1)

    u_val, u_t, u_x, u_xx = pinn.compute_derivatives_1d(tx, eps=1e-3)
    assert u_val.shape == (8, 1)
    assert u_t.shape == (8, 1)
    assert u_x.shape == (8, 1)
    assert u_xx.shape == (8, 1)


def test_burgers_pinn():
    """Test 1D viscous Burgers PINN fitting and residual evaluation."""
    burgers = BurgersPINN(nu=0.01, hidden_layers=[16, 16], lr=0.01, seed=42)

    tx_col = np.random.uniform(-1, 1, (20, 2)).astype(np.float32)
    tx_ic = np.column_stack([np.zeros(10), np.linspace(-1, 1, 10)]).astype(np.float32)
    u_ic = -np.sin(np.pi * tx_ic[:, [1]])
    tx_bc = np.column_stack([np.linspace(0, 1, 10), np.ones(10)]).astype(np.float32)
    u_bc = np.zeros((10, 1), dtype=np.float32)

    burgers.fit(tx_col, tx_ic, u_ic, tx_bc, u_bc, n_epochs=20)

    assert burgers.is_fitted
    assert len(burgers.loss_history_) == 20

    res = burgers.pde_residual(tx_col)
    assert res.shape == (20, 1)

    preds = burgers.predict(tx_col)
    assert preds.shape == (20, 1)


def test_heat_pinn():
    """Test 1D Heat diffusion equation PINN residual."""
    heat = HeatPINN(alpha=0.05, hidden_layers=[16, 16], seed=42)
    tx = np.random.randn(6, 2).astype(np.float32)
    res = heat.pde_residual(tx)

    assert res.shape == (6, 1)


def test_wave_pinn():
    """Test 1D acoustic Wave equation PINN residual."""
    wave = WavePINN(c=2.0, hidden_layers=[16, 16], seed=42)
    tx = np.random.randn(6, 2).astype(np.float32)
    res = wave.pde_residual(tx)

    assert res.shape == (6, 1)


def test_harmonic_oscillator_pinn():
    """Test damped harmonic oscillator ODE residual."""
    ho = HarmonicOscillatorPINN(zeta=0.1, w0=2.0, hidden_layers=[16, 16], seed=42)
    t = np.linspace(0, 5, 10).astype(np.float32)
    res = ho.ode_residual(t)

    assert res.shape == (10, 1)


def test_symbolic_regression_polynomial():
    """Test SymbolicRegressor (SINDy) on quadratic polynomial ground truth."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-2, 2, (100, 2))
    # True equation: y = 2.5 * x0^2 - 1.5 * x1 + 3.0
    y = 2.5 * (X[:, 0] ** 2) - 1.5 * X[:, 1] + 3.0

    model = SymbolicRegressor(threshold=0.2, degree=2, include_trig=False)
    model.fit(X, y)

    assert model.is_fitted
    assert model.coef_ is not None

    preds = model.predict(X)
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r2 > 0.99

    eq_str = model.equation()
    assert isinstance(eq_str, str)
    assert "x0^2" in eq_str or "x1" in eq_str


def test_symbolic_regression_trigonometric():
    """Test SymbolicRegressor on trigonometric ground truth."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-np.pi, np.pi, (80, 2))
    # True equation: y = 2.0 * sin(x0) + 1.0 * cos(x1)
    y = 2.0 * np.sin(X[:, 0]) + 1.0 * np.cos(X[:, 1])

    model = SymbolicRegressor(threshold=0.2, degree=1, include_trig=True)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds) ** 2)
    assert mse < 1e-4


def test_neural_ode_integration():
    """Test NeuralODE integration using RK4 and Euler solvers."""
    node_rk4 = NeuralODE(dim=2, hidden_dim=16, solver="rk4", seed=42)
    node_euler = NeuralODE(dim=2, hidden_dim=16, solver="euler", seed=42)

    h0 = np.array([1.0, 0.0], dtype=np.float32)
    t_span = np.linspace(0.0, 1.0, 10).astype(np.float32)

    traj_rk4 = node_rk4.forward(h0, t_span)
    traj_euler = node_euler.forward(h0, t_span)

    assert traj_rk4.shape == (10, 2)
    assert traj_euler.shape == (10, 2)
    assert np.all(np.isfinite(traj_rk4))
    assert np.all(np.isfinite(traj_euler))

    # Batch integration
    h0_batch = np.random.randn(4, 2).astype(np.float32)
    traj_batch = node_rk4.forward(h0_batch, t_span)
    assert traj_batch.shape == (4, 10, 2)


def test_sciml_exceptions():
    """Test exception handling in SciML models."""
    sr = SymbolicRegressor()
    with pytest.raises(RuntimeError):
        sr.predict(np.zeros((5, 2)))
    with pytest.raises(RuntimeError):
        sr.equation()
