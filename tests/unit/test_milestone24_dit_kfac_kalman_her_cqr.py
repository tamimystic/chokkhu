"""Unit tests for Milestone 24: Diffusion Transformer (DiT), K-FAC Optimizer,
Extended & Unscented Kalman Filters, Hindsight Experience Replay (HER), and Conformalized Quantile Regression (CQR).
"""

from __future__ import annotations

import numpy as np

from chokkhu.models.generative.dit import DiffusionTransformer, AdaLNZero
from chokkhu.optimization.kfac import KFAC
from chokkhu.models.timeseries.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from chokkhu.models.rl.her import HindsightExperienceReplay, GoalConditionedDQN
from chokkhu.uncertainty.cqr import ConformalizedQuantileRegression


# =====================================================================
# 1. Diffusion Transformer (DiT) & adaLN-Zero Modulation
# =====================================================================


def test_dit_patchify_unpatchify():
    dit = DiffusionTransformer(
        input_size=16, in_channels=4, patch_size=2, hidden_dim=32, depth=2
    )
    x = np.random.randn(2, 4, 16, 16)

    patches = dit._patchify(x)
    assert patches.shape == (2, 64, 16)  # 8x8 patches, each 4*2*2 = 16 dim

    x_reconstructed = dit._unpatchify(patches)
    np.testing.assert_allclose(x, x_reconstructed, atol=1e-6)


def test_dit_forward_and_sampling():
    dit = DiffusionTransformer(
        input_size=8,
        in_channels=2,
        patch_size=2,
        hidden_dim=32,
        depth=2,
        num_heads=2,
        num_timesteps=10,
        seed=42,
    )

    x = np.random.randn(2, 2, 8, 8)
    timesteps = np.array([3, 7])

    # 1. Forward noise prediction
    pred_noise = dit.forward(x, timesteps)
    assert pred_noise.shape == x.shape
    assert not np.isnan(pred_noise).any()

    # 2. Forward q_sample
    x_t, noise = dit.q_sample(x, timesteps)
    assert x_t.shape == x.shape
    assert noise.shape == x.shape

    # 3. Reverse sampling loop
    samples = dit.sample(shape=(2, 2, 8, 8), n_steps=3)
    assert samples.shape == (2, 2, 8, 8)
    assert not np.isnan(samples).any()


def test_adaln_zero_initialization():
    adaln = AdaLNZero(cond_dim=16, hidden_dim=32, seed=42)
    cond = np.random.randn(2, 16)

    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln.forward(cond)
    assert gamma1.shape == (2, 32)
    assert alpha1.shape == (2, 32)
    assert alpha2.shape == (2, 32)


# =====================================================================
# 2. K-FAC: Natural Gradient Second-Order Optimization
# =====================================================================


def test_kfac_factors_and_preconditioning():
    kfac = KFAC(lr=0.01, damping=1e-3, momentum=0.9)

    # Layer with in_dim=4, out_dim=3
    activations = np.random.randn(10, 4)
    pre_grads = np.random.randn(10, 3)

    kfac.update_factors("layer1", activations, pre_grads)
    assert "layer1" in kfac.A_factors
    assert "layer1" in kfac.S_factors
    assert kfac.A_factors["layer1"].shape == (4, 4)
    assert kfac.S_factors["layer1"].shape == (3, 3)

    # Precondition gradient
    grad_W = np.random.randn(3, 4)
    nat_grad = kfac.precondition_gradient("layer1", grad_W)
    assert nat_grad.shape == (3, 4)
    assert not np.isnan(nat_grad).any()


def test_kfac_step_update():
    kfac = KFAC(lr=0.05, damping=1e-2)

    W = np.ones((3, 4))
    grad_W = np.ones((3, 4)) * 0.1
    activations = np.ones((8, 4))
    pre_grads = np.ones((8, 3))

    kfac.update_factors("W1", activations, pre_grads)

    params = {"W1": W}
    grads = {"W1": grad_W}

    kfac.step(params, grads)
    # Weights must be updated
    assert not np.array_equal(W, np.ones((3, 4)))


# =====================================================================
# 3. Extended & Unscented Kalman Filters (EKF & UKF)
# =====================================================================


def test_extended_kalman_filter():
    # Non-linear 1D motion: x_t = x_{t-1} + sin(x_{t-1}) + u
    # Measurement: z_t = x_t^2
    def f(x, u):
        u_val = 0.0 if u is None else float(u[0])
        return np.array([x[0] + 0.1 * np.sin(x[0]) + u_val])

    def h(x):
        return np.array([x[0] ** 2])

    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1, f=f, h=h)
    ekf.x = np.array([2.0])

    # Predict and update
    x_pred, P_pred = ekf.predict()
    assert x_pred.shape == (1,)
    assert P_pred.shape == (1, 1)

    z_obs = np.array([4.1])  # measurement near 2^2
    x_up, P_up = ekf.update(z_obs)
    assert x_up.shape == (1,)
    assert not np.isnan(x_up).any()


def test_unscented_kalman_filter():
    # 2D non-linear state
    def f(x, u):
        return np.array([x[0] + 0.5 * x[1], x[1] * 0.9])

    def h(x):
        return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-6)])

    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, f=f, h=h)
    ukf.x = np.array([1.0, 1.0])

    x_pred, P_pred = ukf.predict()
    assert x_pred.shape == (2,)
    assert P_pred.shape == (2, 2)

    z_obs = np.array([1.5])
    x_up, P_up = ukf.update(z_obs)
    assert x_up.shape == (2,)
    assert not np.isnan(x_up).any()


# =====================================================================
# 4. Hindsight Experience Replay (HER) & Goal-Conditioned DQN
# =====================================================================


def test_her_buffer_and_goal_substitution():
    her = HindsightExperienceReplay(capacity=1000, strategy="future", k=2, seed=42)

    # Trajectory of 5 steps in 2D grid
    states = [np.array([float(i), float(i)]) for i in range(6)]
    actions = [0, 1, 0, 1, 0]
    desired_goal = np.array([10.0, 10.0])  # distant target

    her.add_episode(states, actions, desired_goal)

    # Buffer should have 5 original + 5*2 hindsight = 15 transitions
    assert len(her) == 15

    batch_s, batch_a, batch_r, batch_s_next, batch_done, batch_g = her.sample_batch(8)
    assert batch_s.shape == (8, 2)
    assert batch_a.shape == (8,)
    assert batch_g.shape == (8, 2)


def test_goal_conditioned_dqn():
    agent = GoalConditionedDQN(
        state_dim=2, goal_dim=2, num_actions=4, hidden_dim=16, seed=42
    )

    state = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])

    action = agent.select_action(state, goal)
    assert 0 <= action < 4

    # Train on HER buffer
    her = HindsightExperienceReplay(capacity=100, k=2, seed=42)
    states = [np.random.randn(2) for _ in range(5)]
    actions = [0, 1, 2, 3]
    her.add_episode(states, actions, goal)

    loss = agent.train_step(her, batch_size=4)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


# =====================================================================
# 5. Conformalized Quantile Regression (CQR)
# =====================================================================


def test_conformalized_quantile_regression():
    np.random.seed(42)
    # Synthetic heteroscedastic data: y = 2*x + (1 + |x|) * noise
    X = np.random.uniform(-3.0, 3.0, size=(200, 1))
    noise = np.random.randn(200, 1)
    y = 2.0 * X + (1.0 + np.abs(X)) * noise * 0.5

    # Split train, calib, test
    X_train, y_train = X[:100], y[:100]
    X_calib, y_calib = X[100:150], y[100:150]
    X_test, y_test = X[150:], y[150:]

    cqr = ConformalizedQuantileRegression(alpha=0.1)  # 90% target coverage
    cqr.fit(X_train, y_train)
    cqr.calibrate(X_calib, y_calib)

    lower, upper = cqr.predict_interval(X_test)
    assert lower.shape == (50,)
    assert upper.shape == (50,)
    assert (lower <= upper).all()

    metrics = cqr.evaluate_coverage(X_test, y_test)
    assert metrics["empirical_coverage"] >= 0.80  # Verified coverage near target 90%
    assert metrics["average_width"] > 0.0
