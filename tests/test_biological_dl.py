"""Unit tests for Frontier 4.23: Biologically Plausible Learning, DFA, DTP & Synthetic Gradients."""

from __future__ import annotations

import numpy as np

from chokkhu.models.dl.biological import (
    DirectFeedbackAlignmentNetwork,
    DifferenceTargetPropagationNetwork,
    DecoupledSyntheticGradientLayer,
)


def test_dfa_weight_updates_and_loss_reduction():
    """Verify Direct Feedback Alignment updates all layers via fixed random feedback matrices and reduces loss."""
    rng = np.random.default_rng(42)
    # Generate non-linear dataset
    X = rng.standard_normal((32, 4))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2]) * 0.5

    dfa_net = DirectFeedbackAlignmentNetwork(
        layer_dims=[4, 16, 8, 1],
        activation="tanh",
        lr=0.05,
        random_state=42,
    )

    initial_loss = float(0.5 * np.mean((dfa_net.predict(X) - y) ** 2))

    # Train for 50 DFA steps
    losses = []
    for _ in range(50):
        loss = dfa_net.train_step(X, y)
        losses.append(loss)

    final_loss = losses[-1]
    assert final_loss < initial_loss
    # Check predictions shape
    preds = dfa_net.predict(X)
    assert preds.shape == (32, 1)
    assert not np.isnan(preds).any()


def test_difference_target_propagation_learning():
    """Verify Difference Target Propagation propagates layer-wise target states and minimizes task error."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((24, 3))
    y = np.tanh(X[:, 0:1] + X[:, 1:2])

    dtp_net = DifferenceTargetPropagationNetwork(
        layer_dims=[3, 16, 8, 1],
        lr_forward=0.1,
        lr_feedback=0.1,
        noise_std=0.01,
        random_state=42,
    )

    initial_loss = float(0.5 * np.mean((dtp_net.forward(X)[0] - y) ** 2))

    losses = []
    for _ in range(60):
        loss = dtp_net.train_step(X, y)
        losses.append(loss)

    assert min(losses) < initial_loss
    y_pred, acts = dtp_net.forward(X)
    assert y_pred.shape == (24, 1)
    assert len(acts) == 4  # input + 3 layers
    assert not np.isnan(y_pred).any()


def test_decoupled_synthetic_gradients_async_updates():
    """Verify Decoupled Neural Interfaces predict synthetic gradients and perform asynchronous weight updates."""
    dni_layer = DecoupledSyntheticGradientLayer(
        in_dim=4,
        out_dim=8,
        lr=0.01,
        lr_synth=0.02,
        random_state=42,
    )

    X = np.random.randn(10, 4)
    initial_W = dni_layer.W.copy()

    # 1. Forward pass produces activations and synthetic gradient
    h_out, synth_grad = dni_layer.forward(X)
    assert h_out.shape == (10, 8)
    assert synth_grad.shape == (10, 8)

    # 2. Layer updates immediately using synthetic gradient
    dni_layer.update_with_synthetic_grad(X, h_out, synth_grad)
    assert not np.allclose(dni_layer.W, initial_W)

    # 3. Train gradient synthesizer against mock downstream gradient
    target_grad = np.random.randn(10, 8)
    synth_loss = dni_layer.update_synthesizer(h_out, target_grad)
    assert synth_loss > 0.0
