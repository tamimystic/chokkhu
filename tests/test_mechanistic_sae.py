"""Unit tests for Frontier 4.18: Mechanistic Interpretability & Sparse Autoencoders."""

import numpy as np

from chokkhu.explainability.mechanistic import (
    DirectLogitAttribution,
    TopKSAE,
    JumpReLU,
    ActivationPatching,
)


def test_topk_sae_sparsity_enforcement():
    """Verify TopKSAE retains exactly k active features per sample."""
    sae = TopKSAE(d_in=8, d_sae=32, k=4, seed=42)
    x = np.random.randn(10, 8)
    f = sae.encode(x)

    assert f.shape == (10, 32)
    # Count non-zero elements per row
    non_zeros = np.count_nonzero(f, axis=1)
    # Exactly k non-zeros (or <= k if some activations were <= 0)
    assert np.all(non_zeros <= 4)
    assert np.any(non_zeros == 4)


def test_topk_sae_unit_norm_decoder_columns():
    """Verify decoder dictionary vectors maintain unit norm."""
    sae = TopKSAE(d_in=16, d_sae=64, k=8, seed=42)
    norms = np.linalg.norm(sae.W_dec, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_topk_sae_reconstruction_and_fit():
    """Verify TopKSAE fits on data and reduces reconstruction error."""
    rng = np.random.default_rng(42)
    # Low-rank synthetic activations
    basis = rng.standard_normal((4, 8))
    coeffs = rng.standard_normal((100, 4))
    X = coeffs @ basis

    sae = TopKSAE(d_in=8, d_sae=16, k=4, seed=42)
    x_hat_init, _ = sae.forward(X)
    loss_init = np.mean((x_hat_init - X) ** 2)

    sae.fit(X, epochs=40, lr=5e-3, batch_size=25)
    x_hat_post, f_post = sae.forward(X)
    loss_post = np.mean((x_hat_post - X) ** 2)

    assert loss_post < loss_init
    assert x_hat_post.shape == X.shape
    assert f_post.shape == (100, 16)


def test_jumprelu_sae_thresholding():
    """Verify JumpReLU strictly gates activations below threshold."""
    sae = JumpReLU(d_in=4, d_sae=8, threshold=0.5, seed=42)
    x = np.array([1.0, 2.0, -1.0, 0.0])
    f = sae.encode(x)

    assert f.shape == (8,)
    # All non-zero entries must be strictly > 0.5
    active_values = f[f > 0]
    if len(active_values) > 0:
        assert np.all(active_values > 0.5)


def test_activation_patching_indirect_effect():
    """Verify activation patching indirect effect restoration ratio computation."""
    clean_score = 10.0
    corrupted_score = 2.0
    patched_score = 8.0

    ie = ActivationPatching.compute_indirect_effect(
        clean_score, corrupted_score, patched_score
    )
    # (8 - 2) / (10 - 2) = 6 / 8 = 0.75
    np.testing.assert_allclose(ie, 0.75)


def test_activation_patching_tensor_substitution():
    """Verify selective activation tensor patching with boolean mask."""
    corrupted = np.array([[1.0, 1.0], [1.0, 1.0]])
    clean = np.array([[9.0, 9.0], [9.0, 9.0]])
    mask = np.array([[True, False], [False, True]])

    patched = ActivationPatching.patch_activation(corrupted, clean, patch_mask=mask)
    expected = np.array([[9.0, 1.0], [1.0, 9.0]])
    np.testing.assert_allclose(patched, expected)


def test_direct_logit_attribution():
    """Verify DirectLogitAttribution projects hidden states into vocabulary logits."""
    # Hidden dim 4, Vocab size 3
    W_U = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    dla = DirectLogitAttribution(W_U)
    h = np.array([1.0, 1.0, 1.0, 1.0])
    logits = dla.attribute(h)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(logits, expected)

    # Specific token target
    target_score = dla.attribute(h, target_token_id=1)
    assert target_score == 3.0
