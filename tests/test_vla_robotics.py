"""Unit tests for Frontier 4.22: Vision-Language-Action (VLA) & Action Chunking with Transformers."""

from __future__ import annotations

import numpy as np

from chokkhu.models.robotics.vla import (
    ActionTokenizer,
    TemporalEnsembler,
    ActionChunkingTransformer,
    OpenVLAPolicy,
)


def test_action_tokenizer_roundtrip_quantization():
    """Verify ActionTokenizer converts continuous 7-DoF robot actions to discrete tokens and back with minimal bin error."""
    tokenizer = ActionTokenizer(
        num_bins=256, min_action=-1.0, max_action=1.0, action_dim=7
    )

    # Continuous action vector
    original_action = np.array(
        [-0.85, 0.42, 0.0, -0.12, 0.99, -0.99, 1.0], dtype=np.float64
    )

    tokens = tokenizer.encode(original_action)
    assert tokens.shape == (7,)
    assert tokens.dtype == np.int64
    assert np.all(tokens >= 0)
    assert np.all(tokens < 256)

    reconstructed = tokenizer.decode(tokens)
    assert reconstructed.shape == (7,)

    # Max quantization error should not exceed half bin width: 2.0 / 256 / 2 = 0.0039
    max_err = np.max(np.abs(original_action - reconstructed))
    assert max_err <= (2.0 / 256.0)


def test_temporal_ensembler_smoothing():
    """Verify TemporalEnsembler aggregates overlapping predictions with exponential decay weights."""
    ensembler = TemporalEnsembler(chunk_size=4, action_dim=2, exp_weight=0.5)

    # Step 0: predict chunk [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    chunk_0 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    act_0 = ensembler.update(chunk_0)
    np.testing.assert_allclose(act_0, [1.0, 1.0])

    # Step 1: predict chunk [[2.2, 2.2], [3.2, 3.2], [4.2, 4.2], [5.2, 5.2]]
    # Overlapping predictions at step 1: chunk_0[1]=[2.0, 2.0] (weight exp(-0.5)) and chunk_1[0]=[2.2, 2.2] (weight 1.0)
    chunk_1 = np.array([[2.2, 2.2], [3.2, 3.2], [4.2, 4.2], [5.2, 5.2]])
    act_1 = ensembler.update(chunk_1)

    w0 = np.exp(-0.5)
    w1 = 1.0
    expected_act_1 = (w0 * np.array([2.0, 2.0]) + w1 * np.array([2.2, 2.2])) / (w0 + w1)
    np.testing.assert_allclose(act_1, expected_act_1, atol=1e-5)


def test_action_chunking_transformer_forward_and_step():
    """Verify ActionChunkingTransformer predicts multi-step action trajectories and closed-loop steps."""
    act = ActionChunkingTransformer(
        proprio_dim=4,
        visual_dim=16,
        action_dim=4,
        chunk_size=8,
        hidden_dim=32,
        random_state=42,
    )

    vis = np.random.randn(2, 16)
    prop = np.random.randn(2, 4)

    # Batch chunk prediction
    chunks = act.forward_chunk(vis, prop)
    assert chunks.shape == (2, 8, 4)
    assert not np.isnan(chunks).any()

    # Closed-loop single step with temporal ensembling
    vis_single = np.random.randn(16)
    prop_single = np.random.randn(4)

    step_action = act.step(vis_single, prop_single)
    assert step_action.shape == (4,)
    assert not np.isnan(step_action).any()


def test_open_vla_policy_end_to_end():
    """Verify OpenVLAPolicy end-to-end multi-modal vision-language to action execution."""
    vla = OpenVLAPolicy(
        vocab_size=100,
        action_bins=256,
        embed_dim=32,
        action_dim=7,
        chunk_size=6,
        random_state=42,
    )

    # 4 visual patch tokens of dim 32
    visual_tokens = np.random.randn(4, 32)
    # Instruction token IDs (e.g. "pick up the red mug")
    instruction_ids = np.array([5, 12, 44, 9], dtype=np.int64)
    # Current 7-DoF robot joint state
    proprio_state = np.array([0.1, -0.2, 0.5, 0.0, 0.1, -0.1, 1.0], dtype=np.float64)

    cont_act, disc_tokens = vla.predict_action(
        visual_tokens, instruction_ids, proprio_state
    )

    assert cont_act.shape == (7,)
    assert disc_tokens.shape == (7,)
    assert disc_tokens.dtype == np.int64
    assert np.all(disc_tokens >= 0)
    assert np.all(disc_tokens < 256)
