"""Unit tests for Multi-Modal Vision-Language Alignment (Milestone 5)."""

import numpy as np
import pytest
from chokkhu.models.multimodal import (
    CLIP,
    CLIPVisionEncoder,
    CLIPTextEncoder,
    SigLIP,
    LLaVALinearProjector,
    LLaVAMLPProjector,
    PerceiverResampler,
)


def test_clip_vision_encoder():
    """Test CLIPVisionEncoder image tokenization and normalized feature projection."""
    encoder = CLIPVisionEncoder(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        projection_dim=32,
        num_heads=2,
        num_layers=1,
        seed=42,
    )
    # Batch of 4 RGB images
    images = np.random.randn(4, 3, 32, 32).astype(np.float32)
    features = encoder.forward(images)

    assert features.shape == (4, 32)
    norms = np.linalg.norm(features, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_clip_text_encoder():
    """Test CLIPTextEncoder token processing and normalized representation."""
    encoder = CLIPTextEncoder(
        vocab_size=100,
        max_seq_len=16,
        embed_dim=64,
        projection_dim=32,
        num_heads=2,
        num_layers=1,
        seed=42,
    )
    # Batch of 4 text sequences
    tokens = np.random.randint(0, 100, size=(4, 16), dtype=np.int32)
    features = encoder.forward(tokens)

    assert features.shape == (4, 32)
    norms = np.linalg.norm(features, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_clip_forward_and_loss():
    """Test full CLIP dual-encoder forward pass and contrastive InfoNCE loss."""
    model = CLIP(
        embed_dim=32,
        image_size=32,
        patch_size=8,
        vocab_size=100,
        init_temperature=0.07,
        seed=42,
    )
    images = np.random.randn(3, 3, 32, 32).astype(np.float32)
    tokens = np.random.randint(0, 100, size=(3, 12), dtype=np.int32)

    logits_i, logits_t, loss = model.forward(images, tokens)

    assert logits_i.shape == (3, 3)
    assert logits_t.shape == (3, 3)
    assert np.allclose(logits_i, logits_t.T)
    assert loss > 0.0
    assert not np.isnan(loss)


def test_clip_zero_shot_classification():
    """Test CLIP zero-shot classification across candidate prompt classes."""
    model = CLIP(
        embed_dim=32,
        image_size=32,
        patch_size=8,
        vocab_size=100,
        seed=42,
    )
    images = np.random.randn(2, 3, 32, 32).astype(np.float32)
    candidate_tokens = np.random.randint(
        0, 100, size=(5, 8), dtype=np.int32
    )  # 5 classes

    probs = model.predict_proba(images, candidate_tokens)

    assert probs.shape == (2, 5)
    assert np.all(probs >= 0.0)
    assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5)


def test_siglip_forward_and_loss():
    """Test SigLIP pairwise sigmoid loss and zero-shot prediction."""
    siglip = SigLIP(
        embed_dim=32,
        image_size=32,
        patch_size=8,
        vocab_size=100,
        init_temp=10.0,
        init_bias=-5.0,
        seed=42,
    )
    images = np.random.randn(4, 3, 32, 32).astype(np.float32)
    tokens = np.random.randint(0, 100, size=(4, 10), dtype=np.int32)

    logits, loss = siglip.forward(images, tokens)

    assert logits.shape == (4, 4)
    assert loss > 0.0
    assert not np.isnan(loss)

    # Zero shot sigmoid probabilities
    candidate_tokens = np.random.randint(0, 100, size=(3, 8), dtype=np.int32)
    probs = siglip.predict_proba(images, candidate_tokens)

    assert probs.shape == (4, 3)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_llava_linear_projector():
    """Test LLaVALinearProjector dimension projection."""
    proj = LLaVALinearProjector(vision_dim=64, text_dim=128, seed=42)
    vision_feats = np.random.randn(2, 16, 64).astype(np.float32)
    out = proj.forward(vision_feats)

    assert out.shape == (2, 16, 128)


def test_llava_mlp_projector():
    """Test LLaVAMLPProjector with GELU activation."""
    proj = LLaVAMLPProjector(vision_dim=64, text_dim=128, mlp_dim=128, seed=42)
    vision_feats = np.random.randn(2, 16, 64).astype(np.float32)
    out = proj.forward(vision_feats)

    assert out.shape == (2, 16, 128)


def test_perceiver_resampler():
    """Test PerceiverResampler fixed visual token compression."""
    resampler = PerceiverResampler(
        vision_dim=64,
        text_dim=128,
        num_latents=8,
        num_heads=4,
        seed=42,
    )
    # Variable number of visual tokens (25 tokens) -> compressed to exactly 8 tokens
    vision_feats = np.random.randn(3, 25, 64).astype(np.float32)
    out = resampler.forward(vision_feats)

    assert out.shape == (3, 8, 128)


def test_multimodal_exceptions():
    """Test validation errors for invalid architectures."""
    with pytest.raises(ValueError):
        PerceiverResampler(
            vision_dim=64, text_dim=128, num_heads=5
        )  # 128 not divisible by 5
