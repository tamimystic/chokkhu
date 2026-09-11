"""Unit test suite for Milestone 16: Advanced Frontier Architectures.

Validates pure NumPy/SciPy implementations of:
1. Hybrid Retrieval & Reranking: BM25Plus, OkapiBM25, HybridReranker
2. Mechanistic XAI & Counterfactuals: AttentionRollout, DirectLogitAttribution, WachterCounterfactualExplainer, TCAV
3. Semantic & Promptable Vision: DeepLabV3Plus (ASPP), SegmentAnythingModel (SAM)
4. Multi-Modal Audio-Language: CLAP (Contrastive Language-Audio Pretraining)
5. Speech Recognition & Sequence Modeling: Whisper (Audio Conv Stem + Cross-Attention Decoder)
6. Latent Generative Models: LatentDiffusionModel (Conditioned LDM + DDIM Sampler)
"""

from __future__ import annotations

import numpy as np

from chokkhu.models.retrieval import BM25Plus, OkapiBM25, HybridReranker
from chokkhu.explainability import (
    AttentionRollout,
    DirectLogitAttribution,
    WachterCounterfactualExplainer,
    TCAV,
)
from chokkhu.models.vision.architectures import DeepLabV3Plus, SegmentAnythingModel
from chokkhu.models.multimodal import CLAP
from chokkhu.models.audio.architectures import Whisper
from chokkhu.models.generative import LatentDiffusionModel


# =====================================================================
# 1. Hybrid Retrieval & Reranking
# =====================================================================


def test_bm25_retrievers():
    """Test OkapiBM25 and BM25Plus indexing and retrieval."""
    corpus = [
        "quantum computing and machine learning algorithms",
        "deep learning neural networks for computer vision",
        "natural language processing with transformers and attention",
        "quantum physics and quantum entanglement theory",
    ]

    # OkapiBM25
    okapi = OkapiBM25(k1=1.5, b=0.75)
    okapi.fit(corpus)
    scores = okapi.get_scores("quantum theory")
    assert len(scores) == 4
    # Document 3 ("quantum physics and quantum entanglement theory") should have high score
    top_scores, top_indices = okapi.search("quantum computing", top_k=2)
    assert len(top_indices) == 2
    assert top_indices[0] == 0  # doc 0 has both quantum and computing

    # BM25Plus (delta lower bound prevents heavily penalizing long docs)
    bm25_plus = BM25Plus(k1=1.5, b=0.75, delta=1.0)
    bm25_plus.fit(corpus)
    plus_scores = bm25_plus.get_scores("machine learning")
    assert len(plus_scores) == 4
    assert plus_scores[0] > 0.0


def test_hybrid_reranker():
    """Test score blending and Reciprocal Rank Fusion (RRF)."""
    reranker = HybridReranker(rrf_k=60, alpha=0.5)

    # Mock dense and sparse scores for 4 candidate docs
    dense_scores = np.array([0.9, 0.4, 0.7, 0.1])
    sparse_scores = np.array([0.2, 0.8, 0.5, 0.0])

    # 1. Linear Score Blending
    blended_scores = reranker.blend_scores(dense_scores, sparse_scores, alpha=0.7)
    assert len(blended_scores) == 4
    # Blended score should be within convex combination bounds
    assert np.all(blended_scores >= 0.0)

    # 2. Reciprocal Rank Fusion
    dense_ranking = [0, 2, 1, 3]  # doc indices sorted by score
    sparse_ranking = [1, 2, 0, 3]
    fused_results = reranker.reciprocal_rank_fusion(
        [dense_ranking, sparse_ranking], top_k=3
    )
    assert len(fused_results) == 3
    # Returned entries are (doc_id, rrf_score)
    assert isinstance(fused_results[0][0], (int, np.integer))
    assert fused_results[0][1] >= fused_results[1][1]


# =====================================================================
# 2. Mechanistic XAI & Counterfactuals
# =====================================================================


def test_attention_rollout():
    """Test Mechanistic Attention Rollout across transformer layers."""
    rollout = AttentionRollout(discard_ratio=0.1, head_reduction="mean")
    np.random.seed(42)

    # 3 layers, 4 heads, seq_len 6
    num_layers, num_heads, seq_len = 3, 4, 6
    raw_attn = [np.random.rand(num_heads, seq_len, seq_len) for _ in range(num_layers)]
    # Normalize across last dimension (softmax-like)
    raw_attn = [a / a.sum(axis=-1, keepdims=True) for a in raw_attn]

    result = rollout.compute(raw_attn)
    assert result.shape == (seq_len, seq_len)
    # Row sums of rollout matrix should be approximately 1.0
    row_sums = np.sum(result, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(seq_len), atol=1e-5)


def test_direct_logit_attribution():
    """Test Direct Logit Attribution (DLA) on transformer residual stream."""
    np.random.seed(42)

    num_layers = 4
    d_model = 16
    vocab_size = 50

    unembedding = np.random.randn(d_model, vocab_size)
    dla = DirectLogitAttribution(unembedding_matrix=unembedding)

    residual_stream = np.random.randn(num_layers, d_model)
    target_token_id = 7

    attributions = dla.attribute(
        hidden_states=residual_stream,
        target_token_id=target_token_id,
    )
    assert len(attributions) == num_layers
    assert isinstance(attributions[0], float)


def test_wachter_counterfactual_explainer():
    """Test Wachter optimization-based counterfactual explanation."""
    np.random.seed(42)
    # Simple linear decision boundary: class 1 if x[0] + x[1] > 1.0 else 0
    weights = np.array([1.5, 1.2])

    def predict_fn(x: np.ndarray) -> np.ndarray:
        z = np.dot(x, weights)
        return 1.0 / (1.0 + np.exp(-z))

    explainer = WachterCounterfactualExplainer(
        model=None,
        predict_fn=predict_fn,
        lr=0.1,
        max_iter=150,
        lambda_1=0.01,
        lambda_2=0.5,
        tolerance=0.05,
    )

    x_orig = np.array([-1.0, -0.5])
    initial_prob = float(predict_fn(x_orig))
    assert initial_prob < 0.5

    # Find counterfactual for target probability 0.8
    result = explainer.explain(x_orig, target_prediction=0.8)
    cf = result["counterfactual"]
    cf_prob = float(predict_fn(cf))
    assert cf_prob >= 0.70
    # Ensure some perturbation occurred
    assert result["l2_distance"] > 0.1


def test_tcav_concept_activation():
    """Test Testing with Concept Activation Vectors (TCAV)."""
    np.random.seed(42)
    tcav = TCAV(seed=42)

    n_samples, feat_dim = 20, 16
    concept_acts = np.random.randn(n_samples, feat_dim) + 2.0  # shifted
    random_acts = np.random.randn(n_samples, feat_dim) - 2.0

    cav = tcav.compute_cav(concept_acts, random_acts)
    assert cav.shape == (feat_dim,)
    assert np.isclose(np.linalg.norm(cav), 1.0, atol=1e-4)

    # Model prediction function taking activations
    def mock_model(acts: np.ndarray) -> np.ndarray:
        # Score increases when acts align with cav
        return np.dot(acts, cav)[:, np.newaxis]

    test_acts = np.random.randn(15, feat_dim) + 1.0
    tcav_dict = tcav.compute_tcav_score(mock_model, test_acts, target_class=0)
    assert 0.0 <= tcav_dict["tcav_score"] <= 1.0
    assert tcav_dict["total_count"] == 15


# =====================================================================
# 3. Vision: DeepLabV3+ and Segment Anything Model (SAM)
# =====================================================================


def test_deeplabv3_plus_forward():
    """Test DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP)."""
    np.random.seed(42)
    model = DeepLabV3Plus(
        in_channels=3,
        num_classes=3,
        backbone_channels=8,
        aspp_channels=8,
        seed=42,
    )

    batch_size, h, w = 2, 16, 16
    x = np.random.randn(batch_size, 3, h, w).astype(np.float32)

    logits = model.forward(x)
    assert logits.shape == (batch_size, 3, h, w)

    mask = model.predict(x)
    assert mask.shape == (batch_size, h, w)
    assert np.all((mask >= 0) & (mask < 3))


def test_segment_anything_model():
    """Test Segment Anything Model (SAM) promptable mask decoder."""
    np.random.seed(42)
    sam = SegmentAnythingModel(
        embed_dim=16,
        num_heads=2,
        num_mask_tokens=3,
        seed=42,
    )

    batch_size = 1
    h, w = 8, 8
    image_embeddings = np.random.randn(batch_size, 16, h, w).astype(np.float32)

    # Point prompt: 2 coordinates per batch item
    points = np.array([[[2.0, 3.0], [5.0, 6.0]]], dtype=np.float32)
    point_labels = np.array([[1, 0]], dtype=np.int32)  # 1 = foreground, 0 = background

    masks, iou_preds = sam.forward(
        image_embeddings=image_embeddings,
        points=points,
        labels=point_labels,
    )

    assert masks.shape == (batch_size, 3, h, w)
    assert iou_preds.shape == (batch_size, 3)


# =====================================================================
# 4. Multi-Modal Audio-Language: CLAP
# =====================================================================


def test_clap_audio_text_embeddings():
    """Test Contrastive Language-Audio Pretraining (CLAP)."""
    np.random.seed(42)
    clap = CLAP(
        embed_dim=16,
        n_mels=16,
        vocab_size=50,
        init_temperature=0.07,
        seed=42,
    )

    batch_size = 4
    # Audio spectrogram: (B, n_mels=16, time_steps=32)
    spectrograms = np.random.randn(batch_size, 16, 32).astype(np.float32)
    # Text tokens: (B, seq_len=8)
    text_tokens = np.random.randint(0, 50, size=(batch_size, 8))

    audio_emb, text_emb, loss = clap.forward(spectrograms, text_tokens)

    # Embeddings must be L2-normalized
    audio_norms = np.linalg.norm(audio_emb, axis=-1)
    text_norms = np.linalg.norm(text_emb, axis=-1)
    np.testing.assert_allclose(audio_norms, np.ones(batch_size), atol=1e-4)
    np.testing.assert_allclose(text_norms, np.ones(batch_size), atol=1e-4)

    # Similarity matrix
    sim_matrix = clap.predict_similarity(spectrograms, text_tokens)
    assert sim_matrix.shape == (batch_size, batch_size)

    # Contrastive Loss
    assert loss > 0.0
    assert not np.isnan(loss)


# =====================================================================
# 5. Speech Recognition & Sequence Modeling: Whisper
# =====================================================================


def test_whisper_encoder_decoder_and_generate():
    """Test Whisper audio conv stem downsampler, encoder, and autoregressive decoder."""
    np.random.seed(42)
    whisper = Whisper(
        n_mels=16,
        vocab_size=25,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        seed=42,
    )

    batch_size = 2
    mel_len = 32
    mel_spectrogram = np.random.randn(batch_size, 16, mel_len).astype(np.float32)
    target_tokens = np.array([[1, 5, 8, 2], [1, 7, 3, 2]], dtype=np.int32)

    # Forward pass
    logits = whisper.forward(mel_spectrogram, target_tokens)
    assert logits.shape == (batch_size, 4, 25)

    # Autoregressive Generation
    generated_tokens = whisper.generate(
        mel_spectrogram=mel_spectrogram,
        max_len=5,
        prompt_tokens=[1],
    )
    assert generated_tokens.shape[0] == batch_size
    assert generated_tokens.shape[1] == 6  # 1 prompt token + 5 generated
    # First token must be prompt token 1
    assert np.all(generated_tokens[:, 0] == 1)


# =====================================================================
# 6. Latent Generative Models: LatentDiffusionModel
# =====================================================================


def test_latent_diffusion_model():
    """Test Latent Diffusion forward noise scheduling and cross-attention sampling."""
    np.random.seed(42)
    ldm = LatentDiffusionModel(
        latent_dim=8,
        context_dim=8,
        num_timesteps=50,
        beta_start=0.0001,
        beta_end=0.02,
        num_heads=2,
        seed=42,
    )

    batch_size = 2
    x_0 = np.random.randn(batch_size, 8).astype(np.float32)
    t = np.array([5, 20], dtype=np.int32)
    noise = np.random.randn(batch_size, 8).astype(np.float32)
    context = np.random.randn(batch_size, 8).astype(np.float32)

    # 1. Forward diffusion q_sample
    x_t, generated_noise = ldm.q_sample(x_0, t, noise)
    assert x_t.shape == (batch_size, 8)
    assert not np.isnan(x_t).any()

    # 2. Noise prediction with context conditioning
    pred_noise = ldm.predict_noise(x_t, t, context)
    assert pred_noise.shape == (batch_size, 8)
    assert not np.isnan(pred_noise).any()

    # 3. Sampling loop
    sampled_latents = ldm.sample(
        shape=(batch_size, 8),
        context=context,
        n_steps=5,
    )
    assert sampled_latents.shape == (batch_size, 8)
    assert not np.isnan(sampled_latents).any()
