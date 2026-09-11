from __future__ import annotations

from typing import List
import numpy as np

from chokkhu import (
    ControlNet,
    CycleGAN,
    DynamicEnsembleSelection,
    EllipticEnvelope,
    GeneticPipelineSearch,
    LocalOutlierFactor,
    ResidualVectorQuantizer,
    SpeculativeDecoder,
    StructuredJSONDecoder,
    UMAP,
    VoiceActivityDetector,
    ZeroConv2D,
)
from chokkhu.core.tensor import Tensor
from chokkhu.models.generative import (
    ControlNetBlock,
    CycleGANGenerator,
    PatchGANDiscriminator,
)


def test_controlnet_zero_conv_and_block():
    zc = ZeroConv2D(in_channels=4, out_channels=4)
    x = np.random.randn(2, 4, 8, 8).astype(np.float32)
    out = zc.forward(x)
    np.testing.assert_allclose(out, 0.0, atol=1e-7)

    block = ControlNetBlock(channels=4, seed=42)
    base_feat = np.random.randn(2, 4, 8, 8).astype(np.float32)
    cond_feat = np.random.randn(2, 4, 8, 8).astype(np.float32)
    out_feat = block.forward(base_feat, cond_feat)
    assert out_feat.shape == (2, 4, 8, 8)

    cnet = ControlNet(
        in_channels=4, cond_channels=3, base_channels=4, num_stages=2, seed=42
    )
    img = np.random.randn(2, 4, 8, 8).astype(np.float32)
    cond = np.random.randn(2, 3, 8, 8).astype(np.float32)
    out_cnet, residuals = cnet.forward(img, cond)
    assert out_cnet.shape == (2, 4, 8, 8)
    assert len(residuals) == 2


def test_cyclegan_architecture():
    gen = CycleGANGenerator(in_channels=3, out_channels=3, base_channels=8, seed=42)
    img = np.random.randn(1, 3, 8, 8).astype(np.float32)
    fake_img = gen.forward(img)
    assert fake_img.shape == (1, 3, 8, 8)

    disc = PatchGANDiscriminator(in_channels=3, base_channels=8, seed=42)
    pred_patch = disc.forward(fake_img)
    assert pred_patch.ndim == 4
    assert pred_patch.shape[1] == 1

    cyclegan = CycleGAN(channels_a=3, channels_b=3, base_channels=8, seed=42)
    real_A = np.random.randn(1, 3, 8, 8).astype(np.float32)
    real_B = np.random.randn(1, 3, 8, 8).astype(np.float32)
    res = cyclegan.forward(real_A, real_B)
    assert "fake_b" in res
    assert "fake_a" in res
    assert "loss_cycle" in res
    assert "total_loss" in res

    trans_B = cyclegan.translate_a2b(real_A)
    trans_A = cyclegan.translate_b2a(real_B)
    assert trans_B.shape == real_A.shape
    assert trans_A.shape == real_B.shape


def test_residual_vector_quantizer():
    rvq = ResidualVectorQuantizer(
        num_quantizers=3, codebook_size=16, embed_dim=8, seed=42
    )
    z = np.random.randn(2, 6, 8).astype(np.float32)  # (batch, time, dim)
    quantized, codes, loss = rvq.forward(z)
    assert quantized.shape == z.shape
    assert codes.shape == (3, 2, 6)
    assert loss >= 0.0

    # Test decode from codes
    decoded = rvq.decode_codes(codes)
    np.testing.assert_allclose(decoded, quantized, atol=1e-5)


def test_voice_activity_detector():
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_length_ms=25.0,
        hop_length_ms=10.0,
        energy_threshold=0.01,
    )
    # Synthetic audio: silence then tone then silence
    t = np.linspace(0, 0.5, 8000)
    tone = np.sin(2 * np.pi * 440 * t) * 0.5
    silence = np.zeros(4000)
    audio = np.concatenate([silence, tone, silence])

    feats = vad.compute_features(audio)
    assert "ste" in feats
    assert "zcr" in feats
    assert "spectral_flux" in feats

    res = vad.detect(audio)
    assert "mask" in res
    assert "segments" in res
    assert "speech_ratio" in res
    assert isinstance(res["segments"], list)


class MockTokenizer:
    def encode(self, text: str, bos: bool = True) -> List[int]:
        return [1, 2, 3]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return "mock generated text"


def test_speculative_decoder():
    def mock_draft(tensor_in):
        vocab_size = 20
        logits = np.ones((1, 1, vocab_size), dtype=np.float32)
        return Tensor(logits, requires_grad=False)

    def mock_target(tensor_in):
        vocab_size = 20
        logits = np.zeros((1, 1, vocab_size), dtype=np.float32)
        logits[0, 0, 5] = 10.0
        return Tensor(logits, requires_grad=False)

    tokenizer = MockTokenizer()
    spec_decoder = SpeculativeDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=tokenizer,
        gamma=2,
        temperature=1.0,
    )
    res = spec_decoder.generate("Hello world", max_new_tokens=4, eos_token_id=None)
    assert "text" in res
    assert "acceptance_rate" in res
    assert "total_draft_tokens" in res


def test_structured_json_decoder():
    def mock_model(tensor_in):
        vocab_size = 50
        logits = np.random.randn(1, 1, vocab_size).astype(np.float32)
        return Tensor(logits, requires_grad=False)

    tokenizer = MockTokenizer()
    json_decoder = StructuredJSONDecoder(
        model=mock_model,
        tokenizer=tokenizer,
    )
    schema = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        }
    }
    res = json_decoder.generate(
        "Generate user profile", schema=schema, max_new_tokens=10
    )
    assert "json_string" in res
    assert "parsed_json" in res
    assert "name" in res["parsed_json"]


def test_umap_manifold_learning():
    np.random.seed(42)
    c1 = np.random.randn(15, 4) + 4.0
    c2 = np.random.randn(15, 4) - 4.0
    X = np.vstack([c1, c2])

    umap = UMAP(
        n_components=2, n_neighbors=5, min_dist=0.1, n_epochs=20, random_state=42
    )
    embedding = umap.fit_transform(X)
    assert embedding.shape == (30, 2)
    assert np.all(np.isfinite(embedding))


def test_genetic_pipeline_search():
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    search = GeneticPipelineSearch(
        population_size=4,
        generations=2,
        mutation_rate=0.2,
        crossover_rate=0.5,
        cv_folds=2,
        seed=42,
    )
    search.fit(X, y)
    assert search.best_individual is not None
    assert len(search.history) >= 1


def test_dynamic_ensemble_selection():
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)

    pool = [
        lambda data: (data[:, 0] > 0).astype(int),
        lambda data: (data[:, 1] > 0).astype(int),
    ]

    des = DynamicEnsembleSelection(
        pool_classifiers=pool, k_neighbors=5, method="knora_e"
    )
    des.fit(X, y)
    preds = des.predict(X)
    assert len(preds) == len(y)


def test_local_outlier_factor():
    np.random.seed(42)
    X_inliers = np.random.randn(30, 2)
    X_outliers = np.random.uniform(low=-10, high=10, size=(4, 2))
    X = np.vstack([X_inliers, X_outliers])

    lof = LocalOutlierFactor(n_neighbors=8, contamination=0.1, novelty=True)
    preds = lof.fit_predict(X)
    assert len(preds) == 34
    assert np.sum(preds == -1) > 0

    novel_scores = lof.score_samples(X_outliers)
    assert len(novel_scores) == 4


def test_elliptic_envelope():
    np.random.seed(42)
    X_inliers = np.random.randn(30, 2)
    X_outliers = np.random.uniform(low=-10, high=10, size=(4, 2))
    X = np.vstack([X_inliers, X_outliers])

    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    ee.fit(X)
    preds = ee.predict(X)
    assert len(preds) == 34
    assert np.sum(preds == -1) > 0
    scores = ee.score_samples(X)
    assert len(scores) == 34
