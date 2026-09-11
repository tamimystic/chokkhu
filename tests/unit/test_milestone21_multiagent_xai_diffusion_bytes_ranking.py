"""Unit tests for Milestone 21: Multi-Agent Economy, XAI Circuits, Guided Diffusion, Byte Models & Differentiable Sorting.

Tests covering:
- ReplicatorDynamics
- CombinatorialAuction (VCG)
- ActivationPatchingEngine
- ClassifierFreeGuidance
- DiffusionInpainter
- ByteTransformer
- NeuralSort
- DifferentiableRankingLoss
"""

from typing import Dict, Optional, Tuple
import numpy as np

from chokkhu import (
    ActivationPatchingEngine,
    ByteTransformer,
    ClassifierFreeGuidance,
    CombinatorialAuction,
    DifferentiableRankingLoss,
    DiffusionInpainter,
    NeuralSort,
    ReplicatorDynamics,
)


def test_replicator_dynamics():
    # 1. Hawk-Dove Game with V=2, C=4 -> ESS is 50% Hawk, 50% Dove
    hd = ReplicatorDynamics.hawks_doves(v=2.0, c=4.0, dt=0.05)
    init_pop = np.array([0.9, 0.1])
    traj = hd.simulate(init_pop, num_steps=200)
    assert traj.shape == (201, 2)
    # Check probability simplex constraint
    assert np.allclose(np.sum(traj, axis=1), 1.0)
    # Convergence towards ESS [0.5, 0.5]
    final_pop = traj[-1]
    assert np.allclose(final_pop, [0.5, 0.5], atol=0.05)

    # 2. Rock-Paper-Scissors cycle
    rps = ReplicatorDynamics.rock_paper_scissors(dt=0.02)
    rps_init = np.array([0.5, 0.3, 0.2])
    rps_traj = rps.simulate(rps_init, num_steps=50)
    assert rps_traj.shape == (51, 3)
    assert np.allclose(np.sum(rps_traj, axis=1), 1.0)


def test_combinatorial_auction_vcg():
    # Auction for Spectrum bands A and B
    auction = CombinatorialAuction(
        items=["A", "B"], bidders=["Alice", "Bob", "Charlie"]
    )

    # Alice wants bundle {A, B} for $10
    auction.add_bid("Alice", ["A", "B"], 10.0)
    # Bob wants {A} for $6
    auction.add_bid("Bob", ["A"], 6.0)
    # Charlie wants {B} for $5
    auction.add_bid("Charlie", ["B"], 5.0)

    result = auction.solve()
    allocs = result["allocations"]
    payments = result["payments"]
    welfare = result["social_welfare"]

    # Bob + Charlie = 6 + 5 = 11 > Alice's 10 -> Bob and Charlie should win
    assert allocs["Bob"] == ["A"]
    assert allocs["Charlie"] == ["B"]
    assert allocs["Alice"] == []
    assert welfare == 11.0

    # VCG Payments:
    # Bob's payment: Without Bob, Alice wins (10). Others' welfare with Bob = Charlie (5). Payment = 10 - 5 = 5.
    # Charlie's payment: Without Charlie, Alice wins (10). Others' welfare with Charlie = Bob (6). Payment = 10 - 6 = 4.
    assert np.isclose(payments["Bob"], 5.0)
    assert np.isclose(payments["Charlie"], 4.0)
    assert np.isclose(payments["Alice"], 0.0)


def test_activation_patching_engine():
    # Dummy 2-layer network with activation cache
    def dummy_forward_with_cache(
        x: np.ndarray,
        patch_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Layer 1
        l1_act = x * 2.0
        if patch_dict and "layer_1" in patch_dict:
            l1_act = patch_dict["layer_1"]

        # Layer 2
        l2_act = l1_act + 1.0
        if patch_dict and "layer_2" in patch_dict:
            l2_act = patch_dict["layer_2"]

        logits = np.sum(l2_act, axis=-1, keepdims=True)
        cache = {"layer_1": x * 2.0, "layer_2": l1_act + 1.0}
        return logits, cache

    engine = ActivationPatchingEngine(dummy_forward_with_cache)
    clean_inp = np.array([[5.0, 5.0]])
    corrupt_inp = np.array([[1.0, 1.0]])

    # Patching layer_1 with clean activations should fully restore output score
    effect_l1 = engine.compute_causal_effect(clean_inp, corrupt_inp, "layer_1")
    assert np.isclose(effect_l1, 1.0)

    # Trace all layers
    trace = engine.trace_circuit(clean_inp, corrupt_inp, ["layer_1", "layer_2"])
    assert "layer_1" in trace and "layer_2" in trace
    assert np.isclose(trace["layer_1"], 1.0)


def test_classifier_free_guidance():
    cfg = ClassifierFreeGuidance(guidance_scale=3.0)
    uncond = np.array([1.0, 2.0])
    cond = np.array([2.0, 4.0])

    # combined = uncond + 3.0 * (cond - uncond) = [1, 2] + 3.0 * [1, 2] = [4, 8]
    combined = cfg.combine_scores(uncond, cond)
    assert np.allclose(combined, [4.0, 8.0])


def test_diffusion_inpainter():
    class DummyDDPM:
        num_timesteps = 10
        alphas_cumprod = np.linspace(0.99, 0.05, 10)

        def predict_noise(self, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.1 * x_t

    ddpm = DummyDDPM()
    inpainter = DiffusionInpainter(ddpm, num_resample_steps=1, seed=42)

    image_clean = np.ones((1, 8, 8))
    # Mask: 1 in outer boundary, 0 in center 4x4 hole
    mask = np.ones((1, 8, 8))
    mask[:, 2:6, 2:6] = 0.0

    inpainted = inpainter.inpaint(image_clean, mask, num_steps=5)
    assert inpainted.shape == (1, 8, 8)
    assert not np.any(np.isnan(inpainted))
    # Known region must be exactly preserved
    assert np.allclose(inpainted[mask == 1.0], image_clean[mask == 1.0])


def test_byte_transformer():
    model = ByteTransformer(
        d_model=32, patch_size=2, num_heads=2, num_layers=1, max_seq_len=64, seed=42
    )

    # 1. UTF-8 byte encoding / decoding
    text = "Hello Sovereign AI! 🇧🇩"
    byte_arr = ByteTransformer.text_to_bytes(text)
    decoded_text = ByteTransformer.bytes_to_text(byte_arr)
    assert decoded_text == text

    # 2. Forward pass
    batch_bytes = np.array([byte_arr[:8], byte_arr[:8]])
    logits = model.forward(batch_bytes)
    assert logits.shape == (2, 8, 256)
    assert not np.any(np.isnan(logits))

    # 3. Generation
    generated = model.generate("AI", max_new_bytes=6, temperature=1.0)
    assert isinstance(generated, str)
    assert len(generated) >= 2


def test_neural_sort_and_ranking_loss():
    sorter = NeuralSort(tau=0.5)

    scores = np.array([10.0, 50.0, 20.0])
    # Order should be index 1 (50) -> rank 1, index 2 (20) -> rank 2, index 0 (10) -> rank 3
    ranks = sorter.soft_ranks(scores)
    assert ranks[1] < ranks[2] < ranks[0]

    # Permutation matrix
    P = sorter.permutation_matrix(scores)
    assert P.shape == (3, 3)
    assert np.allclose(np.sum(P, axis=-1), 1.0)  # row sums
    assert np.all(P >= 0.0)

    # Soft sort
    values = np.array([1.0, 5.0, 2.0])
    sorted_v = sorter.soft_sort(values, scores=scores)
    assert sorted_v.shape == (3,)
    # Highest score (index 1 -> value 5) should appear first
    assert sorted_v[0] > sorted_v[1] > sorted_v[2]

    # Ranking losses
    pred = np.array([10.0, 50.0, 20.0])
    target = np.array([1.0, 3.0, 2.0])  # identical rank order
    spearman_loss = DifferentiableRankingLoss.spearman_loss(pred, target, tau=0.5)
    assert spearman_loss < 0.2  # Near 0 when ranking matches

    ndcg_loss = DifferentiableRankingLoss.soft_ndcg_loss(pred, target, tau=0.5)
    assert 0.0 <= ndcg_loss <= 1.0
