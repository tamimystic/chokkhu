from __future__ import annotations

import numpy as np

from chokkhu.models.nlp import (
    RWKV6TimeMix,
    RWKV6,
    RetNetRetention,
    RetNet,
    MambaSSM,
    Mamba,
    DPOTrainer,
    KTOTrainer,
    ORPOTrainer,
    DeepSeekMoE,
    DeepSeekV3,
    Qwen2_5,
    BitLinear,
    BitNet158,
)


def test_rwkv6_linear_attention() -> None:
    B, T, D = 2, 4, 16
    x = np.random.randn(B, T, D).astype(np.float32)

    time_mix = RWKV6TimeMix(hidden_dim=D, n_heads=2)
    out = time_mix.forward(x)
    assert out.shape == (B, T, D)

    # Full RWKV6 model
    vocab_size = 50
    model = RWKV6(vocab_size=vocab_size, hidden_dim=D, num_layers=2, n_heads=2)
    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = model.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)


def test_retnet_retention() -> None:
    B, T, D = 2, 4, 16
    x = np.random.randn(B, T, D).astype(np.float32)

    ret = RetNetRetention(hidden_dim=D, n_heads=2)
    out = ret.forward(x)
    assert out.shape == (B, T, D)

    vocab_size = 50
    model = RetNet(vocab_size=vocab_size, hidden_dim=D, num_layers=2, n_heads=2)
    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = model.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)


def test_mamba_selective_ssm() -> None:
    B, T, d_model = 2, 6, 16
    u = np.random.randn(B, T, d_model).astype(np.float32)

    ssm_block = MambaSSM(d_model=d_model, d_state=8, d_conv=3, expand=2)
    out = ssm_block.forward(u)
    assert out.shape == (B, T, d_model)

    vocab_size = 50
    mamba_model = Mamba(vocab_size=vocab_size, d_model=d_model, num_layers=2, d_state=8)
    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = mamba_model.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)


def test_alignment_trainers() -> None:
    B = 4
    policy_chosen = np.array([-1.2, -0.8, -1.5, -0.9], dtype=np.float32)
    policy_rejected = np.array([-2.5, -2.1, -2.8, -2.0], dtype=np.float32)
    ref_chosen = np.array([-1.4, -1.0, -1.6, -1.1], dtype=np.float32)
    ref_rejected = np.array([-2.2, -1.9, -2.4, -1.8], dtype=np.float32)

    # 1. DPO Trainer
    dpo = DPOTrainer(beta=0.1)
    loss_dpo, r_w, r_l = dpo.compute_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    assert isinstance(loss_dpo, float)
    assert loss_dpo > 0.0
    assert r_w.shape == (B,)
    assert r_l.shape == (B,)
    # Chosen rewards should be higher than rejected
    assert np.all(r_w > r_l)

    # 2. KTO Trainer
    kto = KTOTrainer(beta=0.1)
    labels = np.array([True, False, True, False])
    loss_kto = kto.compute_loss(policy_chosen, ref_chosen, labels)
    assert isinstance(loss_kto, float)
    assert loss_kto > 0.0

    # 3. ORPO Trainer
    orpo = ORPOTrainer(lambda_orpo=0.1)
    loss_orpo = orpo.compute_loss(
        sft_loss=1.5,
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
    )
    assert isinstance(loss_orpo, float)
    assert loss_orpo >= 1.5


def test_deepseek_moe_and_v3() -> None:
    B, T, D = 2, 4, 16
    x = np.random.randn(B, T, D).astype(np.float32)

    moe = DeepSeekMoE(
        hidden_dim=D,
        intermediate_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
    )
    out = moe.forward(x)
    assert out.shape == (B, T, D)

    vocab_size = 60
    deepseek = DeepSeekV3(
        vocab_size=vocab_size, hidden_dim=D, num_layers=2, n_routed_experts=4, top_k=2
    )
    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = deepseek.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)

    logits, mtp_logits = deepseek.forward(input_ids, return_mtp=True)
    assert logits.shape == (B, T, vocab_size)
    assert mtp_logits.shape == (B, T, vocab_size)


def test_qwen2_5() -> None:
    B, T, D = 2, 4, 16
    vocab_size = 50
    qwen = Qwen2_5(
        vocab_size=vocab_size,
        hidden_dim=D,
        num_layers=2,
        num_heads=2,
        intermediate_dim=32,
    )

    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = qwen.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)


def test_bitnet_1_58b() -> None:
    B, T, in_features, out_features = 2, 4, 16, 16
    x = np.random.randn(B, T, in_features).astype(np.float32)

    bit_linear = BitLinear(in_features=in_features, out_features=out_features)
    out = bit_linear.forward(x)
    assert out.shape == (B, T, out_features)

    vocab_size = 50
    bitnet = BitNet158(vocab_size=vocab_size, hidden_dim=16, num_layers=2)
    input_ids = np.random.randint(0, vocab_size, size=(B, T))
    logits = bitnet.forward(input_ids)
    assert logits.shape == (B, T, vocab_size)
