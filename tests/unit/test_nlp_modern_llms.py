"""Unit tests for Modern NLP Architectures: RoBERTa, DeBERTa, Mistral, and Gemma."""

import numpy as np

from chokkhu.models import (
    DeBERTa,
    DebertaForSequenceClassification,
    Gemma,
    Mistral,
    RoBERTa,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    train,
)


def test_roberta_forward_and_mlm() -> None:
    input_ids = np.random.randint(2, 500, size=(2, 16))
    attn_mask = np.ones((2, 16))

    roberta = RoBERTa(
        vocab_size=1000,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        dim_feedforward=128,
        max_seq_len=64,
    )
    out = roberta(input_ids, attention_mask=attn_mask)
    assert out.shape == (2, 16, 64)

    # Classification Head
    clf = RobertaForSequenceClassification(
        vocab_size=1000,
        num_classes=3,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        dim_feedforward=128,
    )
    cls_out = clf(input_ids)
    assert cls_out.shape == (2, 3)

    # Masked LM
    mlm = RobertaForMaskedLM(
        vocab_size=1000,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        dim_feedforward=128,
    )
    mlm_out = mlm(input_ids)
    assert mlm_out.shape == (2, 16, 1000)


def test_deberta_disentangled_attention() -> None:
    input_ids = np.random.randint(0, 500, size=(2, 12))

    deberta = DeBERTa(
        vocab_size=1000,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        max_relative_positions=16,
    )
    out = deberta(input_ids)
    assert out.shape == (2, 12, 64)

    clf = DebertaForSequenceClassification(
        vocab_size=1000,
        num_classes=2,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        max_relative_positions=16,
    )
    cls_out = clf(input_ids)
    assert cls_out.shape == (2, 2)


def test_mistral_forward() -> None:
    input_ids = np.random.randint(0, 500, size=(2, 14))

    mistral = Mistral(
        vocab_size=1000,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        window_size=8,
        hidden_dim=128,
        max_seq_len=64,
    )
    logits = mistral(input_ids)
    assert logits.shape == (2, 14, 1000)


def test_gemma_forward() -> None:
    input_ids = np.random.randint(0, 500, size=(2, 10))

    gemma = Gemma(
        vocab_size=1000,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        hidden_dim=128,
        max_seq_len=64,
    )
    logits = gemma(input_ids)
    assert logits.shape == (2, 10, 1000)


def test_train_modern_llms_via_engine() -> None:
    X_train = np.random.randint(0, 500, size=(4, 8))
    y_train = np.array([0, 1, 0, 1])

    clf = train(
        model="roberta",
        X_train=X_train,
        y_train=y_train,
        vocab_size=500,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        num_classes=2,
        epochs=1,
        batch_size=2,
    )
    preds = clf.predict(X_train)
    assert preds.shape == (4,)
