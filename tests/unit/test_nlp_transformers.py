"""Unit tests for NLP tokenizers, embeddings, recurrent models, attention, transformer blocks, architectures, and generation."""

from __future__ import annotations
import os
import tempfile
import numpy as np

import chokkhu as ck
from chokkhu.core.tensor import Tensor
from chokkhu.models.nlp import (
    CharacterTokenizer,
    WordTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RoPE,
    ALiBi,
    RNN,
    LSTM,
    BiLSTM,
    GRU,
    ScaledDotProductAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    RMSNorm,
    SwiGLU,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    BERT,
    BertForSequenceClassification,
    BertForMaskedLM,
    GPTForCausalLM,
    LLaMA,
    Seq2SeqTransformer,
    GenerationConfig,
    TextGenerator,
)


# --- 1. Tokenizer Tests ---


def test_character_tokenizer():
    corpus = ["hello world", "deep learning is fun"]
    tok = CharacterTokenizer()
    tok.train(corpus)
    assert tok.vocab_size >= len(set("".join(corpus)))

    encoded = tok.encode("hello", add_special_tokens=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id

    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == "hello"

    batch = tok.batch_encode(["hi", "hello"], max_length=6, padding=True)
    assert batch.shape == (2, 6)


def test_word_tokenizer():
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is great!",
    ]
    tok = WordTokenizer(min_freq=1, lowercase=True)
    tok.train(corpus)

    tokens = tok.tokenize("The fox jumps.")
    assert "fox" in tokens
    assert "the" in tokens

    encoded = tok.encode("quick dog", bos=True, eos=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id

    decoded = tok.decode(encoded)
    assert "quick dog" in decoded


def test_bpe_tokenizer():
    corpus = [
        "low lower lowest",
        "new newer newest",
        "wide wider widest",
    ]
    tok = BPETokenizer(vocab_size=50)
    tok.train(corpus)

    encoded = tok.encode("lower widest")
    assert len(encoded) > 0

    decoded = tok.decode(encoded)
    assert "lower widest" in decoded or "lower" in decoded

    # Test save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "bpe.json")
        tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        assert tok2.vocab_size == tok.vocab_size


def test_wordpiece_tokenizer():
    corpus = ["playing player played play", "unhappiness happiness happy"]
    tok = WordPieceTokenizer(vocab_size=60)
    tok.train(corpus)

    tokens = tok.tokenize("playing")
    assert len(tokens) > 0

    encoded = tok.encode("playing")
    decoded = tok.decode(encoded)
    assert decoded.replace(" ", "") == "playing"


def test_sentencepiece_tokenizer():
    corpus = [
        "Machine learning with sovereign neural nets",
        "Chokkhu toolkit delivers speed and purity",
    ]
    tok = SentencePieceTokenizer(vocab_size=80)
    tok.train(corpus)

    tokens = tok.tokenize("Machine learning")
    assert len(tokens) > 0

    encoded = tok.encode("learning")
    decoded = tok.decode(encoded)
    assert "learning" in decoded


# --- 2. Embeddings & Positional Encodings ---


def test_token_embedding():
    vocab_size = 100
    embed_dim = 32
    emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=0)

    input_ids = np.array([[1, 5, 12, 0], [4, 9, 0, 0]])
    out = emb(input_ids)
    assert out.shape == (2, 4, 32)
    # Check padding idx is zeroed
    np.testing.assert_allclose(out.data[0, 3], np.zeros(32))

    # Test backprop gradient accumulation
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == (100, 32)


def test_sinusoidal_positional_encoding():
    pe = SinusoidalPositionalEncoding(max_seq_len=64, embed_dim=32)
    x = Tensor(np.zeros((2, 16, 32)), requires_grad=False)
    out = pe(x)
    assert out.shape == (2, 16, 32)
    # Position 0 should have sin(0) = 0 for first dimension
    assert np.isclose(out.data[0, 0, 0], 0.0)


def test_learned_positional_embedding():
    lpe = LearnedPositionalEmbedding(max_seq_len=64, embed_dim=32)
    x = Tensor(np.zeros((2, 10, 32)), requires_grad=True)
    out = lpe(x)
    assert out.shape == (2, 10, 32)


def test_rope_and_alibi():
    # RoPE
    rope = RoPE(dim=16, max_seq_len=32)
    q = np.random.randn(2, 4, 10, 16)
    rotated = rope.apply_rope(q, seq_len=10)
    assert rotated.shape == (2, 4, 10, 16)

    # ALiBi
    alibi_bias = ALiBi.get_bias(num_heads=8, seq_len=12)
    assert alibi_bias.shape == (1, 8, 12, 12)
    # Check diagonal is zero
    for h in range(8):
        for i in range(12):
            assert np.isclose(alibi_bias[0, h, i, i], 0.0)


# --- 3. Recurrent Models ---


def test_rnn():
    rnn = RNN(input_dim=16, hidden_dim=32, num_layers=2, batch_first=True)
    x = Tensor(np.random.randn(4, 8, 16))
    out, h_n = rnn(x)
    assert out.shape == (4, 8, 32)
    assert h_n.shape == (2, 4, 32)


def test_lstm_and_bilstm():
    # LSTM
    lstm = LSTM(input_dim=16, hidden_dim=32, num_layers=2, batch_first=True)
    x = Tensor(np.random.randn(4, 8, 16))
    out, (h_n, c_n) = lstm(x)
    assert out.shape == (4, 8, 32)
    assert h_n.shape == (2, 4, 32)
    assert c_n.shape == (2, 4, 32)

    # BiLSTM
    bilstm = BiLSTM(input_dim=16, hidden_dim=32, num_layers=1, batch_first=True)
    bi_out, (bi_hn, bi_cn) = bilstm(x)
    assert bi_out.shape == (4, 8, 64)
    assert bi_hn.shape == (1, 4, 64)


def test_gru():
    gru = GRU(input_dim=16, hidden_dim=32, num_layers=2, batch_first=True)
    x = Tensor(np.random.randn(4, 8, 16))
    out, h_n = gru(x)
    assert out.shape == (4, 8, 32)
    assert h_n.shape == (2, 4, 32)


# --- 4. Attention & Transformer Blocks ---


def test_attention_mechanisms():
    # ScaledDotProductAttention
    sdpa = ScaledDotProductAttention()
    q = np.random.randn(2, 4, 6, 16)
    k = np.random.randn(2, 4, 6, 16)
    v = np.random.randn(2, 4, 6, 16)
    out, weights = sdpa(q, k, v)
    assert out.shape == (2, 4, 6, 16)
    assert weights.shape == (2, 4, 6, 6)
    np.testing.assert_allclose(weights.sum(axis=-1), np.ones((2, 4, 6)), atol=1e-5)

    # MultiHeadAttention
    mha = MultiHeadAttention(embed_dim=32, num_heads=4)
    x = Tensor(np.random.randn(2, 8, 32))
    mha_out = mha(x, is_causal=True)
    assert mha_out.shape == (2, 8, 32)

    # GroupedQueryAttention (GQA)
    gqa = GroupedQueryAttention(embed_dim=32, num_query_heads=4, num_kv_heads=2)
    gqa_out = gqa(x, is_causal=True)
    assert gqa_out.shape == (2, 8, 32)


def test_transformer_blocks():
    # RMSNorm
    norm = RMSNorm(32)
    x = Tensor(np.random.randn(2, 6, 32))
    normed = norm(x)
    assert normed.shape == (2, 6, 32)

    # SwiGLU
    swiglu = SwiGLU(in_features=32, hidden_features=64)
    swi_out = swiglu(x)
    assert swi_out.shape == (2, 6, 32)

    # TransformerEncoderLayer
    enc_layer = TransformerEncoderLayer(embed_dim=32, num_heads=4, dim_feedforward=64)
    enc_out = enc_layer(x)
    assert enc_out.shape == (2, 6, 32)

    # TransformerDecoderLayer
    dec_layer = TransformerDecoderLayer(embed_dim=32, num_heads=4, dim_feedforward=64)
    dec_out = dec_layer(tgt=x, memory=enc_out)
    assert dec_out.shape == (2, 6, 32)


# --- 5. Architectures ---


def test_bert():
    vocab_size = 100
    bert = BERT(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        dim_feedforward=64,
        max_seq_len=32,
    )
    input_ids = np.random.randint(0, vocab_size, (2, 10))
    out = bert(input_ids)
    assert out.shape == (2, 10, 32)

    clf = BertForSequenceClassification(
        vocab_size=vocab_size,
        num_classes=3,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        dim_feedforward=64,
    )
    logits = clf(input_ids)
    assert logits.shape == (2, 3)

    preds = clf.predict(input_ids)
    assert len(preds) == 2

    mlm = BertForMaskedLM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        dim_feedforward=64,
    )
    mlm_logits = mlm(input_ids)
    assert mlm_logits.shape == (2, 10, vocab_size)


def test_gpt_and_llama():
    vocab_size = 100
    input_ids = np.random.randint(0, vocab_size, (2, 8))

    # GPT
    gpt_lm = GPTForCausalLM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        dim_feedforward=64,
        max_seq_len=32,
    )
    gpt_logits = gpt_lm(input_ids)
    assert gpt_logits.shape == (2, 8, vocab_size)

    # LLaMA
    llama = LLaMA(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
    )
    llama_logits = llama(input_ids)
    assert llama_logits.shape == (2, 8, vocab_size)


def test_seq2seq_transformer():
    src_v = 80
    tgt_v = 80
    src = np.random.randint(0, src_v, (2, 6))
    tgt = np.random.randint(0, tgt_v, (2, 5))

    s2s = Seq2SeqTransformer(
        src_vocab_size=src_v,
        tgt_vocab_size=tgt_v,
        embed_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dim_feedforward=64,
    )
    out = s2s(src, tgt)
    assert out.shape == (2, 5, tgt_v)


# --- 6. Generation Engine ---


def test_generation_engine():
    vocab_size = 50
    model = GPTForCausalLM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        max_seq_len=32,
    )
    tok = CharacterTokenizer()
    tok.train(["abcdefghijklmnopqrstuvwxyz "])

    config = GenerationConfig(
        max_new_tokens=5, temperature=0.7, top_k=10, top_p=0.9, do_sample=True
    )
    generator = TextGenerator(model=model, tokenizer=tok)

    gen_text = generator.generate("abc", config=config)
    assert isinstance(gen_text, str)
    assert len(gen_text) > 0


# --- 7. Engine Integration (1-line API) ---


def test_engine_nlp_training():
    vocab_size = 50
    X = np.random.randint(0, vocab_size, (10, 8))
    y = np.random.randint(0, 2, (10,))

    model = ck.train(
        model="bert",
        X_train=X,
        y_train=y,
        num_classes=2,
        vocab_size=vocab_size,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        epochs=1,
        verbose=False,
    )
    preds = model.predict(X)
    assert len(preds) == 10
