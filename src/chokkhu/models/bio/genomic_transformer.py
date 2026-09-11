"""GenomicBERT Bidirectional Transformer for DNA/RNA Modeling in pure NumPy."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class GenomicBERT:
    """Bidirectional Genomic Transformer Encoder for DNA Sequences in pure NumPy.

    Parameters
    ----------
    vocab_size : int, default=4101
        Genomic vocabulary size (4^6 + special tokens).
    d_model : int, default=64
        Hidden representation dimensionality.
    num_heads : int, default=4
        Number of self-attention heads.
    num_layers : int, default=2
        Number of transformer encoder layers.
    max_len : int, default=128
        Maximum sequence length.
    num_classes : int, default=2
        Number of sequence classification target classes.
    seed : int, default=42
    """

    def __init__(
        self,
        vocab_size: int = 4101,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
        num_classes: int = 2,
        seed: int = 42,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_len = int(max_len)
        self.num_classes = int(num_classes)
        self.head_dim = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        # 1. Embeddings
        self.token_embeddings: np.ndarray = rng.normal(
            0, scale, size=(vocab_size, d_model)
        ).astype(np.float32)
        self.pos_embeddings: np.ndarray = rng.normal(
            0, scale, size=(max_len, d_model)
        ).astype(np.float32)

        # 2. Transformer Encoder Layers
        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(num_layers):
            layer = {
                "w_q": rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32),
                "w_k": rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32),
                "w_v": rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32),
                "w_o": rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32),
                "w_ffn1": rng.normal(0, scale, size=(4 * d_model, d_model)).astype(
                    np.float32
                ),
                "b_ffn1": np.zeros(4 * d_model, dtype=np.float32),
                "w_ffn2": rng.normal(0, scale, size=(d_model, 4 * d_model)).astype(
                    np.float32
                ),
                "b_ffn2": np.zeros(d_model, dtype=np.float32),
            }
            self.layers.append(layer)

        # 3. Heads: Classification and Masked LM
        self.w_cls = rng.normal(0, scale, size=(num_classes, d_model)).astype(
            np.float32
        )
        self.b_cls: np.ndarray = np.zeros(num_classes, dtype=np.float32)

        self.w_mlm = rng.normal(0, scale, size=(vocab_size, d_model)).astype(np.float32)
        self.b_mlm: np.ndarray = np.zeros(vocab_size, dtype=np.float32)

    def forward(
        self, token_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through GenomicBERT encoder.

        Parameters
        ----------
        token_ids : np.ndarray
            Matrix of token indices: (B, L) or (L,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (hidden_states (B, L, D), cls_logits (B, num_classes), mlm_logits (B, L, vocab_size))
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        B, L = ids.shape

        # Token + Positional embeddings
        x = self.token_embeddings[ids] + self.pos_embeddings[:L]

        # Encoder Layers
        for layer in self.layers:
            # Self-Attention
            Q = (
                np.matmul(x, layer["w_q"].T)
                .reshape(B, L, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            K = (
                np.matmul(x, layer["w_k"].T)
                .reshape(B, L, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            V = (
                np.matmul(x, layer["w_v"].T)
                .reshape(B, L, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )

            scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.head_dim)
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

            context = (
                np.matmul(attn_weights, V).swapaxes(1, 2).reshape(B, L, self.d_model)
            )
            attn_out = np.matmul(context, layer["w_o"].T)
            x = x + attn_out  # Residual 1

            # FFN with GELU
            h_ffn = np.matmul(x, layer["w_ffn1"].T) + layer["b_ffn1"]
            gelu = (
                0.5
                * h_ffn
                * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h_ffn + 0.044715 * h_ffn**3)))
            )
            ffn_out = np.matmul(gelu, layer["w_ffn2"].T) + layer["b_ffn2"]
            x = x + ffn_out  # Residual 2

        # CLS classification logits
        cls_repr = x[:, 0, :]  # (B, d_model)
        cls_logits = np.matmul(cls_repr, self.w_cls.T) + self.b_cls

        # Masked LM logits
        mlm_logits = np.matmul(x, self.w_mlm.T) + self.b_mlm

        return x, cls_logits, mlm_logits

    def score_variant(self, ref_tokens: np.ndarray, alt_tokens: np.ndarray) -> float:
        """Computes Variant Impact Score (log-likelihood ratio: log P(alt) - log P(ref))."""
        _, _, mlm_ref = self.forward(ref_tokens)
        _, _, mlm_alt = self.forward(alt_tokens)

        # Log-softmax
        def log_softmax(logits: np.ndarray) -> np.ndarray:
            return logits - np.log(
                np.sum(
                    np.exp(logits - np.max(logits, axis=-1, keepdims=True)),
                    axis=-1,
                    keepdims=True,
                )
                + 1e-12
            )

        log_p_ref = log_softmax(mlm_ref)
        log_p_alt = log_softmax(mlm_alt)

        return float(np.mean(log_p_alt) - np.mean(log_p_ref))
