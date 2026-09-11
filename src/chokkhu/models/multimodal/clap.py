from __future__ import annotations

from typing import Tuple
import numpy as np


class CLAPAudioEncoder:
    """Audio Spectrogram Transformer Encoder for CLAP."""

    def __init__(
        self,
        embed_dim: int = 64,
        n_mels: int = 64,
        patch_size: int = 16,
        num_layers: int = 2,
        num_heads: int = 2,
        seed: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        rng = np.random.default_rng(seed)
        patch_dim = n_mels * patch_size
        scale = np.sqrt(2.0 / patch_dim)

        self.w_patch = rng.normal(0, scale, size=(patch_dim, embed_dim)).astype(
            np.float32
        )
        self.w_qkv = [
            rng.normal(
                0, np.sqrt(2.0 / embed_dim), size=(embed_dim, 3 * embed_dim)
            ).astype(np.float32)
            for _ in range(num_layers)
        ]
        self.w_proj = [
            rng.normal(0, np.sqrt(2.0 / embed_dim), size=(embed_dim, embed_dim)).astype(
                np.float32
            )
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encodes audio spectrograms (B, n_mels, time_steps) -> (B, embed_dim)."""
        if x.ndim == 4:
            x = x[:, 0, :, :]  # drop channel dim
        elif x.ndim == 2:
            x = x[np.newaxis, ...]

        B, F, T = x.shape
        # Extract non-overlapping time patches
        num_patches = max(1, T // self.patch_size)
        T_trim = num_patches * self.patch_size
        x_trim = x[:, :, :T_trim]

        # Reshape to (B, num_patches, F * patch_size)
        patches_4d = x_trim.reshape(B, F, num_patches, self.patch_size).transpose(
            0, 2, 1, 3
        )
        patches = patches_4d.reshape(B, num_patches, F * self.patch_size)

        # Linear patch projection
        h = np.matmul(patches, self.w_patch)  # (B, num_patches, embed_dim)

        # Transformer layers
        head_dim = self.embed_dim // self.num_heads
        for w_qkv, w_proj in zip(self.w_qkv, self.w_proj):
            qkv = np.matmul(h, w_qkv)
            q, k, v = np.split(qkv, 3, axis=-1)

            Q = q.reshape(B, num_patches, self.num_heads, head_dim).transpose(
                0, 2, 1, 3
            )
            K = k.reshape(B, num_patches, self.num_heads, head_dim).transpose(
                0, 2, 1, 3
            )
            V = v.reshape(B, num_patches, self.num_heads, head_dim).transpose(
                0, 2, 1, 3
            )

            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

            out = (
                np.matmul(attn, V)
                .transpose(0, 2, 1, 3)
                .reshape(B, num_patches, self.embed_dim)
            )
            h = h + np.matmul(out, w_proj)

        # Global average pooling
        return np.mean(h, axis=1)


class CLAPTextEncoder:
    """Transformer Text Encoder for CLAP."""

    def __init__(
        self,
        embed_dim: int = 64,
        vocab_size: int = 500,
        max_seq_len: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        seed: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / embed_dim)

        self.w_emb = rng.normal(0, scale, size=(vocab_size, embed_dim)).astype(
            np.float32
        )
        self.pos_emb = rng.normal(0, scale, size=(max_seq_len, embed_dim)).astype(
            np.float32
        )

        self.w_qkv = [
            rng.normal(0, scale, size=(embed_dim, 3 * embed_dim)).astype(np.float32)
            for _ in range(num_layers)
        ]
        self.w_proj = [
            rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(np.float32)
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encodes token index sequences (B, seq_len) -> (B, embed_dim)."""
        tokens = np.asarray(x, dtype=int)
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]

        B, S = tokens.shape
        tokens_clipped = np.clip(tokens, 0, self.vocab_size - 1)
        h = self.w_emb[tokens_clipped] + self.pos_emb[:S]

        head_dim = self.embed_dim // self.num_heads
        for w_qkv, w_proj in zip(self.w_qkv, self.w_proj):
            qkv = np.matmul(h, w_qkv)
            q, k, v = np.split(qkv, 3, axis=-1)

            Q = q.reshape(B, S, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            K = k.reshape(B, S, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            V = v.reshape(B, S, self.num_heads, head_dim).transpose(0, 2, 1, 3)

            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

            out = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            h = h + np.matmul(out, w_proj)

        # Global average pooling
        return np.mean(h, axis=1)


class CLAP:
    """Contrastive Language-Audio Pretraining (CLAP) in pure NumPy.

    Jointly learns multimodal representations by maximizing cosine similarity
    between paired audio spectrograms and natural language descriptions.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_mels: int = 64,
        vocab_size: int = 500,
        init_temperature: float = 0.07,
        seed: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.log_temperature = np.log(1.0 / init_temperature)

        self.audio_encoder = CLAPAudioEncoder(
            embed_dim=embed_dim, n_mels=n_mels, seed=seed
        )
        self.text_encoder = CLAPTextEncoder(
            embed_dim=embed_dim, vocab_size=vocab_size, seed=seed + 1
        )

    def encode_audio(self, spectrograms: np.ndarray) -> np.ndarray:
        """Returns normalized audio embeddings (B, embed_dim)."""
        emb = self.audio_encoder.forward(spectrograms)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
        return emb / norm

    def encode_text(self, text_tokens: np.ndarray) -> np.ndarray:
        """Returns normalized text embeddings (B, embed_dim)."""
        emb = self.text_encoder.forward(text_tokens)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
        return emb / norm

    def forward(
        self,
        spectrograms: np.ndarray,
        text_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Computes audio and text normalized embeddings and symmetric contrastive loss."""
        a_emb = self.encode_audio(spectrograms)
        t_emb = self.encode_text(text_tokens)

        # Scaled cosine similarity matrix: (B, B)
        scale = np.exp(self.log_temperature)
        sim = np.matmul(a_emb, t_emb.T) * scale

        # Symmetric cross-entropy loss
        B = len(sim)
        targets = np.arange(B)

        def _ce_loss(logits: np.ndarray, targets: np.ndarray) -> float:
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)
            log_p = np.log(probs[np.arange(len(targets)), targets] + 1e-12)
            return float(-np.mean(log_p))

        loss_a2t = _ce_loss(sim, targets)
        loss_t2a = _ce_loss(sim.T, targets)
        loss = (loss_a2t + loss_t2a) / 2.0

        return a_emb, t_emb, float(loss)

    def predict_similarity(
        self, spectrograms: np.ndarray, text_tokens: np.ndarray
    ) -> np.ndarray:
        """Returns cosine similarity matrix between audio spectrograms and text prompts."""
        a_emb = self.encode_audio(spectrograms)
        t_emb = self.encode_text(text_tokens)
        return np.matmul(a_emb, t_emb.T)
