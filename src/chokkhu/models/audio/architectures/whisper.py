from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation."""
    return (
        0.5
        * x
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    )


class WhisperEncoder:
    """Whisper Audio Transformer Encoder."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_source_positions: int = 500,
        seed: int = 42,
    ) -> None:
        self.n_mels = n_mels
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        scale_conv1 = np.sqrt(2.0 / (n_mels * 3))
        # 1D conv1: (d_model, n_mels, 3)
        self.w_conv1 = rng.normal(0, scale_conv1, size=(d_model, n_mels, 3)).astype(
            np.float32
        )
        # 1D conv2: (d_model, d_model, 3) with stride 2
        scale_conv2 = np.sqrt(2.0 / (d_model * 3))
        self.w_conv2 = rng.normal(0, scale_conv2, size=(d_model, d_model, 3)).astype(
            np.float32
        )

        # Positional encoding
        self.positional_embedding = rng.normal(
            0, np.sqrt(2.0 / d_model), size=(max_source_positions, d_model)
        ).astype(np.float32)

        # Transformer blocks
        scale = np.sqrt(2.0 / d_model)
        self.w_qkv = [
            rng.normal(0, scale, size=(d_model, 3 * d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_proj = [
            rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_mlp1 = [
            rng.normal(0, scale, size=(d_model, 4 * d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_mlp2 = [
            rng.normal(
                0, np.sqrt(2.0 / (4 * d_model)), size=(4 * d_model, d_model)
            ).astype(np.float32)
            for _ in range(n_layers)
        ]

    def _conv1d(
        self, x: np.ndarray, w: np.ndarray, stride: int = 1, padding: int = 1
    ) -> np.ndarray:
        B, C_in, L = x.shape
        C_out, _, kW = w.shape

        if padding > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        else:
            x_pad = x

        L_pad = x_pad.shape[2]
        out_L = (L_pad - kW) // stride + 1
        out = np.zeros((B, C_out, out_L), dtype=np.float32)

        for i in range(out_L):
            idx = i * stride
            patch = x_pad[:, :, idx : idx + kW]
            out[:, :, i] = np.tensordot(patch, w, axes=((1, 2), (1, 2)))

        return out

    def forward(self, mel: np.ndarray) -> np.ndarray:
        """Encodes log-mel spectrogram (B, n_mels, time_frames) -> (B, T_enc, d_model)."""
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]

        # 1. Conv stem: downsample by 2x
        h = _gelu(self._conv1d(mel, self.w_conv1, stride=1, padding=1))
        h = _gelu(
            self._conv1d(h, self.w_conv2, stride=2, padding=1)
        )  # (B, d_model, T_enc)

        # Transpose to (B, T_enc, d_model)
        h = h.transpose(0, 2, 1)
        B, T_enc, D = h.shape

        # Add positional embedding
        h = h + self.positional_embedding[:T_enc]

        # Transformer layers
        head_dim = self.d_model // self.n_heads
        for w_qkv, w_proj, w_m1, w_m2 in zip(
            self.w_qkv, self.w_proj, self.w_mlp1, self.w_mlp2
        ):
            qkv = np.matmul(h, w_qkv)
            q, k, v = np.split(qkv, 3, axis=-1)

            Q = q.reshape(B, T_enc, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            K = k.reshape(B, T_enc, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            V = v.reshape(B, T_enc, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

            out_att = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T_enc, D)
            h = h + np.matmul(out_att, w_proj)

            # FFN
            h_mlp = _gelu(np.matmul(h, w_m1))
            h = h + np.matmul(h_mlp, w_m2)

        return h


class WhisperDecoder:
    """Whisper Autoregressive Causal Decoder with Cross-Attention over Audio."""

    def __init__(
        self,
        vocab_size: int = 500,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_target_positions: int = 128,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        self.token_embedding = rng.normal(0, scale, size=(vocab_size, d_model)).astype(
            np.float32
        )
        self.positional_embedding = rng.normal(
            0, scale, size=(max_target_positions, d_model)
        ).astype(np.float32)

        # Self-attention weights
        self.w_self_qkv = [
            rng.normal(0, scale, size=(d_model, 3 * d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_self_proj = [
            rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]

        # Cross-attention weights (Q from text, K & V from audio encoder)
        self.w_cross_q = [
            rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_cross_kv = [
            rng.normal(0, scale, size=(d_model, 2 * d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_cross_proj = [
            rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]

        # FFN
        self.w_mlp1 = [
            rng.normal(0, scale, size=(d_model, 4 * d_model)).astype(np.float32)
            for _ in range(n_layers)
        ]
        self.w_mlp2 = [
            rng.normal(
                0, np.sqrt(2.0 / (4 * d_model)), size=(4 * d_model, d_model)
            ).astype(np.float32)
            for _ in range(n_layers)
        ]

        # Language head projection
        self.w_head = rng.normal(0, scale, size=(d_model, vocab_size)).astype(
            np.float32
        )

    def forward(self, tokens: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Decodes token sequence conditioned on audio features -> (B, S, vocab_size)."""
        tok = np.asarray(tokens, dtype=int)
        if tok.ndim == 1:
            tok = tok[np.newaxis, :]

        B, S = tok.shape
        _, T_enc, _ = audio_features.shape

        tok_clipped = np.clip(tok, 0, self.vocab_size - 1)
        h = self.token_embedding[tok_clipped] + self.positional_embedding[:S]

        head_dim = self.d_model // self.n_heads
        causal_mask = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)

        for layer_idx in range(self.n_layers):
            # 1. Masked Self-Attention
            qkv = np.matmul(h, self.w_self_qkv[layer_idx])
            q, k, v = np.split(qkv, 3, axis=-1)

            Q = q.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            K = k.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            V = v.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            scores = (
                np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim) + causal_mask
            )
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

            out_self = (
                np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            )
            h = h + np.matmul(out_self, self.w_self_proj[layer_idx])

            # 2. Cross-Attention over audio encoder features
            q_cross = np.matmul(h, self.w_cross_q[layer_idx])
            kv_cross = np.matmul(audio_features, self.w_cross_kv[layer_idx])
            k_c, v_c = np.split(kv_cross, 2, axis=-1)

            Q_c = q_cross.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            K_c = k_c.reshape(B, T_enc, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            V_c = v_c.reshape(B, T_enc, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            scores_c = np.matmul(Q_c, K_c.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            exp_sc = np.exp(scores_c - np.max(scores_c, axis=-1, keepdims=True))
            attn_c = exp_sc / (np.sum(exp_sc, axis=-1, keepdims=True) + 1e-12)

            out_cross = (
                np.matmul(attn_c, V_c).transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            )
            h = h + np.matmul(out_cross, self.w_cross_proj[layer_idx])

            # 3. FFN
            h_mlp = _gelu(np.matmul(h, self.w_mlp1[layer_idx]))
            h = h + np.matmul(h_mlp, self.w_mlp2[layer_idx])

        # Vocabulary logits
        return np.matmul(h, self.w_head)


class Whisper:
    """OpenAI Whisper Speech Recognition & Translation Transformer in pure NumPy.

    Encoder-Decoder architecture mapping log-mel spectrograms to text tokens.
    """

    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 500,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        seed: int = 42,
    ) -> None:
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.encoder = WhisperEncoder(
            n_mels=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            seed=seed,
        )
        self.decoder = WhisperDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            seed=seed + 1,
        )

    def encode(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Extracts contextual audio representations (B, T_enc, d_model)."""
        return self.encoder.forward(mel_spectrogram)

    def forward(
        self, mel_spectrogram: np.ndarray, text_tokens: np.ndarray
    ) -> np.ndarray:
        """Full forward pass returning vocabulary logits (B, seq_len, vocab_size)."""
        audio_features = self.encode(mel_spectrogram)
        return self.decoder.forward(text_tokens, audio_features)

    def generate(
        self,
        mel_spectrogram: np.ndarray,
        max_len: int = 20,
        prompt_tokens: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Autoregressively generates text token sequences from audio spectrogram."""
        audio_features = self.encode(mel_spectrogram)
        B = audio_features.shape[0]

        curr_tokens = (
            np.array([list(prompt_tokens)] * B, dtype=int)
            if prompt_tokens is not None and len(prompt_tokens) > 0
            else np.zeros((B, 1), dtype=int)
        )

        for _ in range(max_len):
            logits = self.decoder.forward(curr_tokens, audio_features)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            curr_tokens = np.concatenate([curr_tokens, next_token], axis=1)

        return curr_tokens
