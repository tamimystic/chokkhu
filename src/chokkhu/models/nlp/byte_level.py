"""Token-Free Raw UTF-8 Byte Sequence Modeling and Byte Transformers.

Formulated from first principles using fixed 256-byte vocabularies, local patch
convolutional compression, bidirectional/causal transformer processing, and byte unpooling in pure NumPy.
"""

from typing import Dict

import numpy as np


class ByteTransformer:
    """Token-Free Raw UTF-8 Byte Sequence Transformer (ByT5 / MambaByte Architecture).

    Operates directly on binary byte streams (0-255) with zero tokenizer failure modes,
    compressing byte sequences with local patch pooling.

    Parameters
    ----------
    d_model : int, default=64
        Transformer representation hidden dimension.
    patch_size : int, default=4
        Byte sequence compression factor via local patch projection.
    num_heads : int, default=4
        Number of self-attention heads.
    num_layers : int, default=2
        Number of transformer layers.
    max_seq_len : int, default=256
        Maximum allowable byte length.
    seed : int, default=42
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        d_model: int = 64,
        patch_size: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 256,
        seed: int = 42,
    ) -> None:
        self.raw_vocab_size = 256  # standard 8-bit bytes
        self.pad_byte = 256
        self.total_vocab = 257
        self.d_model = int(d_model)
        self.patch_size = max(1, int(patch_size))
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.params: Dict[str, np.ndarray] = {}
        self._init_network()

    def _init_network(self) -> None:
        scale = 1.0 / np.sqrt(self.d_model)
        # 1. Byte Embedding & Patch Projection
        self.params["byte_embed"] = (
            self.rng.randn(self.total_vocab, self.d_model) * scale
        )
        self.params["patch_proj_w"] = (
            self.rng.randn(self.patch_size * self.d_model, self.d_model) * scale
        )
        self.params["patch_proj_b"] = np.zeros(self.d_model)
        self.params["pos_embed"] = (
            self.rng.randn(self.max_seq_len // self.patch_size + 1, self.d_model)
            * scale
        )

        # 2. Transformer Layers
        for l_idx in range(self.num_layers):
            self.params[f"l{l_idx}_q"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_k"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_v"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_out"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_ffn1"] = (
                self.rng.randn(self.d_model, self.d_model * 2) * scale
            )
            self.params[f"l{l_idx}_ffn2"] = (
                self.rng.randn(self.d_model * 2, self.d_model) * scale
            )

        # 3. Unpooling & Byte Classification Head
        self.params["unpatch_w"] = (
            self.rng.randn(self.d_model, self.patch_size * self.d_model) * scale
        )
        self.params["unpatch_b"] = np.zeros(self.patch_size * self.d_model)
        self.params["head_w"] = (
            self.rng.randn(self.d_model, self.raw_vocab_size) * scale
        )
        self.params["head_b"] = np.zeros(self.raw_vocab_size)

    @staticmethod
    def text_to_bytes(text: str) -> np.ndarray:
        """Encode UTF-8 string into 1D NumPy array of uint8 bytes."""
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(int)

    @staticmethod
    def bytes_to_text(byte_array: np.ndarray, errors: str = "replace") -> str:
        """Decode NumPy byte array back into UTF-8 text string."""
        raw_bytes = bytes(np.clip(byte_array, 0, 255).astype(np.uint8).tolist())
        return raw_bytes.decode("utf-8", errors=errors)

    def forward(self, byte_sequences: np.ndarray) -> np.ndarray:
        """Forward pass converting raw byte sequences to per-byte next-byte logits.

        Parameters
        ----------
        byte_sequences : np.ndarray of shape (B, S)
            Input byte integer sequences in [0, 255].

        Returns
        -------
        logits : np.ndarray of shape (B, S, 256)
            Predicted logits over 256 possible next byte values.
        """
        B, S = byte_sequences.shape
        pad_len = (self.patch_size - (S % self.patch_size)) % self.patch_size
        if pad_len > 0:
            padded_bytes = np.pad(
                byte_sequences, ((0, 0), (0, pad_len)), constant_values=self.pad_byte
            )
        else:
            padded_bytes = byte_sequences

        B, S_pad = padded_bytes.shape
        n_patches = S_pad // self.patch_size

        # 1. Byte Embeddings: (B, S_pad, d_model)
        tok_emb = self.params["byte_embed"][padded_bytes]
        # Reshape to patches: (B, n_patches, patch_size * d_model)
        patch_flat = tok_emb.reshape(B, n_patches, self.patch_size * self.d_model)
        # Linear patch projection: (B, n_patches, d_model)
        h = (
            np.dot(patch_flat, self.params["patch_proj_w"])
            + self.params["patch_proj_b"]
        )

        # Positional embedding
        pos_idx = np.arange(n_patches) % self.params["pos_embed"].shape[0]
        h = h + self.params["pos_embed"][pos_idx]

        # 2. Transformer Processing
        head_dim = self.d_model // self.num_heads
        for l_idx in range(self.num_layers):
            q = (
                np.dot(h, self.params[f"l{l_idx}_q"])
                .reshape(B, n_patches, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )
            k = (
                np.dot(h, self.params[f"l{l_idx}_k"])
                .reshape(B, n_patches, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )
            v = (
                np.dot(h, self.params[f"l{l_idx}_v"])
                .reshape(B, n_patches, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )

            scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(head_dim)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            attn_weights = np.exp(scores - scores_max)
            attn_weights = attn_weights / (
                np.sum(attn_weights, axis=-1, keepdims=True) + 1e-12
            )

            attn_out = (
                np.matmul(attn_weights, v)
                .swapaxes(1, 2)
                .reshape(B, n_patches, self.d_model)
            )
            h = h + np.dot(attn_out, self.params[f"l{l_idx}_out"])
            h = (h - np.mean(h, axis=-1, keepdims=True)) / (
                np.std(h, axis=-1, keepdims=True) + 1e-6
            )

            ffn = np.maximum(0.0, np.dot(h, self.params[f"l{l_idx}_ffn1"]))
            h = h + np.dot(ffn, self.params[f"l{l_idx}_ffn2"])
            h = (h - np.mean(h, axis=-1, keepdims=True)) / (
                np.std(h, axis=-1, keepdims=True) + 1e-6
            )

        # 3. Unpooling / Patch Expansion: (B, S_pad, d_model)
        unpatched = np.dot(h, self.params["unpatch_w"]) + self.params["unpatch_b"]
        unpooled = unpatched.reshape(B, S_pad, self.d_model)

        # Output Logits: (B, S, 256)
        logits_padded = np.dot(unpooled, self.params["head_w"]) + self.params["head_b"]
        logits = logits_padded[:, :S, :]
        return logits

    def generate(
        self,
        prompt: str,
        max_new_bytes: int = 16,
        temperature: float = 1.0,
    ) -> str:
        """Autoregressively generate continuation directly in byte space."""
        bytes_in = self.text_to_bytes(prompt)
        if len(bytes_in) == 0:
            bytes_in = np.array([32])  # space

        curr_bytes = list(bytes_in)
        for _ in range(max_new_bytes):
            inp = np.array([curr_bytes[-self.max_seq_len :]])
            logits = self.forward(inp)[0, -1, :] / max(temperature, 1e-5)
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            next_byte = int(self.rng.choice(256, p=probs))
            curr_bytes.append(next_byte)

        return self.bytes_to_text(np.array(curr_bytes))
