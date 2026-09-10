"""CLIP: Contrastive Language-Image Pre-training.

Pure NumPy implementation of OpenAI CLIP dual-encoder architecture:
- Vision Encoder: Patch-based Vision Transformer projection
- Text Encoder: Token embedding + Multi-Head Self-Attention Transformer
- Symmetric InfoNCE / Cross-Entropy contrastive loss with learnable temperature
- Zero-shot classification, cross-modal retrieval, and feature extraction
"""

import math
from typing import Optional, Tuple
import numpy as np


class _MultiHeadAttention:
    """Internal Multi-Head Self-Attention block."""

    def __init__(
        self, embed_dim: int, num_heads: int, seed: Optional[int] = 42
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        rng = np.random.RandomState(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        self.w_q = (rng.randn(embed_dim, embed_dim) * scale).astype(np.float32)
        self.w_k = (rng.randn(embed_dim, embed_dim) * scale).astype(np.float32)
        self.w_v = (rng.randn(embed_dim, embed_dim) * scale).astype(np.float32)
        self.w_out = (rng.randn(embed_dim, embed_dim) * scale).astype(np.float32)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # x: (B, S, D)
        B, S, D = x.shape
        Q = (
            np.dot(x, self.w_q)
            .reshape(B, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )  # (B, H, S, d)
        K = (
            np.dot(x, self.w_k)
            .reshape(B, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        V = (
            np.dot(x, self.w_v)
            .reshape(B, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        scores = np.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, H, S, S)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.maximum(
            np.sum(exp_scores, axis=-1, keepdims=True), 1e-12
        )

        context = np.matmul(attn_weights, V)  # (B, H, S, d)
        context = context.swapaxes(1, 2).reshape(B, S, D)
        return np.dot(context, self.w_out)


class _TransformerEncoderLayer:
    """Internal Transformer encoder layer with LayerNorm and FeedForward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        seed: Optional[int] = 42,
    ) -> None:
        self.mha = _MultiHeadAttention(embed_dim, num_heads, seed=seed)
        mlp_dim = embed_dim * mlp_ratio
        rng = np.random.RandomState(seed)
        scale1 = 1.0 / math.sqrt(embed_dim)
        scale2 = 1.0 / math.sqrt(mlp_dim)
        self.w1 = (rng.randn(embed_dim, mlp_dim) * scale1).astype(np.float32)
        self.b1: np.ndarray = np.zeros(mlp_dim, dtype=np.float32)
        self.w2 = (rng.randn(mlp_dim, embed_dim) * scale2).astype(np.float32)
        self.b2: np.ndarray = np.zeros(embed_dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-LN MHA
        norm_x = self._layer_norm(x)
        attn_out = self.mha.forward(norm_x, mask=mask)
        x = x + attn_out

        # Pre-LN MLP
        norm_x2 = self._layer_norm(x)
        mlp_act = np.maximum(0.0, np.dot(norm_x2, self.w1) + self.b1)  # ReLU / GELU
        mlp_out = np.dot(mlp_act, self.w2) + self.b2
        return x + mlp_out


class CLIPVisionEncoder:
    """Vision Transformer patch encoder for CLIP."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        projection_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        seed: Optional[int] = 42,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        rng = np.random.RandomState(seed)
        self.patch_proj = (
            rng.randn(patch_dim, embed_dim) * (1.0 / math.sqrt(patch_dim))
        ).astype(np.float32)
        self.cls_token = (rng.randn(1, 1, embed_dim) * 0.02).astype(np.float32)
        self.pos_embed = (rng.randn(1, num_patches + 1, embed_dim) * 0.02).astype(
            np.float32
        )

        self.layers = [
            _TransformerEncoderLayer(
                embed_dim, num_heads, seed=seed + i if seed else None
            )
            for i in range(num_layers)
        ]
        self.head_proj = (
            rng.randn(embed_dim, projection_dim) * (1.0 / math.sqrt(embed_dim))
        ).astype(np.float32)

    def _patchify(self, images: np.ndarray) -> np.ndarray:
        # images: (B, C, H, W) or (B, H, W, C)
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.shape[1] != self.in_channels and arr.shape[-1] == self.in_channels:
            # (B, H, W, C) -> (B, C, H, W)
            arr = np.transpose(arr, (0, 3, 1, 2))

        B, C, H, W = arr.shape
        P = self.patch_size
        num_h = H // P
        num_w = W // P

        # Extract non-overlapping patches
        patches = []
        for i in range(num_h):
            for j in range(num_w):
                p = arr[:, :, i * P : (i + 1) * P, j * P : (j + 1) * P]
                patches.append(p.reshape(B, -1))
        # (B, num_patches, patch_dim)
        return np.stack(patches, axis=1)

    def forward(self, images: np.ndarray) -> np.ndarray:
        patches = self._patchify(images)  # (B, N, patch_dim)
        B = patches.shape[0]

        patch_embeds = np.dot(patches, self.patch_proj)  # (B, N, embed_dim)
        cls_tokens = np.repeat(self.cls_token, B, axis=0)  # (B, 1, embed_dim)
        x = (
            np.concatenate([cls_tokens, patch_embeds], axis=1)
            + self.pos_embed[:, : patch_embeds.shape[1] + 1]
        )

        for layer in self.layers:
            x = layer.forward(x)

        # Extract CLS token representation
        cls_repr = x[:, 0]  # (B, embed_dim)
        proj = np.dot(cls_repr, self.head_proj)  # (B, projection_dim)
        # Normalize embeddings to unit sphere
        norm = np.maximum(np.linalg.norm(proj, axis=-1, keepdims=True), 1e-12)
        return proj / norm


class CLIPTextEncoder:
    """Transformer text encoder for CLIP."""

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 64,
        embed_dim: int = 256,
        projection_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        seed: Optional[int] = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        rng = np.random.RandomState(seed)
        self.token_embed = (rng.randn(vocab_size, embed_dim) * 0.02).astype(np.float32)
        self.pos_embed = (rng.randn(1, max_seq_len, embed_dim) * 0.02).astype(
            np.float32
        )

        self.layers = [
            _TransformerEncoderLayer(
                embed_dim, num_heads, seed=seed + 100 + i if seed else None
            )
            for i in range(num_layers)
        ]
        self.head_proj = (
            rng.randn(embed_dim, projection_dim) * (1.0 / math.sqrt(embed_dim))
        ).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        # input_ids: (B, S) integer token ids
        ids = np.asarray(input_ids, dtype=np.int32)
        if ids.ndim == 1:
            ids = ids[None, :]

        B, S = ids.shape
        ids_clipped = np.clip(ids, 0, self.vocab_size - 1)
        x = self.token_embed[ids_clipped] + self.pos_embed[:, :S]

        for layer in self.layers:
            x = layer.forward(x)

        # Global average pooling over token sequence
        text_repr = np.mean(x, axis=1)  # (B, embed_dim)
        proj = np.dot(text_repr, self.head_proj)  # (B, projection_dim)
        norm = np.maximum(np.linalg.norm(proj, axis=-1, keepdims=True), 1e-12)
        return proj / norm


class CLIP:
    r"""CLIP: Contrastive Language-Image Pre-Training Model.

        Dual-encoder architecture optimizing symmetric cross-entropy infoNCE loss:

        .. math::
            \mathcal{L} =
    rac{1}{2} (\mathcal{L}_{I 	o T} + \mathcal{L}_{T 	o I})

        Parameters
        ----------
        embed_dim : int, default=128
            Shared projection embedding dimension.
        image_size : int, default=32
            Input image resolution.
        patch_size : int, default=8
            Vision transformer patch size.
        vocab_size : int, default=500
            Vocabulary size for text encoder.
        init_temperature : float, default=0.07
            Initial temperature parameter.
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        image_size: int = 32,
        patch_size: int = 8,
        vocab_size: int = 500,
        init_temperature: float = 0.07,
        seed: Optional[int] = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.logit_scale: float = float(math.log(1.0 / max(init_temperature, 1e-5)))

        self.vision_encoder = CLIPVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim * 2,
            projection_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            seed=seed,
        )
        self.text_encoder = CLIPTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim * 2,
            projection_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            seed=seed + 1 if seed else None,
        )

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Extract unit-normalized image embeddings."""
        return self.vision_encoder.forward(images)

    def encode_text(self, text_tokens: np.ndarray) -> np.ndarray:
        """Extract unit-normalized text embeddings."""
        return self.text_encoder.forward(text_tokens)

    def forward(
        self,
        images: np.ndarray,
        text_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute image-text similarity matrix and contrastive loss."""
        img_features = self.encode_image(images)  # (N, D)
        text_features = self.encode_text(text_tokens)  # (N, D)

        temperature = math.exp(self.logit_scale)
        # Cosine similarity matrix scaled by temperature
        logits_per_image = np.dot(img_features, text_features.T) * temperature  # (N, N)
        logits_per_text = logits_per_image.T  # (N, N)

        # Symmetric cross-entropy contrastive loss
        n = logits_per_image.shape[0]
        labels = np.arange(n)

        # Image-to-text loss
        exp_i = np.exp(
            logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True)
        )
        probs_i = exp_i / np.maximum(np.sum(exp_i, axis=-1, keepdims=True), 1e-12)
        loss_i = -np.mean(np.log(np.maximum(probs_i[np.arange(n), labels], 1e-12)))

        # Text-to-image loss
        exp_t = np.exp(
            logits_per_text - np.max(logits_per_text, axis=-1, keepdims=True)
        )
        probs_t = exp_t / np.maximum(np.sum(exp_t, axis=-1, keepdims=True), 1e-12)
        loss_t = -np.mean(np.log(np.maximum(probs_t[np.arange(n), labels], 1e-12)))

        total_loss = float(0.5 * (loss_i + loss_t))
        return logits_per_image, logits_per_text, total_loss

    def predict_proba(
        self,
        images: np.ndarray,
        candidate_text_tokens: np.ndarray,
    ) -> np.ndarray:
        """Perform zero-shot classification probabilities across candidate text prompts."""
        img_features = self.encode_image(images)  # (N, D)
        text_features = self.encode_text(candidate_text_tokens)  # (K, D)

        temperature = math.exp(self.logit_scale)
        logits = np.dot(img_features, text_features.T) * temperature  # (N, K)

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.maximum(
            np.sum(exp_logits, axis=-1, keepdims=True), 1e-12
        )
        return probs
