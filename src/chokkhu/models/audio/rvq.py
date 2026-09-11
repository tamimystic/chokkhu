"""Residual Vector Quantization (RVQ) for Neural Audio Codecs in pure NumPy."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class VectorQuantizerStage:
    """Single Vector Quantization Stage with codebook dictionary learning."""

    def __init__(
        self, codebook_size: int = 1024, embed_dim: int = 64, seed: int = 42
    ) -> None:
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        # Initialize codebook embedding vectors
        self.embedding = rng.normal(
            0, 1.0 / np.sqrt(embed_dim), size=(codebook_size, embed_dim)
        ).astype(np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantizes input continuous tensor to nearest codebook vectors.

        Args:
            x: Input continuous tensor of shape (B, T, D) or (B, D).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - Quantized vectors of same shape as x.
                - Code indices of shape (B, T) or (B,).
                - Commitment loss.
        """
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.embed_dim)  # (N, D)

        # Compute Euclidean distance: ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x e^T
        x_sq = np.sum(flat_x**2, axis=1, keepdims=True)  # (N, 1)
        e_sq = np.sum(self.embedding**2, axis=1, keepdims=True).T  # (1, K)
        dist = x_sq + e_sq - 2.0 * np.matmul(flat_x, self.embedding.T)  # (N, K)

        indices = np.argmin(dist, axis=1)  # (N,)
        quantized = self.embedding[indices]  # (N, D)

        # Commitment loss: ||sg(e) - x||^2 + 0.25 * ||e - sg(x)||^2
        commitment_loss = float(np.mean((flat_x - quantized) ** 2))

        # Straight-Through Estimator in pure NumPy:
        # Forward pass uses quantized, backward pass copies gradients
        quantized_out = flat_x + (quantized - flat_x)
        quantized_out = quantized_out.reshape(orig_shape)

        indices_out = indices.reshape(orig_shape[:-1])
        return quantized_out, indices_out, commitment_loss

    def decode_indices(self, indices: np.ndarray) -> np.ndarray:
        """Looks up continuous embeddings for code indices."""
        idx = np.asarray(indices, dtype=int)
        return self.embedding[idx]


class ResidualVectorQuantizer:
    """Residual Vector Quantizer (RVQ) for Neural Audio Compression (SoundStream/EnCodec).

    Cascades N_q quantizer stages where each stage quantizes the residual
    error left by previous stages, enabling massive bandwidth compression
    and discrete tokenization.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        embed_dim: int = 64,
        commitment_weight: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = float(commitment_weight)

        self.quantizers = [
            VectorQuantizerStage(
                codebook_size=codebook_size,
                embed_dim=embed_dim,
                seed=seed + i,
            )
            for i in range(num_quantizers)
        ]

    def forward(
        self,
        x: np.ndarray,
        num_stages: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantizes continuous representations through cascaded residual stages.

        Args:
            x: Continuous input features (B, T, D) or (B, D).
            num_stages: Optional subset of quantizers to use (for bitrate scalability).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - Total quantized representation z_q.
                - Discrete code tokens matrix of shape (num_stages, B, T).
                - Total commitment loss across stages.
        """
        n_stages = (
            self.num_quantizers
            if num_stages is None
            else min(num_stages, self.num_quantizers)
        )

        residual = x.copy()
        quantized_sum = np.zeros_like(x)
        all_codes: List[np.ndarray] = []
        total_loss = 0.0

        for i in range(n_stages):
            q_stage = self.quantizers[i]
            z_q_i, indices_i, loss_i = q_stage.forward(residual)

            quantized_sum += z_q_i
            residual -= z_q_i
            all_codes.append(indices_i)
            total_loss += loss_i

        codes = np.stack(all_codes, axis=0)  # (n_stages, ...)
        return quantized_sum, codes, float(total_loss * self.commitment_weight)

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Reconstructs continuous representations from discrete token codes.

        Args:
            codes: Array of shape (n_stages, B, T) or (n_stages, B).

        Returns:
            Reconstructed continuous latent representation (B, T, D) or (B, D).
        """
        n_stages = codes.shape[0]
        reconstructed: Optional[np.ndarray] = None

        for i in range(n_stages):
            z_i = self.quantizers[i].decode_indices(codes[i])
            if reconstructed is None:
                reconstructed = z_i
            else:
                reconstructed = reconstructed + z_i

        assert reconstructed is not None
        return reconstructed
