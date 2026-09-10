"""SigLIP: Sigmoid Loss for Language-Image Pre-Training.

Pure NumPy implementation of Google SigLIP (Zhai et al., 2023):
- Replaces softmax denominator with pairwise sigmoid loss:
    L = - 1/N * sum_{i,j} log(sigmoid(z_{ij} * (t * sim(I_i, T_j) + b)))
- Enables scalable, decoupled multimodal contrastive learning
"""

from typing import Optional, Tuple
import numpy as np
from .clip import CLIPVisionEncoder, CLIPTextEncoder


class SigLIP:
    r"""SigLIP: Sigmoid Language-Image Pre-training Model.

    Optimizes binary logistic loss over all :math:`N \times N` image-text pairs:

    .. math::
        \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \ln \sigma(y_{ij} (t \cdot \mathbf{u}_i^T \mathbf{v}_j + b))

    where :math:`y_{ij} = +1` for matched pairs (:math:`i=j`) and :math:`-1` for unmatched pairs (:math:`i \neq j`).

    Parameters
    ----------
    embed_dim : int, default=128
        Shared projection dimension.
    image_size : int, default=32
        Input image resolution.
    patch_size : int, default=8
        Patch size.
    vocab_size : int, default=500
        Vocab size.
    init_temp : float, default=10.0
        Initial temperature parameter.
    init_bias : float, default=-10.0
        Initial bias parameter.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        image_size: int = 32,
        patch_size: int = 8,
        vocab_size: int = 500,
        init_temp: float = 10.0,
        init_bias: float = -10.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size

        self.temp = float(init_temp)
        self.bias = float(init_bias)

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
        """Encode images into normalized embeddings."""
        return self.vision_encoder.forward(images)

    def encode_text(self, text_tokens: np.ndarray) -> np.ndarray:
        """Encode text tokens into normalized embeddings."""
        return self.text_encoder.forward(text_tokens)

    def forward(
        self,
        images: np.ndarray,
        text_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute pairwise logits and SigLIP sigmoid loss."""
        img_features = self.encode_image(images)  # (N, D)
        text_features = self.encode_text(text_tokens)  # (N, D)

        # Dot product similarities in [-1, 1]
        sim = np.dot(img_features, text_features.T)  # (N, N)
        logits = sim * self.temp + self.bias  # (N, N)

        n = sim.shape[0]
        # Labels: +1 on diagonal (matched), -1 off-diagonal (unmatched)
        targets = np.eye(n, dtype=np.float32) * 2.0 - 1.0  # +1 and -1

        # Sigmoid binary cross entropy: log(1 + exp(-targets * logits))
        z = targets * logits
        # Numerically stable softplus log(1 + exp(-z))
        loss_matrix = np.where(z > 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z)))
        loss = float(np.sum(loss_matrix) / n)

        return logits, loss

    def predict_proba(
        self,
        images: np.ndarray,
        candidate_text_tokens: np.ndarray,
    ) -> np.ndarray:
        """Predict zero-shot sigmoid probabilities."""
        img_features = self.encode_image(images)  # (N, D)
        text_features = self.encode_text(candidate_text_tokens)  # (K, D)

        sim = np.dot(img_features, text_features.T)  # (N, K)
        logits = sim * self.temp + self.bias
        # Sigmoid per candidate
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
