"""Knowledge Distillation and Teacher-Student Model Compression.

Pure NumPy implementations of:
- KnowledgeDistiller: Hinton's KD loss (Soft KL Divergence + Task Cross-Entropy/MSE)
- FeatureDistiller: Intermediate Hint-Layer feature representation matching
"""

from typing import Optional, Tuple
import numpy as np


class KnowledgeDistiller:
    r"""Hinton Knowledge Distillation Engine.

    Transfers dark knowledge from a large teacher network to a compact student network:

    .. math::
        \mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{task}}(z_s, y) +
        \alpha T^2 D_{\text{KL}}\left( \sigma(z_t / T) \parallel \sigma(z_s / T) \right)

    Parameters
    ----------
    temperature : float, default=2.0
        Softmax smoothing temperature :math:`T > 0`.
    alpha : float, default=0.5
        Balance factor between task loss and soft distillation loss :math:`\alpha \in [0.0, 1.0]`.
    task_type : str, default="classification"
        Task loss type: "classification" (Cross-Entropy) or "regression" (MSE).
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        task_type: str = "classification",
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0.0, 1.0].")
        if task_type not in ("classification", "regression"):
            raise ValueError(f"Unknown task_type: '{task_type}'")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.task_type = task_type

    def _softmax(self, z: np.ndarray, temp: float = 1.0) -> np.ndarray:
        scaled = z / temp
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-12)

    def compute_loss(
        self,
        student_logits: np.ndarray,
        teacher_logits: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute (total_loss, task_loss, distillation_loss)."""
        zs = np.asarray(student_logits, dtype=np.float32)
        zt = np.asarray(teacher_logits, dtype=np.float32)
        yt = np.asarray(y_true)

        if self.task_type == "classification":
            # Task loss: standard cross-entropy on student logits
            p_s = self._softmax(zs, temp=1.0)
            n_samples = zs.shape[0]

            if yt.ndim == 1:
                # Integer labels
                log_probs = np.log(p_s[np.arange(n_samples), yt.astype(int)] + 1e-12)
                task_loss = -float(np.mean(log_probs))
            else:
                # One-hot labels
                task_loss = -float(np.mean(np.sum(yt * np.log(p_s + 1e-12), axis=-1)))

            # Distillation soft KL loss
            soft_t = self._softmax(zt, temp=self.temperature)
            soft_s = self._softmax(zs, temp=self.temperature)

            # KL(T || S) = sum( T * (log(T) - log(S)) )
            kl = np.sum(
                soft_t * (np.log(soft_t + 1e-12) - np.log(soft_s + 1e-12)), axis=-1
            )
            distill_loss = float(np.mean(kl)) * (self.temperature**2)

        else:
            # Regression task
            task_loss = float(np.mean((zs - yt) ** 2))
            distill_loss = float(np.mean((zs - zt) ** 2))

        total_loss = (1.0 - self.alpha) * task_loss + self.alpha * distill_loss
        return total_loss, task_loss, distill_loss


class FeatureDistiller:
    r"""Hint-Layer Intermediate Feature Distillation (Romero et al., FitNets).

    Matches intermediate feature representations between student and teacher:

    .. math::
        \mathcal{L}_{\text{hint}} = \frac{1}{2 N} \sum_{i=1}^N \| W_{\text{proj}} F_s^{(i)} - F_t^{(i)} \|_2^2
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        # Projection adapter if dimensions differ
        if student_dim != teacher_dim:
            self.proj: Optional[np.ndarray] = (
                np.random.randn(student_dim, teacher_dim).astype(np.float32) * 0.01
            )
        else:
            self.proj = None

    def compute_feature_loss(
        self,
        student_features: np.ndarray,
        teacher_features: np.ndarray,
    ) -> float:
        """Compute MSE feature alignment loss."""
        fs = np.asarray(student_features, dtype=np.float32)
        ft = np.asarray(teacher_features, dtype=np.float32)

        if self.proj is not None:
            fs_projected = fs @ self.proj
        else:
            fs_projected = fs

        loss = 0.5 * float(np.mean(np.sum((fs_projected - ft) ** 2, axis=-1)))
        return loss
