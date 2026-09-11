"""Dark Experience Replay (DER++) and Reservoir Memory for Continual Learning in pure NumPy."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


class DarkExperienceReplay:
    """Dark Experience Replay (DER++) Continual Learning Engine in pure NumPy (Buzzega et al. 2020).

    Maintains a reservoir replay memory storing past inputs, labels, and output logits,
    applying knowledge distillation to prevent catastrophic forgetting.

    Parameters
    ----------
    buffer_capacity : int, default=500
        Maximum number of exemplar samples stored in reservoir memory.
    alpha : float, default=0.5
        Distillation loss weight regularizing current student logits against stored teacher logits.
    beta : float, default=0.5
        Replay task classification loss weight.
    seed : int, default=42
    """

    def __init__(
        self,
        buffer_capacity: int = 500,
        alpha: float = 0.5,
        beta: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.capacity = int(buffer_capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)

        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.buffer_logits: List[np.ndarray] = []
        self.total_seen_samples: int = 0

    def add_sample(self, x: np.ndarray, y: int, logits: np.ndarray) -> None:
        """Inserts a sample into memory using Reservoir Sampling."""
        self.total_seen_samples += 1
        x_flat = np.asarray(x, dtype=np.float32)
        logits_flat = np.asarray(logits, dtype=np.float32)

        if len(self.buffer_X) < self.capacity:
            self.buffer_X.append(x_flat)
            self.buffer_y.append(int(y))
            self.buffer_logits.append(logits_flat)
        else:
            # Reservoir replacement with probability capacity / total_seen
            rand_idx = self.rng.integers(0, self.total_seen_samples)
            if rand_idx < self.capacity:
                self.buffer_X[rand_idx] = x_flat
                self.buffer_y[rand_idx] = int(y)
                self.buffer_logits[rand_idx] = logits_flat

    def add_batch(self, X: np.ndarray, y: np.ndarray, logits: np.ndarray) -> None:
        """Inserts a batch of samples into reservoir replay memory."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        logits_arr = np.asarray(logits, dtype=np.float32)

        for i in range(len(X_arr)):
            self.add_sample(X_arr[i], int(y_arr[i]), logits_arr[i])

    def sample(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draws a random mini-batch of stored exemplars and past logits.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (sampled_X, sampled_y, sampled_teacher_logits)
        """
        if len(self.buffer_X) == 0:
            raise ValueError("Cannot sample from an empty DER++ buffer.")

        n = min(batch_size, len(self.buffer_X))
        indices = self.rng.choice(len(self.buffer_X), size=n, replace=False)

        sub_X = np.stack([self.buffer_X[i] for i in indices], axis=0)
        sub_y = np.array([self.buffer_y[i] for i in indices], dtype=int)
        sub_logits = np.stack([self.buffer_logits[i] for i in indices], axis=0)

        return sub_X, sub_y, sub_logits

    def distillation_loss(
        self, student_logits: np.ndarray, teacher_logits: np.ndarray
    ) -> float:
        """Computes Mean Squared Error logit distillation loss: ||z_student - z_teacher||^2."""
        s = np.asarray(student_logits, dtype=np.float32)
        t = np.asarray(teacher_logits, dtype=np.float32)
        return float(np.mean((s - t) ** 2))

    def compute_loss(
        self,
        current_task_loss: float,
        student_replay_logits: np.ndarray,
        teacher_replay_logits: np.ndarray,
        replay_task_loss: float = 0.0,
    ) -> float:
        """Combines current task loss with DER++ distillation and replay loss."""
        distill_loss = self.distillation_loss(
            student_replay_logits, teacher_replay_logits
        )
        total = (
            current_task_loss + self.alpha * distill_loss + self.beta * replay_task_loss
        )
        return float(total)

    def __len__(self) -> int:
        return len(self.buffer_X)


DERPlusPlus = DarkExperienceReplay
