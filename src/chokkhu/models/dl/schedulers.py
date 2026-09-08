from __future__ import annotations

import numpy as np
from .optimizers import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return float(self.base_lr * (self.gamma ** (self.step_count // self.step_size)))


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return float(
            self.eta_min
            + 0.5
            * (self.base_lr - self.eta_min)
            * (1.0 + np.cos(np.pi * self.step_count / self.T_max))
        )
