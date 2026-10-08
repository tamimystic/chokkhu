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


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.95) -> None:
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return float(self.base_lr * (self.gamma**self.step_count))


class InverseScalingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, power_t: float = 0.5) -> None:
        self.power_t = power_t
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return float(self.base_lr / ((self.step_count + 1) ** self.power_t))


class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0,
        end_factor: float = 0.0,
        total_iters: int = 100,
    ) -> None:
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer)

    def get_lr(self) -> float:
        frac = min(1.0, self.step_count / max(1, self.total_iters))
        factor = self.start_factor + (self.end_factor - self.start_factor) * frac
        return float(self.base_lr * factor)
