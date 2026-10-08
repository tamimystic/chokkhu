from __future__ import annotations

from typing import List

import numpy as np

from .layers import Parameter


class Optimizer:
    def __init__(
        self, params: List[Parameter], lr: float = 0.001, weight_decay: float = 0.0
    ) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with Momentum and Nesterov option."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                if self.nesterov:
                    step_val = grad + self.momentum * self.velocities[i]
                else:
                    step_val = self.velocities[i]
            else:
                step_val = grad

            p.data -= self.lr * step_val


class Adam(Optimizer):
    """Adam Optimizer."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1.0 - (self.beta1**self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2**self.t))

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW with decoupled weight decay."""

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1.0 - (self.beta1**self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2**self.t))

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """RMSProp Optimizer."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * (grad**2)
            p.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


class Adagrad(Optimizer):
    """Adagrad Optimizer."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.eps = eps
        self.accumulators = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.accumulators[i] += grad**2
            p.data -= self.lr * grad / (np.sqrt(self.accumulators[i]) + self.eps)


class Adadelta(Optimizer):
    """Adadelta optimizer (Zeiler, 2012)."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.rho = rho
        self.eps = eps
        self.square_avgs = [np.zeros_like(p.data) for p in self.params]
        self.acc_deltas = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.square_avgs[i] = self.rho * self.square_avgs[i] + (1.0 - self.rho) * (
                grad**2
            )
            std = np.sqrt(self.square_avgs[i] + self.eps)
            delta = np.sqrt(self.acc_deltas[i] + self.eps) / std * grad
            p.data -= self.lr * delta
            self.acc_deltas[i] = self.rho * self.acc_deltas[i] + (1.0 - self.rho) * (
                delta**2
            )


class NAdam(Optimizer):
    """NAdam (Nesterov-accelerated Adaptive Moment Estimation)."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1.0 - (self.beta1**self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2**self.t))
            g_hat = grad / (1.0 - (self.beta1**self.t))

            nesterov_m = self.beta1 * m_hat + (1.0 - self.beta1) * g_hat
            p.data -= self.lr * nesterov_m / (np.sqrt(v_hat) + self.eps)
