from typing import List
from .tensor import Tensor
from .backend import xp


class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = xp.zeros_like(p.grad)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class AdamW(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

            grad = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
