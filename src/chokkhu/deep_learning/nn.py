import math
from typing import List
from .tensor import Tensor
from .backend import xp


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = xp.zeros_like(p.grad)

    def parameters(self) -> List[Tensor]:
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # Kaiming uniform initialization
        bound = 1 / math.sqrt(in_features)

        weight_data = xp.random.uniform(-bound, bound, (in_features, out_features))
        self.weight = Tensor(weight_data, requires_grad=True)

        if bias:
            bias_data = xp.random.uniform(-bound, bound, (out_features,))
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> List[Tensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
