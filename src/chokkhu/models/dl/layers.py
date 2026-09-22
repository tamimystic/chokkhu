from __future__ import annotations

from typing import Any, Dict, List, Mapping, Union
import numpy as np
from chokkhu.core.tensor import Tensor


class Parameter(Tensor):
    """Learnable model parameter Tensor."""

    def __init__(self, data: Union[int, float, list, np.ndarray]) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for all Neural Network Modules and Layers."""

    def __init__(self) -> None:
        self.training = True
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for p in self._parameters.values():
            if p.requires_grad:
                params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def state_dict(self, prefix: str = "") -> Dict[str, Tensor]:
        """Return a dictionary containing all named parameters in the module."""
        state: Dict[str, Tensor] = {}
        for name, param in self._parameters.items():
            key = f"{prefix}{name}" if prefix else name
            state[key] = param
        for name, mod in self._modules.items():
            mod_prefix = f"{prefix}{name}." if prefix else f"{name}."
            state.update(mod.state_dict(prefix=mod_prefix))
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Union[Tensor, np.ndarray]],
        strict: bool = True,
    ) -> None:
        """Copy parameters from state_dict into this module and its submodules."""
        current_state = self.state_dict()
        for name, param in current_state.items():
            if name in state_dict:
                val = state_dict[name]
                arr = val.data if isinstance(val, Tensor) else np.asarray(val)
                if param.data.shape != arr.shape:
                    raise ValueError(
                        f"Size mismatch for {name}: copying param with shape {arr.shape}, target model shape is {param.data.shape}."
                    )
                param.data[...] = arr.astype(param.data.dtype)
            elif strict:
                raise KeyError(f"Missing key in state_dict: {name}")

    def load_weights(self, filepath: Union[str, Any], strict: bool = False) -> None:
        """Load model weights directly from a .safetensors file."""
        from chokkhu.io.safetensors import load_safetensors

        tensors = load_safetensors(filepath)
        self.load_state_dict(tensors, strict=strict)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)


class Linear(Module):
    """Fully Connected (Dense) Linear Layer: Y = X W + b."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, init: str = "he"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if init == "he":
            w_data = np.random.randn(in_features, out_features) * np.sqrt(
                2.0 / in_features
            )
        elif init == "xavier":
            w_data = np.random.randn(in_features, out_features) * np.sqrt(
                2.0 / (in_features + out_features)
            )
        else:
            w_data = np.random.randn(in_features, out_features) * 0.01

        self.weight = Parameter(w_data)
        if bias:
            self.bias = Parameter(np.zeros((1, out_features)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


Dense = Linear


class Dropout(Module):
    """Inverted Dropout Layer."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = np.random.binomial(1, 1.0 - self.p, size=x.shape) / (1.0 - self.p)
            return x * Tensor(mask, requires_grad=False)
        return x


class BatchNorm1d(Module):
    """1D Batch Normalization Layer."""

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones((1, num_features)))
        self.beta = Parameter(np.zeros((1, num_features)))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0, keepdims=True)
            var = x.data.var(axis=0, keepdims=True)
            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var
            x_norm = (x - Tensor(mean, requires_grad=False)) / Tensor(
                np.sqrt(var + self.eps), requires_grad=False
            )
        else:
            x_norm = (x - Tensor(self.running_mean, requires_grad=False)) / Tensor(
                np.sqrt(self.running_var + self.eps), requires_grad=False
            )
        return x_norm * self.gamma + self.beta


class LayerNorm(Module):
    """Layer Normalization across normalized_shape."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(np.ones((1, normalized_shape)))
        self.beta = Parameter(np.zeros((1, normalized_shape)))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / ((variance + self.eps) ** 0.5)
        return x_norm * self.gamma + self.beta


class Flatten(Module):
    """Flattens input dimensions while preserving the batch size (N, -1)."""

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
