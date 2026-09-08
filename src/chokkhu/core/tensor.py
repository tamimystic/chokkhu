from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Union
import numpy as np


class Tensor:
    """Pure NumPy Autograd Tensor with Dynamic Computational Graph and Automatic Differentiation."""

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray],
        requires_grad: bool = False,
        creator: Optional[Function] = None,
    ) -> None:
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64) if data.dtype != np.float64 else data
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

        self.requires_grad = requires_grad
        self.creator = creator
        self.grad: Optional[np.ndarray] = None
        self._generation = 0 if creator is None else creator.generation + 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def backward(self, grad: Optional[Union[np.ndarray, Tensor]] = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.shape == () or self.shape == (1,):
                grad_arr: np.ndarray = np.ones_like(self.data, dtype=np.float64)
            else:
                raise RuntimeError(
                    "Grad can only be implicitly created for scalar outputs"
                )
        elif isinstance(grad, Tensor):
            grad_arr = grad.data
        else:
            grad_arr = grad

        if self.grad is None:
            self.grad = grad_arr.copy()
        else:
            self.grad += grad_arr

        funcs: List[Function] = []
        seen_funcs: Set[Function] = set()

        def add_func(f: Optional[Function]) -> None:
            if f is not None and f not in seen_funcs:
                seen_funcs.add(f)
                funcs.append(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [
                output.grad
                for output in f.outputs
                if output is not None and output.grad is not None
            ]
            if not gys:
                continue
            gy = gys[0]
            gxs = f.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if gx is not None and x.requires_grad:
                    if x.grad is None:
                        x.grad = gx.copy()
                    else:
                        x.grad += gx
                    add_func(x.creator)

    def __add__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Add()(self, _as_tensor(other))

    def __radd__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Add()(_as_tensor(other), self)

    def __sub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Sub()(self, _as_tensor(other))

    def __rsub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Sub()(_as_tensor(other), self)

    def __mul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Mul()(self, _as_tensor(other))

    def __rmul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Mul()(_as_tensor(other), self)

    def __truediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Div()(self, _as_tensor(other))

    def __rtruediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return Div()(_as_tensor(other), self)

    def __matmul__(self, other: Tensor) -> Tensor:
        return MatMul()(self, _as_tensor(other))

    def __neg__(self) -> Tensor:
        return Neg()(self)

    def __pow__(self, power: float) -> Tensor:
        return Pow(power)(self)

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Tensor:
        return Sum(axis=axis, keepdims=keepdims)(self)

    def mean(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        numel = (
            self.size
            if axis is None
            else (
                self.data.shape[axis]
                if isinstance(axis, int)
                else int(np.prod([self.data.shape[a] for a in axis]))
            )
        )
        return s / float(max(1, numel))

    def reshape(self, *shape: Any) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Reshape(shape)(self)

    def transpose(self, *axes: Any) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])
        return Transpose(axes if axes else None)(self)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def relu(self) -> Tensor:
        return ReLU()(self)

    def sigmoid(self) -> Tensor:
        return Sigmoid()(self)

    def tanh(self) -> Tensor:
        return Tanh()(self)

    def gelu(self) -> Tensor:
        return GELU()(self)

    def __repr__(self) -> str:
        grad_fn = (
            f", grad_fn=<{self.creator.__class__.__name__}>"
            if self.creator is not None
            else ""
        )
        req_grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_fn}{req_grad})"


def _as_tensor(obj: Any) -> Tensor:
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


def _unbroadcast_grad(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == target_shape:
        return grad
    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    for _ in range(grad_ndim - target_ndim):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(target_shape)


class Function:
    """Base computational graph node representing a differentiable mathematical operation."""

    def __init__(self) -> None:
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.generation = 0

    def __call__(self, *inputs: Tensor) -> Tensor:
        self.inputs = list(inputs)
        self.generation = max([x._generation for x in inputs]) if inputs else 0
        xs = [x.data for x in inputs]
        y_data = self.forward(*xs)
        requires_grad = any(x.requires_grad for x in inputs)
        y = Tensor(
            y_data, requires_grad=requires_grad, creator=self if requires_grad else None
        )
        self.outputs = [y]
        return y

    def forward(self, *xs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(self, gy: np.ndarray) -> Any:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0: Any, x1: Any = None) -> Any:  # type: ignore[override]
        return x0 + x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx0 = _unbroadcast_grad(gy, self.inputs[0].shape)
        gx1 = _unbroadcast_grad(gy, self.inputs[1].shape)
        return gx0, gx1


class Sub(Function):
    def forward(self, x0: Any, x1: Any = None) -> Any:  # type: ignore[override]
        return x0 - x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx0 = _unbroadcast_grad(gy, self.inputs[0].shape)
        gx1 = _unbroadcast_grad(-gy, self.inputs[1].shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0: Any, x1: Any = None) -> Any:  # type: ignore[override]
        return x0 * x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = _unbroadcast_grad(gy * x1, self.inputs[0].shape)
        gx1 = _unbroadcast_grad(gy * x0, self.inputs[1].shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0: Any, x1: Any = None) -> Any:  # type: ignore[override]
        return x0 / x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = _unbroadcast_grad(gy / x1, self.inputs[0].shape)
        gx1 = _unbroadcast_grad(-gy * x0 / (x1**2 + 1e-12), self.inputs[1].shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


class Pow(Function):
    def __init__(self, power: float) -> None:
        super().__init__()
        self.power = power

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return x**self.power

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        return gy * self.power * (x ** (self.power - 1))


class MatMul(Function):
    def forward(self, x0: Any, x1: Any = None) -> Any:  # type: ignore[override]
        return np.matmul(x0, x1)

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        if x0.ndim == 2 and x1.ndim == 2:
            gx0 = np.dot(gy, x1.T)
            gx1 = np.dot(x0.T, gy)
        else:
            gx0 = np.matmul(gy, np.swapaxes(x1, -1, -2))
            gx1 = np.matmul(np.swapaxes(x0, -1, -2), gy)
        gx0 = _unbroadcast_grad(gx0, self.inputs[0].shape)
        gx1 = _unbroadcast_grad(gx1, self.inputs[1].shape)
        return gx0, gx1


class Sum(Function):
    def __init__(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x_shape = self.inputs[0].shape
        if not self.keepdims and self.axis is not None:
            axes: Tuple[int, ...]
            if isinstance(self.axis, int):
                axes = (self.axis,)
            else:
                axes = self.axis
            shape = list(x_shape)
            for a in axes:
                shape[a] = 1
            gy = gy.reshape(shape)
        return np.broadcast_to(gy, x_shape).copy()


class Reshape(Function):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return x.reshape(self.shape)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return gy.reshape(self.inputs[0].shape)


class Transpose(Function):
    def __init__(self, axes: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.axes = axes

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return np.transpose(x, self.axes)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return np.transpose(gy)
        inv_axes = np.argsort(self.axes)
        return np.transpose(gy, inv_axes)


class ReLU(Function):
    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        return np.maximum(0.0, x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        return gy * (x > 0).astype(np.float64)


class Sigmoid(Function):
    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
        return self.out

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return gy * self.out * (1.0 - self.out)


class Tanh(Function):
    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        self.out = np.tanh(x)
        return self.out

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return gy * (1.0 - self.out**2)


class GELU(Function):
    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        self.x = x
        self.tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        return 0.5 * x * (1.0 + self.tanh_term)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.x
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        sech2 = 1.0 - self.tanh_term**2
        grad = 0.5 * (1.0 + self.tanh_term) + 0.5 * x * sech2 * sqrt_2_pi * (
            1.0 + 3.0 * 0.044715 * x**2
        )
        return gy * grad
