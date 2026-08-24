
from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
from .backend import xp, get_array_module

class Tensor:
    def __init__(self, data, requires_grad: bool = False, _children: Tuple = (), _op: str = ''):
        # Convert to numpy/cupy array
        self.data = xp.array(data, dtype=xp.float32)
        self.grad = xp.zeros_like(self.data)
        self.requires_grad = requires_grad
        
        # Autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
        if not self.requires_grad:
            return
            
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize the gradient of the root to 1.0
        self.grad = xp.ones_like(self.data)
        
        # Backpropagate
        for v in reversed(topo):
            if v.requires_grad:
                v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # --- Math Operations with Autograd ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(xp.ones_like(self.data) * other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self_grad = out.grad
                # Handle broadcasting
                while len(self_grad.shape) > len(self.data.shape):
                    self_grad = self_grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        self_grad = self_grad.sum(axis=i, keepdims=True)
                self.grad += self_grad
                
            if other.requires_grad:
                other_grad = out.grad
                while len(other_grad.shape) > len(other.data.shape):
                    other_grad = other_grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        other_grad = other_grad.sum(axis=i, keepdims=True)
                other.grad += other_grad
                
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(xp.ones_like(self.data) * other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self_grad = out.grad * other.data
                while len(self_grad.shape) > len(self.data.shape):
                    self_grad = self_grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        self_grad = self_grad.sum(axis=i, keepdims=True)
                self.grad += self_grad
                
            if other.requires_grad:
                other_grad = out.grad * self.data
                while len(other_grad.shape) > len(other.data.shape):
                    other_grad = other_grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        other_grad = other_grad.sum(axis=i, keepdims=True)
                other.grad += other_grad
                
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, requires_grad=self.requires_grad, _children=(self,), _op=f'**{other}')
        
        def _backward():
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(xp.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')
        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
        
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='sum')
        def _backward():
            if self.requires_grad:
                self.grad += xp.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # Reverse and sub overrides
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)
    def __neg__(self): return self * -1
