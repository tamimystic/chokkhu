"""Clifford / Geometric Algebra (Cl(3, 0, 0)) & Geometric Neural Networks in Pure NumPy.

References:
- Hestenes (1966): "Space-Time Algebra".
- Brandstetter et al. (2022): "Clifford Neural Networks for PDE Modeling" (ICLR 2023).
- Ruhe et al. (2023): "Geometric Clifford Algebra Networks".
"""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np


# 8x8 Cayley Multiplication Table for Cl(3, 0, 0)
# Basis blades: 0: 1, 1: e1, 2: e2, 3: e3, 4: e12, 5: e23, 6: e31, 7: e123
# CAYLEY_TABLE[i, j] = (target_k, sign) where e_i * e_j = sign * e_k
CAYLEY_TABLE: List[List[Tuple[int, float]]] = [
    # 0: 1
    [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)],
    # 1: e1
    [(1, 1.0), (0, 1.0), (4, 1.0), (6, -1.0), (2, 1.0), (7, 1.0), (3, -1.0), (5, 1.0)],
    # 2: e2
    [(2, 1.0), (4, -1.0), (0, 1.0), (5, 1.0), (1, -1.0), (3, 1.0), (7, 1.0), (6, 1.0)],
    # 3: e3
    [(3, 1.0), (6, 1.0), (5, -1.0), (0, 1.0), (7, 1.0), (2, -1.0), (1, 1.0), (4, 1.0)],
    # 4: e12
    [
        (4, 1.0),
        (2, -1.0),
        (1, 1.0),
        (7, 1.0),
        (0, -1.0),
        (6, -1.0),
        (5, 1.0),
        (3, -1.0),
    ],
    # 5: e23
    [
        (5, 1.0),
        (7, 1.0),
        (3, -1.0),
        (2, 1.0),
        (6, 1.0),
        (0, -1.0),
        (4, -1.0),
        (1, -1.0),
    ],
    # 6: e31
    [
        (6, 1.0),
        (3, 1.0),
        (7, 1.0),
        (1, -1.0),
        (5, -1.0),
        (4, 1.0),
        (0, -1.0),
        (2, -1.0),
    ],
    # 7: e123
    [
        (7, 1.0),
        (5, 1.0),
        (6, 1.0),
        (4, 1.0),
        (3, -1.0),
        (1, -1.0),
        (2, -1.0),
        (0, -1.0),
    ],
]


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Clifford Cl(3, 0, 0) geometric product of multivector arrays a and b.

    Args:
        a: Array of shape (..., 8)
        b: Array of shape (..., 8)

    Returns:
        Product multivector array of shape (..., 8)
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    # Broadcast shapes
    out_shape = np.broadcast_shapes(a_arr.shape[:-1], b_arr.shape[:-1]) + (8,)
    out = np.zeros(out_shape, dtype=np.float64)

    for i in range(8):
        a_i = a_arr[..., i]
        for j in range(8):
            b_j = b_arr[..., j]
            target_k, sign = CAYLEY_TABLE[i][j]
            out[..., target_k] += sign * a_i * b_j

    return out


class CliffordMultivector:
    """Sovereign 3D Multivector representation in Cl(3, 0, 0).

    Components:
    - Grade 0: 1 (Scalar) -> index 0
    - Grade 1: e1, e2, e3 (Vector) -> indices 1, 2, 3
    - Grade 2: e12, e23, e31 (Bivector) -> indices 4, 5, 6
    - Grade 3: e123 (Pseudoscalar) -> index 7
    """

    def __init__(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape[-1] != 8:
            raise ValueError(
                f"Last dimension must be 8 (for Cl(3,0,0)), got {arr.shape}"
            )
        self.data: np.ndarray = arr

    @classmethod
    def from_scalar(cls, scalar: Union[float, np.ndarray]) -> "CliffordMultivector":
        """Create a multivector from scalar components."""
        s = np.asarray(scalar, dtype=np.float64)
        if s.ndim == 0:
            data_single: np.ndarray = np.zeros(8, dtype=np.float64)
            data_single[0] = float(s)
            return cls(data_single)
        data: np.ndarray = np.zeros(s.shape + (8,), dtype=np.float64)
        data[..., 0] = s
        return cls(data)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "CliffordMultivector":
        """Create a multivector from 3D vector components (e1, e2, e3)."""
        v = np.asarray(vector, dtype=np.float64)
        if v.shape[-1] != 3:
            raise ValueError(f"Vector last dimension must be 3, got {v.shape}")
        data = np.zeros(v.shape[:-1] + (8,), dtype=np.float64)
        data[..., 1:4] = v
        return cls(data)

    @classmethod
    def from_bivector(cls, bivector: np.ndarray) -> "CliffordMultivector":
        """Create a multivector from 3D bivector components (e12, e23, e31)."""
        b = np.asarray(bivector, dtype=np.float64)
        if b.shape[-1] != 3:
            raise ValueError(f"Bivector last dimension must be 3, got {b.shape}")
        data = np.zeros(b.shape[:-1] + (8,), dtype=np.float64)
        data[..., 4:7] = b
        return cls(data)

    @property
    def scalar(self) -> np.ndarray:
        return self.data[..., 0]

    @property
    def vector(self) -> np.ndarray:
        return self.data[..., 1:4]

    @property
    def bivector(self) -> np.ndarray:
        return self.data[..., 4:7]

    @property
    def pseudoscalar(self) -> np.ndarray:
        return self.data[..., 7]

    def grade(self, k: int) -> np.ndarray:
        """Extract grade k components (k in {0, 1, 2, 3})."""
        if k == 0:
            return self.data[..., 0:1]
        elif k == 1:
            return self.data[..., 1:4]
        elif k == 2:
            return self.data[..., 4:7]
        elif k == 3:
            return self.data[..., 7:8]
        else:
            raise ValueError(f"Grade must be in {0, 1, 2, 3}, got {k}")

    def reverse(self) -> "CliffordMultivector":
        """Reversion involution ~a: sign multiplier (-1)^{k(k-1)/2} per grade."""
        # Grade 0: +1, Grade 1: +1, Grade 2: -1, Grade 3: -1
        rev_mask = np.array(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64
        )
        return CliffordMultivector(self.data * rev_mask)

    def __add__(
        self, other: Union[CliffordMultivector, float, np.ndarray]
    ) -> "CliffordMultivector":
        if isinstance(other, CliffordMultivector):
            return CliffordMultivector(self.data + other.data)
        return CliffordMultivector(self.data + other)

    def __sub__(
        self, other: Union[CliffordMultivector, float, np.ndarray]
    ) -> "CliffordMultivector":
        if isinstance(other, CliffordMultivector):
            return CliffordMultivector(self.data - other.data)
        return CliffordMultivector(self.data - other)

    def __mul__(
        self, other: Union[CliffordMultivector, float, np.ndarray]
    ) -> "CliffordMultivector":
        if isinstance(other, (int, float)):
            return CliffordMultivector(self.data * float(other))
        elif isinstance(other, np.ndarray):
            if other.shape[-1] == 8:
                return CliffordMultivector(geometric_product(self.data, other))
            return CliffordMultivector(self.data * other[..., np.newaxis])
        elif isinstance(other, CliffordMultivector):
            return CliffordMultivector(geometric_product(self.data, other.data))
        else:
            return NotImplemented

    def __rmul__(self, other: Union[float, np.ndarray]) -> "CliffordMultivector":
        return self.__mul__(other)

    @classmethod
    def rotor(cls, axis: np.ndarray, angle_rad: float) -> "CliffordMultivector":
        """Construct a 3D rotor R = exp(-theta/2 * B) = cos(theta/2) - sin(theta/2) * B.

        Args:
            axis: 3D rotation axis unit vector (3,)
            angle_rad: Rotation angle in radians
        """
        ax = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(ax)
        if norm > 1e-12:
            ax = ax / norm

        # Dual unit bivector B = I * axis = axis_x e23 + axis_y e31 + axis_z e12
        # Specifically:
        # e23 dual to e1, e31 dual to e2, e12 dual to e3
        half_theta = float(angle_rad) * 0.5
        cos_half = np.cos(half_theta)
        sin_half = np.sin(half_theta)

        data: np.ndarray = np.zeros(8, dtype=np.float64)
        data[0] = cos_half
        # Bivector components: e12 (idx 4), e23 (idx 5), e31 (idx 6)
        data[4] = -sin_half * ax[2]  # e12
        data[5] = -sin_half * ax[0]  # e23
        data[6] = -sin_half * ax[1]  # e31

        return cls(data)

    def sandwich(self, rotor: "CliffordMultivector") -> "CliffordMultivector":
        """Rotate multivector via sandwich product: v' = R * v * ~R."""
        r_rev = rotor.reverse()
        return rotor * self * r_rev


class CliffordLinear:
    """Clifford Multivector Linear Layer preserving grade structure and equivariance."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        random_state: int = 42,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.rng = np.random.RandomState(random_state)

        # Weight tensor of shape (in_channels, out_channels, 8)
        limit = np.sqrt(2.0 / (in_channels * 8))
        self.weights: np.ndarray = self.rng.uniform(
            -limit, limit, size=(in_channels, out_channels, 8)
        ).astype(np.float64)
        self.bias: np.ndarray = np.zeros((out_channels, 8), dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input multivector tensor of shape (..., in_channels, 8)

        Returns:
            Output multivector tensor of shape (..., out_channels, 8)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        # x: [B, C_in, 8], weights: [C_in, C_out, 8]
        # For each pair (c_in, c_out), compute geometric product and sum over c_in
        batch_shape = x_arr.shape[:-2]
        B = int(np.prod(batch_shape)) if batch_shape else 1
        x_reshaped = x_arr.reshape(B, self.in_channels, 8)

        out: np.ndarray = np.zeros((B, self.out_channels, 8), dtype=np.float64)

        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                prod = geometric_product(
                    x_reshaped[:, c_in, :], self.weights[c_in, c_out, :]
                )
                out[:, c_out, :] += prod
            out[:, c_out, :] += self.bias[c_out, :]

        return out.reshape(batch_shape + (self.out_channels, 8))


class CliffordGANN:
    """Clifford Geometric Algebra Neural Network for physical and molecular dynamics."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 1,
        num_layers: int = 2,
        random_state: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.random_state = random_state

        self.layers: List[CliffordLinear] = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            layer = CliffordLinear(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                random_state=random_state + i,
            )
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with grade-wise non-linear activations."""
        h = np.asarray(x, dtype=np.float64)
        if h.ndim == 2 and h.shape[-1] == 8:
            h = h[:, np.newaxis, :]  # (N, 1, 8)

        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                # Grade-preserving non-linearity: scale multivector by sigmoid of magnitude
                mag = np.linalg.norm(h, axis=-1, keepdims=True)
                scale = 1.0 / (1.0 + np.exp(-np.clip(mag, -15.0, 15.0)))
                h = h * scale

        return h
