r"""Hyperdimensional Computing and Vector Symbolic Architectures (HDC/VSA) in pure NumPy."""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


class HyperdimensionalVector:
    r"""Hyperdimensional Computing Vector (HDC/VSA) in pure NumPy (Kanerva 2009).

    Supports core Vector Symbolic Architecture primitive algebra:
    Binding ($\\otimes$), Bundling ($\\oplus$), and Permutation ($\\Pi$).

    Parameters
    ----------
    values : np.ndarray
        Raw hypervector values array of shape $(D,)$.
    """

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.dim = len(self.values)

    @classmethod
    def random_bipolar(
        cls, dim: int = 10000, seed: Optional[int] = None
    ) -> HyperdimensionalVector:
        """Samples a random dense bipolar hypervector with elements in {-1, +1}."""
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=dim) * 2 - 1
        return cls(bits.astype(np.float32))

    @classmethod
    def random_real(
        cls, dim: int = 10000, seed: Optional[int] = None
    ) -> HyperdimensionalVector:
        """Samples a normalized isotropic Gaussian real-valued hypervector."""
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(size=dim).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return cls(vec / norm)

    def bind(self, other: HyperdimensionalVector) -> HyperdimensionalVector:
        """Binding operation (\\otimes): Elementwise Hadamard product for associative pairing."""
        bound_values = self.values * other.values
        return HyperdimensionalVector(bound_values)

    def __mul__(self, other: HyperdimensionalVector) -> HyperdimensionalVector:
        return self.bind(other)

    @classmethod
    def bundle(
        cls, vectors: Sequence[HyperdimensionalVector], binarize: bool = True
    ) -> HyperdimensionalVector:
        """Bundling / Superposition (\\oplus): Elementwise sum + majority-rule sign binarization."""
        if len(vectors) == 0:
            raise ValueError("Cannot bundle an empty sequence of hypervectors.")

        stacked = np.stack([v.values for v in vectors], axis=0)
        summed = np.sum(stacked, axis=0)

        if binarize:
            # Majority rule sign binarization with tie-breaking
            bundled = np.where(summed >= 0, 1.0, -1.0).astype(np.float32)
            return cls(bundled)
        else:
            norm = np.linalg.norm(summed) + 1e-12
            return cls(summed / norm)

    def permute(self, shift: int = 1) -> HyperdimensionalVector:
        """Permutation (\\Pi^k): Circular shift representing temporal order and structural index."""
        shifted_values = np.roll(self.values, shift=shift)
        return HyperdimensionalVector(shifted_values)

    def similarity(self, other: HyperdimensionalVector) -> float:
        """Computes normalized cosine similarity in [-1, +1]."""
        dot = np.dot(self.values, other.values)
        norm_a = np.linalg.norm(self.values) + 1e-12
        norm_b = np.linalg.norm(other.values) + 1e-12
        return float(dot / (norm_a * norm_b))

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()
