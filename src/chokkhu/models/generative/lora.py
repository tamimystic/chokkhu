"""Sovereign Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning in Pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class LoRALinear:
    """Low-Rank Adaptation (LoRA) Layer wrapping a base linear transformation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        random_state: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(max(r, 1))
        self.lora_dropout = lora_dropout
        self.use_bias = bias
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Base frozen weights W0
        self.weight: np.ndarray = (
            self.rng.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        ).astype(np.float32)
        self.bias: Optional[np.ndarray] = (
            np.zeros((1, out_features), dtype=np.float32) if bias else None
        )

        # Trainable low-rank adaptation matrices: A ~ N(0, 1/r), B = 0
        if r > 0:
            self.lora_A: np.ndarray = (
                self.rng.randn(in_features, r) * (1.0 / np.sqrt(r))
            ).astype(np.float32)
            self.lora_B: np.ndarray = np.zeros((r, out_features), dtype=np.float32)
        else:
            self.lora_A = np.zeros((in_features, 0), dtype=np.float32)
            self.lora_B = np.zeros((0, out_features), dtype=np.float32)

        self.merged = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward evaluation: h = x * W0 + (alpha / r) * (x * A) * B + bias."""
        x = np.asarray(x, dtype=np.float32)
        out = np.dot(x, self.weight)

        if not self.merged and self.r > 0:
            h_adapter = x
            if training and self.lora_dropout > 0.0:
                mask = (self.rng.rand(*h_adapter.shape) > self.lora_dropout).astype(
                    np.float32
                )
                h_adapter = (h_adapter * mask) / (1.0 - self.lora_dropout)

            lora_term = (
                np.dot(np.dot(h_adapter, self.lora_A), self.lora_B) * self.scaling
            )
            out = out + lora_term

        if self.bias is not None:
            out = out + self.bias

        return out

    def merge(self) -> None:
        """Merge LoRA weights into base weights: W = W0 + (alpha / r) * A * B."""
        if not self.merged and self.r > 0:
            delta_w = np.dot(self.lora_A, self.lora_B) * self.scaling
            self.weight += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from base weights: W0 = W - (alpha / r) * A * B."""
        if self.merged and self.r > 0:
            delta_w = np.dot(self.lora_A, self.lora_B) * self.scaling
            self.weight -= delta_w
            self.merged = False

    def get_trainable_params(self) -> List[np.ndarray]:
        """Return references to trainable adapter matrices A and B."""
        if self.r > 0:
            return [self.lora_A, self.lora_B]
        return []


class LoRAAdapter:
    """Manager utility to attach, merge, unmerge, and track LoRA layers."""

    @staticmethod
    def attach_lora_to_linear(
        linear_layer: Any,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        random_state: int = 42,
    ) -> LoRALinear:
        """Convert a standard Linear layer into a LoRALinear layer retaining existing weights."""
        in_f = getattr(linear_layer, "in_features", None)
        out_f = getattr(linear_layer, "out_features", None)

        if in_f is None or out_f is None:
            w = getattr(linear_layer, "weight", None)
            if w is not None:
                in_f, out_f = w.shape
            else:
                raise ValueError(
                    "Could not deduce dimensions from provided linear layer."
                )

        has_bias = getattr(linear_layer, "bias", None) is not None
        lora_layer = LoRALinear(
            in_features=in_f,
            out_features=out_f,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=has_bias,
            random_state=random_state,
        )

        if hasattr(linear_layer, "weight") and linear_layer.weight is not None:
            lora_layer.weight = np.copy(linear_layer.weight)
        if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
            lora_layer.bias = np.copy(linear_layer.bias)

        return lora_layer

    @staticmethod
    def count_parameters(layers: List[LoRALinear]) -> Dict[str, Any]:
        """Count total, frozen, and trainable LoRA parameters."""
        total_params = 0
        trainable_params = 0

        for layer in layers:
            base_count = layer.weight.size + (
                layer.bias.size if layer.bias is not None else 0
            )
            adapter_count = layer.lora_A.size + layer.lora_B.size
            total_params += base_count + adapter_count
            trainable_params += adapter_count

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_percent": 100.0
            * float(trainable_params)
            / float(max(total_params, 1)),
        }
