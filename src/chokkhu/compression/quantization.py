"""Quantization and Low-Bit Precision Modeling.

Pure NumPy implementations of:
- UniformQuantizer: INT8/INT4 symmetric and asymmetric quantization/dequantization
- PostTrainingQuantizer: MinMax, Percentile, and KL-Divergence calibration
- QuantizedLinear: INT8 weight linear layer with FP32/INT32 accumulation
"""

from typing import Optional
import numpy as np


class UniformQuantizer:
    r"""Uniform Symmetric / Asymmetric Quantizer.

    Quantizes floating-point tensors into fixed integer bit representations:

    .. math::
        q = \text{clip}\left( \left\lfloor \frac{x}{S} \right\rceil + Z, q_{\min}, q_{\max} \right)
        \hat{x} = S (q - Z)

    Parameters
    ----------
    bits : int, default=8
        Bit-width of target integer representation (e.g. 8 for INT8, 4 for INT4).
    symmetric : bool, default=True
        Whether to use zero-centered symmetric quantization (zero-point = 0).
    narrow_range : bool, default=False
        Whether to restrict range to :math:`[-q_{\max}, q_{\max}]` (e.g. [-127, 127] for INT8).
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        narrow_range: bool = False,
    ) -> None:
        if bits < 2 or bits > 32:
            raise ValueError("bits must be between 2 and 32.")
        self.bits = bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range

        if symmetric:
            if narrow_range:
                self.q_min = -(2 ** (bits - 1) - 1)
                self.q_max = 2 ** (bits - 1) - 1
            else:
                self.q_min = -(2 ** (bits - 1))
                self.q_max = 2 ** (bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2**bits - 1

        self.scale: float = 1.0
        self.zero_point: int = 0
        self.is_calibrated: bool = False

    def calibrate(self, x: np.ndarray) -> "UniformQuantizer":
        """Compute scale S and zero-point Z from input data."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("Cannot calibrate empty array.")

        if self.symmetric:
            max_val = float(np.max(np.abs(arr)))
            max_val = max(max_val, 1e-8)
            self.scale = max_val / float(self.q_max)
            self.zero_point = 0
        else:
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            diff = max(max_val - min_val, 1e-8)
            self.scale = diff / float(self.q_max - self.q_min)
            zp_float = -min_val / self.scale + self.q_min
            self.zero_point = int(np.clip(np.round(zp_float), self.q_min, self.q_max))

        self.is_calibrated = True
        return self

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize floating-point array to integer array."""
        if not self.is_calibrated:
            self.calibrate(x)

        arr = np.asarray(x, dtype=np.float32)
        scaled = arr / self.scale + self.zero_point
        rounded = np.round(scaled)
        clipped = np.clip(rounded, self.q_min, self.q_max)

        if self.bits <= 8:
            dtype = np.int8 if self.symmetric else np.uint8
        elif self.bits <= 16:
            dtype = np.int16 if self.symmetric else np.uint16
        else:
            dtype = np.int32 if self.symmetric else np.uint32

        return clipped.astype(dtype)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        """Dequantize integer array back to floating-point representation."""
        if not self.is_calibrated:
            raise RuntimeError("Quantizer must be calibrated before dequantization.")

        q_float = np.asarray(q, dtype=np.float32)
        return (q_float - self.zero_point) * self.scale

    def fake_quantize(self, x: np.ndarray) -> np.ndarray:
        """Simulate quantization error by quantizing and immediately dequantizing."""
        q = self.quantize(x)
        return self.dequantize(q)


class PostTrainingQuantizer:
    """Post-Training Quantization (PTQ) Calibration Engine.

    Supports MinMax, Percentile (outlier-robust), and KL-divergence calibration.
    """

    def __init__(
        self,
        method: str = "minmax",
        bits: int = 8,
        percentile: float = 99.99,
    ) -> None:
        if method not in ("minmax", "percentile", "kl_divergence"):
            raise ValueError(f"Unknown PTQ method: '{method}'")
        self.method = method
        self.bits = bits
        self.percentile = percentile
        self.quantizer = UniformQuantizer(bits=bits, symmetric=True)

    def calibrate(self, data: np.ndarray) -> UniformQuantizer:
        """Calibrate quantization scale from sample calibration dataset."""
        arr = np.asarray(data, dtype=np.float32)
        if self.method == "minmax":
            self.quantizer.calibrate(arr)
        elif self.method == "percentile":
            p = float(np.percentile(np.abs(arr), self.percentile))
            p = max(p, 1e-8)
            self.quantizer.scale = p / float(self.quantizer.q_max)
            self.quantizer.zero_point = 0
            self.quantizer.is_calibrated = True
        elif self.method == "kl_divergence":
            # Select threshold minimizing KL divergence between FP32 histogram and quantized histogram
            abs_arr = np.abs(arr.ravel())
            hist, bin_edges = np.histogram(abs_arr, bins=128)
            hist = hist.astype(np.float32) + 1e-5
            p_dist = hist / np.sum(hist)

            best_kl = float("inf")
            best_thresh = float(np.max(abs_arr))

            for i in range(32, 128, 8):
                thresh = float(bin_edges[i])
                clipped_data = np.clip(abs_arr, 0.0, thresh)
                q_hist, _ = np.histogram(clipped_data, bins=128)
                q_hist = q_hist.astype(np.float32) + 1e-5
                q_dist = q_hist / np.sum(q_hist)

                # KL(P || Q)
                kl = float(np.sum(p_dist * np.log(p_dist / q_dist)))
                if kl < best_kl:
                    best_kl = kl
                    best_thresh = thresh

            self.quantizer.scale = max(best_thresh, 1e-8) / float(self.quantizer.q_max)
            self.quantizer.zero_point = 0
            self.quantizer.is_calibrated = True

        return self.quantizer


class QuantizedLinear:
    """Linear Layer with INT8 Quantized Weights and FP32 Activation.

    Parameters
    ----------
    in_features : int
        Input dimensionality.
    out_features : int
        Output dimensionality.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quantizer = UniformQuantizer(bits=8, symmetric=True)
        self.q_weight: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def quantize_from_float(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> "QuantizedLinear":
        """Quantize floating point weights and store in INT8 format."""
        w = np.asarray(weight, dtype=np.float32)
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight shape {w.shape} does not match ({self.out_features}, {self.in_features})"
            )

        self.q_weight = self.weight_quantizer.quantize(w)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=np.float32).copy()
        else:
            self.bias = None
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute linear forward pass: x @ W_dequant.T + b."""
        if self.q_weight is None:
            raise RuntimeError("QuantizedLinear must be initialized with weights.")

        x_arr = np.asarray(x, dtype=np.float32)
        w_dequant = self.weight_quantizer.dequantize(self.q_weight)

        out = x_arr @ w_dequant.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
