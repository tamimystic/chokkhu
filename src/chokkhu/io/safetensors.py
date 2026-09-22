"""Pure NumPy HuggingFace SafeTensors Serializer & Deserializer.

Zero-dependency, byte-level SafeTensors parser supporting open-weights loading
for SmolLM, TinyLlama, GPT-2, LLaMA, Mistral, Gemma, and Vision Transformers.
Supports FP64, FP32, FP16, BF16 (bfloat16), INT64, INT32, INT16, INT8, UINT8, and BOOL.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor


_DTYPE_TO_STR: Dict[np.dtype, str] = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("int16"): "I16",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}

_STR_TO_DTYPE: Dict[str, np.dtype] = {
    "F64": np.dtype("float64"),
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "I16": np.dtype("int16"),
    "I8": np.dtype("int8"),
    "U8": np.dtype("uint8"),
    "BOOL": np.dtype("bool"),
}


def _encode_bf16(arr: np.ndarray) -> bytes:
    """Encode float array to bfloat16 bytes in pure NumPy."""
    arr_f32: np.ndarray = arr.astype(np.float32)
    u32: np.ndarray = arr_f32.view(np.uint32)
    u16: np.ndarray = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _decode_bf16(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Decode bfloat16 bytes to float64 NumPy array."""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape).astype(np.float64)


def save_safetensors(
    tensors: Dict[str, Union[np.ndarray, Tensor]],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
    bf16: bool = False,
) -> None:
    """Save a dictionary of tensors to a SafeTensors binary file.

    Args:
        tensors: Dictionary mapping tensor names to Tensors or NumPy arrays.
        filepath: Target destination file path.
        metadata: Optional string dictionary metadata.
        bf16: If True, saves floating point weights in bfloat16 format.
    """
    header: Dict[str, Any] = {}
    current_offset = 0
    raw_data_chunks: list[bytes] = []

    for name, t in tensors.items():
        arr = t.data if isinstance(t, Tensor) else np.asarray(t)
        arr = np.ascontiguousarray(arr)

        if bf16 and np.issubdtype(arr.dtype, np.floating):
            arr_bytes = _encode_bf16(arr)
            dtype_str = "BF16"
        else:
            arr_bytes = arr.tobytes()
            dtype_str = _DTYPE_TO_STR.get(arr.dtype, "F64")

        byte_len = len(arr_bytes)

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        current_offset += byte_len
        raw_data_chunks.append(arr_bytes)

    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    # Encode header as compact JSON
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    # Pack 8-byte uint64 header length
    header_len_bytes = struct.pack("<Q", header_len)

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_json)
        for chunk in raw_data_chunks:
            f.write(chunk)


def load_safetensors(filepath: Union[str, Path]) -> Dict[str, Tensor]:
    """Load tensors from a SafeTensors binary file into Chokkhu Tensors.

    Args:
        filepath: Path to the .safetensors file.
    Returns:
        Dictionary mapping tensor names to Chokkhu Tensors.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SafeTensors file not found: {path}")

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(
                f"Invalid SafeTensors file: too short ({len(header_len_bytes)} bytes)"
            )
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))
        data_buffer = f.read()

    tensors: Dict[str, Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start_off, end_off = info["data_offsets"]
        chunk_bytes = data_buffer[start_off:end_off]

        if dtype_str == "BF16":
            arr = _decode_bf16(chunk_bytes, shape)
        else:
            np_dtype = _STR_TO_DTYPE.get(dtype_str, np.dtype("float64"))
            arr = np.frombuffer(chunk_bytes, dtype=np_dtype).reshape(shape)
            if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float64:
                arr = arr.astype(np.float64)

        tensors[name] = Tensor(arr, requires_grad=False)

    return tensors


def save_quantized_safetensors(
    tensors: Dict[str, Union[np.ndarray, Tensor]],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Quantize floating point tensors to symmetric INT8 with per-tensor scale and save."""
    quantized_dict: Dict[str, Union[np.ndarray, Tensor]] = {}
    meta = dict(metadata or {})
    meta["quantization"] = "int8_symmetric"

    for name, t in tensors.items():
        arr = t.data if isinstance(t, Tensor) else np.asarray(t)
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(np.abs(arr)))
            scale = max(1e-12, max_val / 127.0)
            int8_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
            quantized_dict[name] = int8_arr
            quantized_dict[f"{name}.scale"] = np.array([scale], dtype=np.float32)
        else:
            quantized_dict[name] = arr

    save_safetensors(quantized_dict, filepath, metadata=meta)


def load_quantized_safetensors(filepath: Union[str, Path]) -> Dict[str, Tensor]:
    """Load INT8 quantized SafeTensors file and dequantize back to float64."""
    loaded = load_safetensors(filepath)
    dequantized: Dict[str, Tensor] = {}

    scale_keys = {k for k in loaded.keys() if k.endswith(".scale")}

    for name, tensor in loaded.items():
        if name.endswith(".scale"):
            continue
        scale_key = f"{name}.scale"
        if scale_key in scale_keys:
            scale_val = float(loaded[scale_key].data[0])
            deq_arr = tensor.data.astype(np.float64) * scale_val
            dequantized[name] = Tensor(deq_arr, requires_grad=False)
        else:
            dequantized[name] = tensor

    return dequantized
