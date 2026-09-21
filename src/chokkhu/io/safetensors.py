"""Pure NumPy HuggingFace SafeTensors Serializer & Deserializer.

Zero-dependency, byte-level SafeTensors parser supporting open-weights loading
for SmolLM, TinyLlama, GPT-2, LLaMA, Mistral, and Vision Transformers.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Union
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


def save_safetensors(
    tensors: Dict[str, Union[np.ndarray, Tensor]],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save a dictionary of tensors to a SafeTensors binary file.

    Args:
        tensors: Dictionary mapping tensor names to Tensors or NumPy arrays.
        filepath: Target destination file path.
        metadata: Optional string dictionary metadata.
    """
    header: Dict[str, Any] = {}
    current_offset = 0
    raw_data_chunks: list[bytes] = []

    for name, t in tensors.items():
        arr = t.data if isinstance(t, Tensor) else np.asarray(t)
        # Ensure array is contiguous in memory
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        byte_len = len(arr_bytes)

        dtype_str = _DTYPE_TO_STR.get(arr.dtype, "F64")
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
        # 1. Read 8-byte uint64 header length
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(
                f"Invalid SafeTensors file: too short ({len(header_len_bytes)} bytes)"
            )
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        # 2. Read JSON header
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))

        # 3. Read raw data buffer
        data_buffer = f.read()

    tensors: Dict[str, Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start_off, end_off = info["data_offsets"]

        np_dtype = _STR_TO_DTYPE.get(dtype_str, np.dtype("float64"))
        chunk_bytes = data_buffer[start_off:end_off]
        arr = np.frombuffer(chunk_bytes, dtype=np_dtype).reshape(shape)

        # Convert to float64 Tensor for Chokkhu execution if float
        if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float64:
            arr = arr.astype(np.float64)

        tensors[name] = Tensor(arr, requires_grad=False)

    return tensors
