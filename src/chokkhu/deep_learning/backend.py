from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    cp = None  # type: ignore
    GPU_AVAILABLE = False
    xp = np


def get_array_module(x):
    if GPU_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np
