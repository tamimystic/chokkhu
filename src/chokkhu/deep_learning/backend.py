
try:
    import cupy as cp

    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    import numpy as np

    GPU_AVAILABLE = False
    xp = np


def get_array_module(x):
    if GPU_AVAILABLE and isinstance(x, cp.ndarray):
        return cp
    return np
