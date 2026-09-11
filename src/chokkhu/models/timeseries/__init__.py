"""Time Series & Forecasting Subsystem from First Principles."""

from __future__ import annotations
from .kalman import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from .transforms import (
    decompose_series,
    create_lag_matrix,
    difference,
    inverse_difference,
)
from .statistical import (
    ARIMA,
    ExponentialSmoothing,
)
from .neural import (
    NBEATS,
    NBEATSBlock,
    NHITS,
    NHITSBlock,
    PatchTST,
)
from .conformal import (
    ConformalPredictor,
    conformal_interval,
)
from .matrix_profile import (
    MatrixProfile,
    find_motifs,
    find_discords,
)

__all__ = [
    "decompose_series",
    "create_lag_matrix",
    "difference",
    "inverse_difference",
    "ARIMA",
    "ExponentialSmoothing",
    "NBEATS",
    "NBEATSBlock",
    "NHITS",
    "NHITSBlock",
    "PatchTST",
    "ConformalPredictor",
    "conformal_interval",
    "MatrixProfile",
    "find_motifs",
    "find_discords",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
]
