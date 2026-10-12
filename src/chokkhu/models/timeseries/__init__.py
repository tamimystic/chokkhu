"""Time Series & Forecasting Subsystem from First Principles."""

from __future__ import annotations

from .conformal import (
    ConformalPredictor,
    conformal_interval,
)
from .kalman import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from .matrix_profile import (
    MatrixProfile,
    find_discords,
    find_motifs,
)
from .neural import (
    NBEATS,
    NHITS,
    NBEATSBlock,
    NHITSBlock,
    PatchTST,
)
from .particle_filter import (
    ParticleFilter,
    RaoBlackwellizedParticleFilter,
)
from .statistical import (
    ARIMA,
    SARIMAX,
    AutoARIMA,
    ExponentialSmoothing,
)
from .transforms import (
    create_lag_matrix,
    decompose_series,
    difference,
    inverse_difference,
)

__all__ = [
    "decompose_series",
    "create_lag_matrix",
    "difference",
    "inverse_difference",
    "ARIMA",
    "AutoARIMA",
    "SARIMAX",
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
    "ParticleFilter",
    "RaoBlackwellizedParticleFilter",
]
