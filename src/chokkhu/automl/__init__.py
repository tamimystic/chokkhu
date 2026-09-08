"""Zero-Dependency Automated Machine Learning (AutoML) Subsystem."""

from __future__ import annotations
from .surrogate import (
    GaussianProcessSurrogate,
    expected_improvement,
    upper_confidence_bound,
)
from .bayesian import BayesianOptimization
from .hyperband import Hyperband
from .autotrainer import (
    AutoTrainer,
    AutoMLResult,
    auto_train,
)

__all__ = [
    "GaussianProcessSurrogate",
    "expected_improvement",
    "upper_confidence_bound",
    "BayesianOptimization",
    "Hyperband",
    "AutoTrainer",
    "AutoMLResult",
    "auto_train",
]
