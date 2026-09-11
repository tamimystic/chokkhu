"""Zero-Dependency Automated Machine Learning (AutoML) Subsystem."""

from __future__ import annotations
from .surrogate import (
    GaussianProcessSurrogate,
    expected_improvement,
    upper_confidence_bound,
)
from .bayesian import BayesianOptimization
from .hyperband import Hyperband
from .bohb import BOHB
from .stacking import (
    SuperLearner,
    StackingEnsemble,
)
from .nas import DARTS
from .genetic_pipeline import (
    GeneticPipelineSearch,
    PipelineIndividual,
)
from .dynamic_selection import DynamicEnsembleSelection
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
    "BOHB",
    "DARTS",
    "GeneticPipelineSearch",
    "PipelineIndividual",
    "DynamicEnsembleSelection",
    "SuperLearner",
    "StackingEnsemble",
    "AutoTrainer",
    "AutoMLResult",
    "auto_train",
]
