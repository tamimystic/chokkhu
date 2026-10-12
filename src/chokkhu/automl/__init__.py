"""Zero-Dependency Automated Machine Learning (AutoML) Subsystem."""

from __future__ import annotations

from .autotrainer import (
    AutoMLResult,
    AutoTrainer,
    auto_train,
    automl,
)
from .bayesian import BayesianOptimization
from .bohb import BOHB
from .dynamic_selection import DynamicEnsembleSelection
from .genetic_pipeline import (
    GeneticPipelineSearch,
    PipelineIndividual,
)
from .hyperband import Hyperband
from .nas import DARTS
from .stacking import (
    StackingEnsemble,
    SuperLearner,
)
from .surrogate import (
    GaussianProcessSurrogate,
    expected_improvement,
    upper_confidence_bound,
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
    "automl",
]
