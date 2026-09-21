from __future__ import annotations

import sys
import types
from .dispatcher import MultiTaskDispatcher, MultiTaskPipelineResult, dispatch_pipeline
from .engine import ChokkhuPipeline, PipelineResult, TransformationState, pipeline
from .feature_synthesis import DeepFeatureSynthesizer
from .stacking import StackingPipeline, SuperLearner


class _PipelineModule(types.ModuleType):
    """Callable module wrapper allowing chokkhu.pipeline(...) function invocation."""

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return pipeline(*args, **kwargs)


sys.modules[__name__].__class__ = _PipelineModule

__all__ = [
    "pipeline",
    "ChokkhuPipeline",
    "PipelineResult",
    "TransformationState",
    "DeepFeatureSynthesizer",
    "SuperLearner",
    "StackingPipeline",
    "MultiTaskDispatcher",
    "dispatch_pipeline",
    "MultiTaskPipelineResult",
]
