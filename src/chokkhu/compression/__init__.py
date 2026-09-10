"""Edge AI, Quantization & Model Compression Package.

Pure NumPy implementations of:
- UniformQuantizer, PostTrainingQuantizer, QuantizedLinear
- MagnitudePruner, GlobalMagnitudePruner
- KnowledgeDistiller, FeatureDistiller
"""

from .quantization import (
    UniformQuantizer,
    PostTrainingQuantizer,
    QuantizedLinear,
)
from .pruning import (
    MagnitudePruner,
    GlobalMagnitudePruner,
)
from .distillation import (
    KnowledgeDistiller,
    FeatureDistiller,
)

__all__ = [
    "UniformQuantizer",
    "PostTrainingQuantizer",
    "QuantizedLinear",
    "MagnitudePruner",
    "GlobalMagnitudePruner",
    "KnowledgeDistiller",
    "FeatureDistiller",
]
