"""Edge AI, Quantization & Model Compression Package.

Pure NumPy implementations of:
- UniformQuantizer, PostTrainingQuantizer, QuantizedLinear
- MagnitudePruner, GlobalMagnitudePruner
- KnowledgeDistiller, FeatureDistiller
"""

from .distillation import (
    FeatureDistiller,
    KnowledgeDistiller,
)
from .pruning import (
    GlobalMagnitudePruner,
    MagnitudePruner,
)
from .quantization import (
    PostTrainingQuantizer,
    QuantizedLinear,
    UniformQuantizer,
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
