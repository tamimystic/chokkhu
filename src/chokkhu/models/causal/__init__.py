"""Chokkhu Causal Inference, Uplift Modeling & Treatment Effects Subsystem.

Sovereign pure NumPy implementations of:
- Potential Outcomes & Propensity: PropensityModel, PropensityScoreMatching, InverseProbabilityWeighting
- Doubly Robust: DoublyRobustLearner (ATE & CATE)
- Meta-Learners: SLearner, TLearner, XLearner
- Uplift Modeling: TwoModelUplift, ClassTransformationUplift
- Uplift Evaluation: qini_curve, qini_score, cumulative_gain_curve, uplift_at_k
- Instrumental Variables: TwoStageLeastSquares (2SLS), InstrumentalGMM
- Double ML & Heterogeneous Effects: DoubleMLPLR, RLearner
"""

from chokkhu.models.causal.potential_outcomes import (
    InverseProbabilityWeighting,
    PropensityModel,
    PropensityScoreMatching,
)
from chokkhu.models.causal.doubly_robust import DoublyRobustLearner
from chokkhu.models.causal.meta_learners import SLearner, TLearner, XLearner
from chokkhu.models.causal.discovery import (
    NOTEARSCausalDiscovery,
    PCAlgorithm,
)
from chokkhu.models.causal.uplift import (
    ClassTransformationUplift,
    TwoModelUplift,
    cumulative_gain_curve,
    qini_curve,
    qini_score,
    uplift_at_k,
)
from chokkhu.models.causal.instrumental import (
    TwoStageLeastSquares,
    InstrumentalGMM,
)
from chokkhu.models.causal.double_ml import (
    DoubleMLPLR,
    RLearner,
)

__all__ = [
    "PropensityModel",
    "PropensityScoreMatching",
    "InverseProbabilityWeighting",
    "DoublyRobustLearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "TwoModelUplift",
    "ClassTransformationUplift",
    "qini_curve",
    "qini_score",
    "cumulative_gain_curve",
    "uplift_at_k",
    "NOTEARSCausalDiscovery",
    "PCAlgorithm",
    "TwoStageLeastSquares",
    "InstrumentalGMM",
    "DoubleMLPLR",
    "RLearner",
]
