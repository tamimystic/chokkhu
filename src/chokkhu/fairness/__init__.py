"""Algorithmic Fairness, Bias Mitigation & Ethical AI Package.

Pure NumPy implementations of:
- Fairness metrics: demographic parity, equalized odds, equal opportunity, disparate impact, Theil index
- Pre-processing mitigation: ReweighingTransformer, DisparateImpactRemover
- Post-processing mitigation: ThresholdOptimizer
"""

from .metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    fairness_report,
    theil_index,
)
from .post_processing import ThresholdOptimizer
from .pre_processing import DisparateImpactRemover, ReweighingTransformer

__all__ = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "disparate_impact_ratio",
    "theil_index",
    "fairness_report",
    "ReweighingTransformer",
    "DisparateImpactRemover",
    "ThresholdOptimizer",
]
