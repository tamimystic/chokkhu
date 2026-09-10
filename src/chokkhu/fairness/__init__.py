"""Algorithmic Fairness, Bias Mitigation & Ethical AI Package.

Pure NumPy implementations of:
- Fairness metrics: demographic parity, equalized odds, equal opportunity, disparate impact, Theil index
- Pre-processing mitigation: ReweighingTransformer, DisparateImpactRemover
- Post-processing mitigation: ThresholdOptimizer
"""

from .metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    disparate_impact_ratio,
    theil_index,
    fairness_report,
)
from .pre_processing import ReweighingTransformer, DisparateImpactRemover
from .post_processing import ThresholdOptimizer

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
