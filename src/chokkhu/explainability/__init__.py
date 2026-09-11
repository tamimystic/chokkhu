from __future__ import annotations

from .counterfactuals import WachterCounterfactualExplainer
from .engine import ExplanationResult, explain
from .importance import permutation_feature_importance
from .mechanistic import AttentionRollout, DirectLogitAttribution
from .neural import DeepLIFT, IntegratedGradients, SmoothGrad
from .pdp import partial_dependence
from .shap import kernel_shap
from .tcav import TCAV

__all__ = [
    "explain",
    "ExplanationResult",
    "permutation_feature_importance",
    "kernel_shap",
    "partial_dependence",
    "IntegratedGradients",
    "SmoothGrad",
    "DeepLIFT",
    "AttentionRollout",
    "DirectLogitAttribution",
    "WachterCounterfactualExplainer",
    "TCAV",
]
