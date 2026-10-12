from __future__ import annotations

from .circuits import ActivationPatchingEngine
from .counterfactuals import WachterCounterfactualExplainer
from .engine import ExplanationResult, explain
from .importance import permutation_feature_importance
from .mechanistic import (
    ActivationPatching,
    AttentionRollout,
    DirectLogitAttribution,
    JumpReLU,
    TopKSAE,
)
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
    "TopKSAE",
    "JumpReLU",
    "ActivationPatching",
    "WachterCounterfactualExplainer",
    "TCAV",
    "ActivationPatchingEngine",
]
