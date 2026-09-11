"""Continual and Lifelong Learning algorithms for Chokkhu."""

from __future__ import annotations

from .ewc import EWC, ElasticWeightConsolidation
from .replay import DERPlusPlus, DarkExperienceReplay

__all__ = [
    "ElasticWeightConsolidation",
    "EWC",
    "DarkExperienceReplay",
    "DERPlusPlus",
]
