"""Geometric Optimal Transport, Wasserstein Metrics, and Sinkhorn Barycenters for Chokkhu."""

from __future__ import annotations

from .sinkhorn import (
    SinkhornOptimalTransport,
    sinkhorn_distance,
)
from .barycenter import (
    WassersteinBarycenter,
    wasserstein_barycenter,
)

__all__ = [
    "SinkhornOptimalTransport",
    "sinkhorn_distance",
    "WassersteinBarycenter",
    "wasserstein_barycenter",
]
