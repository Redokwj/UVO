"""
UVO Metric Depth Conditioning Module (Module 2).
Provides scale-consistent 3D geometric depth estimation (UniDepth V2 default, Metric3D v2).
"""

from .base import (
    BaseDepthProvider,
    DepthPrediction,
    DepthModelType
)
from .unidepth_provider import UniDepthProvider
from .metric3d_provider import Metric3DProvider
from .road_prior import RoadPlanePriorStub

__all__ = [
    "BaseDepthProvider",
    "DepthPrediction",
    "DepthModelType",
    "UniDepthProvider",
    "Metric3DProvider",
    "RoadPlanePriorStub"
]
