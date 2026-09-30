"""
UVO Factor Graph Optimization and Bundle Adjustment Module (Module 3).
Provides sliding window Levenberg-Marquardt optimizer with metric depth anchoring.
"""

from .base import (
    BaseOptimizer,
    VisualObservation,
    VisualTrack,
    OptimizationResult
)
from .factors import (
    RobustLoss,
    VisualReprojectionFactor,
    MetricDepthFactor,
    RelativePoseFactor,
    GravityAlignmentFactor,
    NonHolonomicTrackedFactor,
    MarginalizationPriorFactor
)
from .solver import SlidingWindowOptimizer
from .theseus_optimizer import TheseusFactorGraphOptimizer
from .track_manager import TrackManager

__all__ = [
    "BaseOptimizer",
    "VisualObservation",
    "VisualTrack",
    "OptimizationResult",
    "RobustLoss",
    "VisualReprojectionFactor",
    "MetricDepthFactor",
    "RelativePoseFactor",
    "GravityAlignmentFactor",
    "NonHolonomicTrackedFactor",
    "MarginalizationPriorFactor",
    "SlidingWindowOptimizer",
    "TheseusFactorGraphOptimizer",
    "TrackManager"
]
