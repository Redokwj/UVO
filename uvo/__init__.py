"""
UVO: Unnamed Visual Odometry & SLAM
Next-Generation Modular Metric Visual SLAM Pipeline.
"""

from .core.geometry import SE3, project_points, unproject_pixels
from .core.frame import CameraIntrinsics, Frame, Keyframe
from .core.sensors import (
    NavSatStatus,
    CameraExtrinsics,
    ImuMeasurement,
    ImuBuffer,
    NavSatMeasurement,
    GeoCoordinateTransformer
)
from .core.trajectory import TrajectoryPoint, Trajectory
from .frontend.base import BaseTracker, TrackingResult, FrontendMode
from .frontend.xfeat_tracker import XFeatTracker
from .frontend.liftfeat_tracker import LiftFeatTracker
from .frontend.hybrid_tracker import HybridTracker
from .metric.base import BaseDepthProvider, DepthPrediction, DepthModelType
from .metric.unidepth_provider import UniDepthProvider
from .metric.metric3d_provider import Metric3DProvider
from .backend.base import BaseOptimizer, VisualTrack, OptimizationResult
from .backend.solver import SlidingWindowOptimizer
from .backend.track_manager import TrackManager
from .loop_closure.vpr import VPREngine, VPRDatabase
from .loop_closure.verifier import LoopVerifier, LoopVerificationResult
from .loop_closure.pgo import PoseGraphOptimizer, PoseGraphEdge
from .loop_closure.manager import LoopClosureManager, LoopEvent
from .pipeline import UVOPipeline, UVOConfig

__version__ = "0.1.0"

__all__ = [
    "SE3",
    "project_points",
    "unproject_pixels",
    "CameraIntrinsics",
    "Frame",
    "Keyframe",
    "NavSatStatus",
    "CameraExtrinsics",
    "ImuMeasurement",
    "ImuBuffer",
    "NavSatMeasurement",
    "GeoCoordinateTransformer",
    "TrajectoryPoint",
    "Trajectory",
    "BaseTracker",
    "TrackingResult",
    "FrontendMode",
    "XFeatTracker",
    "LiftFeatTracker",
    "HybridTracker",
    "BaseDepthProvider",
    "DepthPrediction",
    "DepthModelType",
    "UniDepthProvider",
    "Metric3DProvider",
    "BaseOptimizer",
    "VisualTrack",
    "OptimizationResult",
    "SlidingWindowOptimizer",
    "TrackManager",
    "VPREngine",
    "VPRDatabase",
    "LoopVerifier",
    "LoopVerificationResult",
    "PoseGraphOptimizer",
    "PoseGraphEdge",
    "LoopClosureManager",
    "LoopEvent",
    "UVOPipeline",
    "UVOConfig"
]
