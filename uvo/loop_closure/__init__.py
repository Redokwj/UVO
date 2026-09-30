"""
UVO Loop Closure, Visual Place Recognition, and Pose Graph Optimization Module (Module 4).
Provides DINOv2 place recognition, metric 3D-3D Umeyama RANSAC verification, and global PGO.
"""

from .vpr import VPREngine, VPRDatabase
from .verifier import LoopVerifier, LoopVerificationResult
from .pgo import PoseGraphEdge, PoseGraphOptimizer
from .manager import LoopClosureManager, LoopEvent

__all__ = [
    "VPREngine",
    "VPRDatabase",
    "LoopVerifier",
    "LoopVerificationResult",
    "PoseGraphEdge",
    "PoseGraphOptimizer",
    "LoopClosureManager",
    "LoopEvent"
]
