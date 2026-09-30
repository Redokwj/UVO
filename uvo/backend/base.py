import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from ..core.geometry import SE3
from ..core.frame import Keyframe


@dataclass
class VisualObservation:
    """
    Observation of a 3D landmark in a specific keyframe.
    """
    keyframe_id: int
    pixel_uv: np.ndarray             # (2,) [u, v]
    depth_prior: Optional[float] = None # Metric depth in meters (from UniDepth V2)
    depth_weight: float = 1.0        # Confidence / inverse variance
    pixel_covariance: Optional[np.ndarray] = None # (2, 2)


@dataclass
class VisualTrack:
    """
    Multi-view visual landmark track across multiple keyframes.
    """
    track_id: int
    observations: List[VisualObservation] = field(default_factory=list)
    point_3d_world: Optional[np.ndarray] = None # (3,) [X, Y, Z] in world coordinate frame
    is_outlier: bool = False

    def add_observation(self, obs: VisualObservation):
        self.observations.append(obs)


@dataclass
class OptimizationResult:
    """
    Summary of Bundle Adjustment / Factor Graph optimization.
    """
    optimized_poses: Dict[int, SE3] = field(default_factory=dict)       # keyframe_id -> SE3 pose_cw
    optimized_landmarks: Dict[int, np.ndarray] = field(default_factory=dict) # track_id -> (3,) world pos
    initial_cost: float = 0.0
    final_cost: float = 0.0
    num_iterations: int = 0
    is_converged: bool = False
    elapsed_time_sec: float = 0.0
    solver_name: str = "BaseOptimizer"


class BaseOptimizer(ABC):
    """
    Abstract interface for Sliding Window Factor Graph Optimizer (Module 3).
    """
    def __init__(self, window_size: int = 8, max_iterations: int = 15):
        self.window_size = window_size
        self.max_iterations = max_iterations

    @abstractmethod
    def optimize(
        self,
        keyframes: List[Keyframe],
        tracks: List[VisualTrack],
        gravity_vector: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Executes non-linear optimization over active keyframes and visual landmarks.
        """
        pass
