import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np

import torch

from ..core.frame import CameraIntrinsics


class DepthModelType(str, Enum):
    UNIDEPTH = "unidepth"       # UniDepth V2 (ViT-Small/14, ETH Zurich) - Default profile
    METRIC3D = "metric3d"       # Metric3D v2 (ViT-Small, ByteDance)
    DEPTH_ANYTHING = "depth_anything" # Depth Anything v2 Metric


@dataclass
class DepthPrediction:
    """
    Standard output container from metric monocular depth foundation models.
    """
    depth_map: np.ndarray                   # (H, W) float32 array in absolute metric meters
    confidence_map: Optional[np.ndarray] = None # (H, W) float32 confidence [0.0, 1.0]
    uncertainty_map: Optional[np.ndarray] = None# (H, W) float32 variance/sigma in meters
    predicted_intrinsics: Optional[CameraIntrinsics] = None
    gravity_cam: Optional[np.ndarray] = None    # (3,) unit vector pointing down in camera frame (Visual IMU)
    ground_height: Optional[float] = None      # Measured distance along gravity to ground plane (meters)
    elapsed_time_sec: float = 0.0
    model_name: str = "BaseDepthProvider"


class BaseDepthProvider(ABC):
    """
    Abstract interface for metric foundation depth estimators (Module 2).
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict_depth(
        self,
        image: np.ndarray,
        intrinsics: Optional[CameraIntrinsics] = None
    ) -> DepthPrediction:
        """
        Predict metric depth map in meters for a given image.
        """
        pass
