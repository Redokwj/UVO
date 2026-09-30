import os
import sys
import time
from typing import Optional
import numpy as np
import cv2

from ..core.frame import CameraIntrinsics
from .base import BaseDepthProvider, DepthPrediction


class Metric3DProvider(BaseDepthProvider):
    """
    Secondary Metric Depth Provider based on Metric3D v2 (ViT-Small).
    Canonical focal length scaling (f=1000) for outdoor monocular scenes.
    """
    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self.device_str = device or ("cuda" if np.show_config() else "cpu")
        
        from .models.metric3d import Metric3DV2
        self.model = Metric3DV2(device=self.device_str)

    def predict_depth(
        self,
        image: np.ndarray,
        intrinsics: Optional[CameraIntrinsics] = None
    ) -> DepthPrediction:
        t0 = time.perf_counter()
        
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        k_tuple = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) if intrinsics else None
        depth = self.model.predict_depth(image_rgb, intrinsics=k_tuple)

        elapsed = time.perf_counter() - t0

        return DepthPrediction(
            depth_map=depth,
            confidence_map=None,
            uncertainty_map=None,
            predicted_intrinsics=intrinsics,
            elapsed_time_sec=elapsed,
            model_name="Metric3D-v2-ViT-Small"
        )
