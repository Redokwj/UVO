import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from ..core.geometry import SE3
from ..core.frame import Frame, CameraIntrinsics


class FrontendMode(str, Enum):
    KLT = "klt"             # Sub-pixel Pyramidal Lucas-Kanade Optical Flow (60-100+ FPS)
    HYBRID = "hybrid"       # XFeat (fast VO) + LiftFeat (turns & keyframes)
    LIFTFEAT = "liftfeat"   # ICRA 2025 SOTA robust turn tracker
    XFEAT = "xfeat"         # CVPR 2024 edge-optimized lightweight tracker
    ELOFOR = "eloftr"       # CVPR 2024 EfficientLoFTR semi-dense matcher


@dataclass
class TrackingResult:
    """
    Result of 2D-2D feature tracking or 2D-3D PnP tracking between two visual frames.
    """
    matched_kpts0: np.ndarray              # (M, 2) float32 coordinates in frame0
    matched_kpts1: np.ndarray              # (M, 2) float32 coordinates in frame1
    inlier_mask: np.ndarray                # (M,) bool mask from robust estimator (MAGSAC++)
    num_matches: int                       # Raw match count M
    num_inliers: int                       # Inlier match count
    inlier_ratio: float                    # Inlier percentage (0.0 to 100.0)
    mean_parallax_px: float                # Average pixel displacement of inliers
    
    relative_pose: Optional[SE3] = None    # Relative motion: P_1 = relative_pose * P_0
    is_keyframe_candidate: bool = False    # True if baseline/parallax justifies new keyframe
    elapsed_time_sec: float = 0.0          # Inference + matching latency
    tracker_name: str = "BaseTracker"


class BaseTracker(ABC):
    """
    Abstract interface for Front-End visual trackers.
    Provides standard 2D-2D relative pose recovery via Essential Matrix MAGSAC++ and PnP solver.
    """
    def __init__(self, keyframe_parallax_threshold: float = 30.0, min_inliers: int = 40):
        self.keyframe_parallax_threshold = keyframe_parallax_threshold
        self.min_inliers = min_inliers

    @abstractmethod
    def track(self, frame0: Frame, frame1: Frame) -> TrackingResult:
        """
        Track visual features between frame0 and frame1 and compute relative camera motion.
        """
        pass

    def estimate_relative_pose_2d2d(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        intrinsics: CameraIntrinsics,
        reproj_thresh_px: float = 2.0
    ) -> Tuple[Optional[SE3], np.ndarray, int]:
        """
        Recovers relative camera rotation and up-to-scale translation using Essential Matrix with USAC_MAGSAC.
        """
        if len(kpts0) < 8 or len(kpts1) < 8:
            return None, np.zeros(len(kpts0), dtype=bool), 0

        # Ignore bottom hood / rover bumper
        max_y = intrinsics.height * 0.82
        valid_y = (kpts0[:, 1] < max_y) & (kpts1[:, 1] < max_y)
        if valid_y.sum() >= 8:
            kpts0_use = kpts0[valid_y]
            kpts1_use = kpts1[valid_y]
        else:
            kpts0_use = kpts0
            kpts1_use = kpts1

        K = intrinsics.to_matrix()
        
        # 1. Estimate Essential Matrix with MAGSAC++
        E, inlier_mask = cv2.findEssentialMat(
            kpts0_use, kpts1_use,
            cameraMatrix=K,
            method=cv2.USAC_MAGSAC,
            prob=0.999,
            threshold=reproj_thresh_px,
            maxIters=2500
        )

        if E is None or inlier_mask is None:
            return None, np.zeros(len(kpts0), dtype=bool), 0

        mask_bool = inlier_mask.ravel().astype(bool)
        num_inliers = int(mask_bool.sum())

        full_mask = np.zeros(len(kpts0), dtype=bool)
        if valid_y.sum() >= 8:
            full_mask[valid_y] = mask_bool
        else:
            full_mask = mask_bool

        if num_inliers < self.min_inliers:
            return None, full_mask, num_inliers

        # Check parallax for stationary detection
        parallax = self.compute_parallax(kpts0_use, kpts1_use, mask_bool)
        if parallax < 1.2:
            # Vehicle is stationary: zero translation and rotation
            return SE3.identity(), full_mask, num_inliers

        # 2. Decompose Essential Matrix and resolve cheirality (points in front of camera)
        num_valid, R, t, pose_mask = cv2.recoverPose(E, kpts0_use, kpts1_use, cameraMatrix=K, mask=inlier_mask.copy())
        
        sub_final_mask = (pose_mask.ravel() > 0) & mask_bool
        final_inliers = int(sub_final_mask.sum())

        full_final_mask = np.zeros(len(kpts0), dtype=bool)
        if valid_y.sum() >= 8:
            full_final_mask[valid_y] = sub_final_mask
        else:
            full_final_mask = sub_final_mask

        if final_inliers < self.min_inliers:
            if num_inliers >= self.min_inliers and parallax < 2.5:
                return SE3.identity(), full_final_mask, num_inliers
            return None, full_final_mask, num_inliers

        relative_pose = SE3(R=R, t=t.flatten())
        return relative_pose, full_final_mask, final_inliers

    @staticmethod
    def compute_parallax(kpts0: np.ndarray, kpts1: np.ndarray, inlier_mask: np.ndarray) -> float:
        """
        Computes the median pixel displacement between matched inlier feature points.
        """
        if inlier_mask is None or inlier_mask.sum() == 0:
            return 0.0
        pts0 = kpts0[inlier_mask]
        pts1 = kpts1[inlier_mask]
        displacements = np.linalg.norm(pts1 - pts0, axis=1)
        return float(np.median(displacements))
