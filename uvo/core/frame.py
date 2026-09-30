from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
import cv2

from .geometry import SE3, unproject_pixels

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters with distortion coefficients.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        if abs(self.k1) < 1e-8 and abs(self.k2) < 1e-8:
            return img
        K = self.to_matrix()
        D = self.dist_coeffs()
        return cv2.undistort(img, K, D, None, K)

    @classmethod
    def from_tuple(cls, calib: tuple, width: int = 1920, height: int = 1080) -> "CameraIntrinsics":
        """
        Supports tuple: (fx, fy, cx, cy) or (fx, fy, cx, cy, k1, k2, p1, p2, k3)
        """
        fx, fy, cx, cy = calib[:4]
        k1 = calib[4] if len(calib) > 4 else 0.0
        k2 = calib[5] if len(calib) > 5 else 0.0
        p1 = calib[6] if len(calib) > 6 else 0.0
        p2 = calib[7] if len(calib) > 7 else 0.0
        k3 = calib[8] if len(calib) > 8 else 0.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)


@dataclass
class Frame:
    """
    Standard visual odometry frame processed by Front-End (Module 1).
    """
    frame_id: int
    timestamp: float
    image: np.ndarray  # Undistorted RGB or Grayscale
    intrinsics: CameraIntrinsics
    pose_cw: SE3 = field(default_factory=SE3.identity)  # World to Camera: P_cam = pose_cw * P_world
    
    # Visual features from Front-End (LiftFeat / XFeat)
    keypoints: Optional[np.ndarray] = None    # (N, 2) float32 coordinates
    descriptors: Optional[np.ndarray] = None  # (N, D) float32 or uint8 feature vectors
    scores: Optional[np.ndarray] = None       # (N,) detection confidence
    covariances: Optional[np.ndarray] = None  # (N, 2, 2) subpixel spatial uncertainty (from RaCo)
    
    # Associated 3D map point IDs in global SLAM map
    landmark_ids: Optional[np.ndarray] = None # (N,) int64 (-1 if untracked)
    
    # Multi-tracker feature cache: {tracker_name: {"keypoints": ..., "descriptors": ..., "scores": ...}}
    feature_cache: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    
    is_keyframe: bool = False

    @property
    def pose_wc(self) -> SE3:
        """Camera to World pose (Vehicle position in world coordinate frame)"""
        return self.pose_cw.inv()


@dataclass
class Keyframe:
    """
    Full Keyframe with Metric Conditioning and Place Recognition descriptors.
    Maintained in Sliding Window BA (Module 3) and Global Pose Graph (Module 4).
    """
    frame: Frame
    
    # Metric conditioning data from UniDepth V2 (Module 2)
    depth_map: Optional[np.ndarray] = None        # (H, W) float32 in meters
    depth_uncertainty: Optional[np.ndarray] = None# (H, W) float32 sigma
    points_3d: Optional[np.ndarray] = None        # (M, 3) 3D points in camera frame in meters
    gravity_cam: Optional[np.ndarray] = None      # (3,) Visual IMU ground gravity normal in camera frame
    
    # Place recognition global descriptor (DINOv2 + SALAD / AnyLoc)
    vpr_descriptor: Optional[np.ndarray] = None   # (512,) float32 embedding
    
    # Covisibility Graph connections: keyframe_id -> number of shared visual tracks
    connected_keyframes: Dict[int, int] = field(default_factory=dict)
    
    # Accumulated odometric arc-length along the trajectory (speed-independent)
    accumulated_distance_m: float = 0.0

    @property
    def frame_id(self) -> int:
        return self.frame.frame_id

    @property
    def timestamp(self) -> float:
        return self.frame.timestamp

    @property
    def pose_cw(self) -> SE3:
        return self.frame.pose_cw

    @property
    def pose_wc(self) -> SE3:
        return self.frame.pose_wc

    def extract_metric_points(self, kpts: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts 3D coordinates in meters for given 2D keypoints using the metric depth map.
        """
        if self.depth_map is None:
            return np.empty((0, 3), dtype=np.float64)
            
        pts = kpts if kpts is not None else self.frame.keypoints
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
            
        H, W = self.depth_map.shape[:2]
        u = np.clip(pts[:, 0].astype(int), 0, W - 1)
        v = np.clip(pts[:, 1].astype(int), 0, H - 1)
        depths = self.depth_map[v, u]
        
        K = self.frame.intrinsics.to_matrix()
        return unproject_pixels(pts, depths, K)
