import math
from typing import Optional, Tuple, List
import numpy as np
import torch

from ..core.geometry import SE3, so3_log


class RobustLoss:
    """
    Huber loss function for downweighting outlier reprojections.
    """
    @staticmethod
    def huber_weight(error: float, delta: float = 1.5) -> float:
        if error <= delta:
            return 1.0
        return float(delta / max(1e-8, error))


class VisualReprojectionFactor:
    """
    Reprojection factor evaluating difference between observed 2D pixel coordinates
    and projected 3D world landmark position.
    """
    def __init__(
        self,
        keyframe_id: int,
        track_id: int,
        observed_uv: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        weight: float = 1.0,
        huber_delta: float = 2.0
    ):
        self.keyframe_id = keyframe_id
        self.track_id = track_id
        self.observed_uv = np.asarray(observed_uv, dtype=np.float64)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.weight = weight
        self.huber_delta = huber_delta

    def evaluate(self, pose_cw: SE3, point_3d_world: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes 2D reprojection residual [r_u, r_v] and robust Huber loss weight.
        """
        # Transform point to camera frame: P_cam = R * P_world + t
        P_cam = pose_cw.R @ point_3d_world + pose_cw.t
        z = P_cam[2]
        
        if z <= 0.05:
            # Point behind or right at camera plane: heavy penalty
            res = np.array([100.0, 100.0], dtype=np.float64)
            return res, 0.001

        u_proj = self.fx * (P_cam[0] / z) + self.cx
        v_proj = self.fy * (P_cam[1] / z) + self.cy

        raw_res = np.array([u_proj - self.observed_uv[0], v_proj - self.observed_uv[1]], dtype=np.float64)
        err = float(np.linalg.norm(raw_res))
        w = RobustLoss.huber_weight(err, self.huber_delta) * self.weight
        return raw_res * math.sqrt(w), w


class MetricDepthFactor:
    """
    Scale anchor factor from UniDepth V2 metric depth map.
    Penalizes deviations between camera-frame Z coordinate and foundation depth prediction.
    """
    def __init__(
        self,
        keyframe_id: int,
        track_id: int,
        depth_prior: float,
        confidence_weight: float = 1.0
    ):
        self.keyframe_id = keyframe_id
        self.track_id = track_id
        self.depth_prior = depth_prior
        self.confidence_weight = confidence_weight

    def evaluate(self, pose_cw: SE3, point_3d_world: np.ndarray) -> float:
        """
        Computes 1D metric depth scalar residual in meters.
        """
        P_cam = pose_cw.R @ point_3d_world + pose_cw.t
        z_est = P_cam[2]
        res = (z_est - self.depth_prior) * math.sqrt(self.confidence_weight)
        return float(res)


class RelativePoseFactor:
    """
    Smoothness and odometry motion prior between successive keyframes i and j.
    """
    def __init__(
        self,
        keyframe_i: int,
        keyframe_j: int,
        T_ji_measured: SE3,
        rot_weight: float = 10.0,
        trans_weight: float = 5.0
    ):
        self.keyframe_i = keyframe_i
        self.keyframe_j = keyframe_j
        self.T_ji_measured = T_ji_measured # P_j = T_ji * P_i
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def evaluate(self, pose_cw_i: SE3, pose_cw_j: SE3) -> np.ndarray:
        """
        Computes 6-vector SE(3) error between estimated and measured relative motion.
        T_ji_est = pose_cw_j * (pose_cw_i)^-1
        Error = Log( (T_ji_measured)^-1 * T_ji_est )
        """
        T_ji_est = pose_cw_j @ pose_cw_i.inv()
        T_err = self.T_ji_measured.inv() @ T_ji_est
        xi = T_err.log() # 6-vector: [vx, vy, vz, wx, wy, wz]
        
        # Apply weights: translation vs rotation
        res = np.zeros(6, dtype=np.float64)
        res[:3] = xi[:3] * math.sqrt(self.trans_weight)
        res[3:] = xi[3:] * math.sqrt(self.rot_weight)
        return res


class GravityAlignmentFactor:
    """
    Locks camera pitch and roll against measured gravity vector from rigid IMU mounting
    or foundation model ground-plane normal (Visual IMU).
    """
    def __init__(
        self,
        keyframe_id: int,
        gravity_measured_cam: np.ndarray,
        gravity_world: Optional[np.ndarray] = None,
        weight: float = 15.0
    ):
        self.keyframe_id = keyframe_id
        self.gravity_measured_cam = np.asarray(gravity_measured_cam, dtype=np.float64).flatten()
        self.gravity_world = np.asarray(gravity_world if gravity_world is not None else [0.0, 1.0, 0.0], dtype=np.float64).flatten()
        self.weight = weight

    def evaluate(self, pose_cw: SE3) -> np.ndarray:
        """
        Computes 3D alignment residual: R_cw * g_world - g_measured_cam
        """
        g_pred = pose_cw.R @ self.gravity_world
        res = (g_pred - self.gravity_measured_cam) * math.sqrt(self.weight)
        return res


class NonHolonomicTrackedFactor:
    """
    Kinematic non-holonomic motion constraints for tracked crawler chassis (гусеничні платформи).
    Enforces zero lateral slip (v_lateral = 0 in body frame) and bounds parasitic roll oscillations.
    """
    def __init__(
        self,
        keyframe_i: int,
        keyframe_j: int,
        w_lateral: float = 25.0,
        w_roll: float = 12.0
    ):
        self.keyframe_i = keyframe_i
        self.keyframe_j = keyframe_j
        self.w_lateral = w_lateral
        self.w_roll = w_roll

    def evaluate(self, pose_cw_i: SE3, pose_cw_j: SE3) -> np.ndarray:
        """
        Computes lateral slip and roll residuals in chassis frame.
        """
        C_i = -pose_cw_i.R.T @ pose_cw_i.t
        C_j = -pose_cw_j.R.T @ pose_cw_j.t
        disp_body = pose_cw_i.R @ (C_j - C_i)
        
        # Lateral displacement in camera body frame (X is lateral)
        lat_slip = disp_body[0]
        
        # Relative rotation
        R_ji = pose_cw_j.R @ pose_cw_i.R.T
        roll_err = R_ji[0, 1]  # Parasitic roll
        
        return np.array([lat_slip * math.sqrt(self.w_lateral), roll_err * math.sqrt(self.w_roll)], dtype=np.float64)


class MarginalizationPriorFactor:
    """
    Schur complement information prior preserving historical pose constraints
    when older keyframes exit the sliding optimization window.
    """
    def __init__(
        self,
        keyframe_id: int,
        prior_pose: SE3,
        information_sqrt: Optional[np.ndarray] = None,
        default_weight: float = 20.0
    ):
        self.keyframe_id = keyframe_id
        self.prior_pose = prior_pose
        if information_sqrt is not None:
            self.L = np.asarray(information_sqrt, dtype=np.float64)
        else:
            self.L = np.eye(6, dtype=np.float64) * math.sqrt(default_weight)

    def evaluate(self, pose_cw: SE3) -> np.ndarray:
        """
        Computes 6D SE(3) prior residual: L * Log(prior_pose^-1 * pose_cw)
        """
        T_err = self.prior_pose.inv() @ pose_cw
        xi = T_err.log()
        return self.L @ xi
