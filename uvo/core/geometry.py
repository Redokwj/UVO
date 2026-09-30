import math
from typing import Union, Tuple, Optional
import numpy as np
import torch

class SE3:
    """
    Special Euclidean Group SE(3) representation for 3D rigid body transformations.
    Stores rotation as 3x3 matrix R and translation as 3x1 vector t:
        P_world = R * P_cam + t
    Supports both NumPy and PyTorch tensors.
    """
    def __init__(
        self,
        R: Optional[Union[np.ndarray, torch.Tensor]] = None,
        t: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        if R is None:
            self.R = np.eye(3, dtype=np.float64)
        elif isinstance(R, torch.Tensor):
            self.R = R.detach().cpu().numpy().astype(np.float64)
        else:
            self.R = np.array(R, dtype=np.float64)
            
        if t is None:
            self.t = np.zeros(3, dtype=np.float64)
        elif isinstance(t, torch.Tensor):
            self.t = t.detach().cpu().numpy().flatten().astype(np.float64)
        else:
            self.t = np.array(t, dtype=np.float64).flatten()
            
        assert self.R.shape == (3, 3), f"R must be 3x3, got {self.R.shape}"
        assert self.t.shape == (3,), f"t must be 3, got {self.t.shape}"

    @classmethod
    def identity(cls) -> "SE3":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "SE3":
        assert T.shape == (4, 4), f"T must be 4x4 matrix, got {T.shape}"
        return cls(T[:3, :3], T[:3, 3])

    @classmethod
    def from_quat_and_trans(cls, q: Union[list, tuple, np.ndarray], t: Union[list, tuple, np.ndarray]) -> "SE3":
        """
        Create SE3 from quaternion [qx, qy, qz, qw] and translation [tx, ty, tz].
        """
        qx, qy, qz, qw = q
        # Normalize quaternion
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 1e-12:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        R = np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)]
        ], dtype=np.float64)
        return cls(R, t)

    def to_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def to_quat(self) -> np.ndarray:
        """
        Returns quaternion [qx, qy, qz, qw] using Shepperd's algorithm.
        """
        tr = np.trace(self.R)
        if tr > 0.0:
            S = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (self.R[2, 1] - self.R[1, 2]) / S
            qy = (self.R[0, 2] - self.R[2, 0]) / S
            qz = (self.R[1, 0] - self.R[0, 1]) / S
        elif (self.R[0, 0] > self.R[1, 1]) and (self.R[0, 0] > self.R[2, 2]):
            S = math.sqrt(1.0 + self.R[0, 0] - self.R[1, 1] - self.R[2, 2]) * 2.0
            qw = (self.R[2, 1] - self.R[1, 2]) / S
            qx = 0.25 * S
            qy = (self.R[0, 1] + self.R[1, 0]) / S
            qz = (self.R[0, 2] + self.R[2, 0]) / S
        elif self.R[1, 1] > self.R[2, 2]:
            S = math.sqrt(1.0 + self.R[1, 1] - self.R[0, 0] - self.R[2, 2]) * 2.0
            qw = (self.R[0, 2] - self.R[2, 0]) / S
            qx = (self.R[0, 1] + self.R[1, 0]) / S
            qy = 0.25 * S
            qz = (self.R[1, 2] + self.R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + self.R[2, 2] - self.R[0, 0] - self.R[1, 1]) * 2.0
            qw = (self.R[1, 0] - self.R[0, 1]) / S
            qx = (self.R[0, 2] + self.R[2, 0]) / S
            qy = (self.R[1, 2] + self.R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw], dtype=np.float64)

    def inv(self) -> "SE3":
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE3(R_inv, t_inv)

    def __matmul__(self, other: Union["SE3", np.ndarray]) -> Union["SE3", np.ndarray]:
        if isinstance(other, SE3):
            R_new = self.R @ other.R
            t_new = self.R @ other.t + self.t
            return SE3(R_new, t_new)
        elif isinstance(other, np.ndarray):
            # Transform 3D points
            if other.ndim == 1 and other.shape[0] == 3:
                return self.R @ other + self.t
            elif other.ndim == 2 and other.shape[1] == 3:
                return (self.R @ other.T).T + self.t
            elif other.ndim == 2 and other.shape[0] == 3:
                return self.R @ other + self.t[:, None]
            else:
                raise ValueError(f"Unsupported points shape for SE3 transform: {other.shape}")
        else:
            raise TypeError(f"Cannot multiply SE3 with {type(other)}")

    def log(self) -> np.ndarray:
        """
        Lie algebra logarithmic map: SE(3) -> se(3) tangent vector (6-vector: [v_x, v_y, v_z, w_x, w_y, w_z])
        """
        # Rotation angle
        cos_theta = np.clip((np.trace(self.R) - 1.0) / 2.0, -1.0, 1.0)
        theta = math.acos(cos_theta)
        
        if abs(theta) < 1e-6:
            w = np.zeros(3)
            V_inv = np.eye(3)
        else:
            w_hat = (self.R - self.R.T) / (2.0 * math.sin(theta))
            w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta
            
            # Left Jacobian inverse
            A = math.sin(theta) / theta
            B = (1.0 - math.cos(theta)) / (theta * theta)
            w_hat_sq = w_hat @ w_hat
            V = np.eye(3) + B * w_hat * theta + ((1.0 - A) / (theta * theta)) * w_hat_sq * (theta * theta)
            V_inv = np.linalg.inv(V)
            
        v = V_inv @ self.t
        return np.concatenate([v, w])

    @classmethod
    def exp(cls, xi: np.ndarray) -> "SE3":
        """
        Lie algebra exponential map: se(3) 6-vector -> SE(3)
        xi: [v_x, v_y, v_z, w_x, w_y, w_z]
        """
        v = xi[:3]
        w = xi[3:]
        theta = np.linalg.norm(w)
        
        if theta < 1e-6:
            R = np.eye(3) + cls._skew(w)
            V = np.eye(3)
        else:
            axis = w / theta
            w_hat = cls._skew(axis)
            R = np.eye(3) + math.sin(theta) * w_hat + (1.0 - math.cos(theta)) * (w_hat @ w_hat)
            V = np.eye(3) + ((1.0 - math.cos(theta)) / theta) * w_hat + ((theta - math.sin(theta)) / theta) * (w_hat @ w_hat)
            
        t = V @ v
        return cls(R, t)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return skew_symmetric(v)

    def __repr__(self) -> str:
        q = self.to_quat()
        return f"SE3(t=[{self.t[0]:.3f}, {self.t[1]:.3f}, {self.t[2]:.3f}], q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}])"


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Computes 3x3 skew-symmetric matrix [v]_x from 3-vector."""
    v = np.asarray(v).flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) Lie algebra (3-vector) to SO(3) rotation matrix (Rodrigues formula)."""
    w = np.asarray(w, dtype=np.float64).flatten()
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) + skew_symmetric(w)
    axis = w / theta
    k = skew_symmetric(axis)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) rotation matrix to so(3) Lie algebra (3-vector)."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    w_hat = (R - R.T) / (2.0 * math.sin(theta))
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta


def project_points(pts_3d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points in camera frame to 2D image coordinates.
    pts_3d: (N, 3)
    K: (3, 3) camera intrinsic matrix
    Returns: (pixels: (N, 2), valid_mask: (N,) bool for positive depth Z > 0.1)
    """
    assert pts_3d.shape[1] == 3, f"pts_3d must have shape (N, 3), got {pts_3d.shape}"
    z = pts_3d[:, 2]
    valid_mask = z > 0.1
    
    # Avoid division by zero
    z_safe = np.where(valid_mask, z, 1.0)
    x_norm = pts_3d[:, 0] / z_safe
    y_norm = pts_3d[:, 1] / z_safe
    
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    
    pixels = np.column_stack([u, v])
    return pixels, valid_mask


def unproject_pixels(pixels: np.ndarray, depths: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Unproject 2D pixel coordinates and corresponding metric depths into 3D points.
    pixels: (N, 2)
    depths: (N,) or (N, 1) in meters
    K: (3, 3) camera intrinsic matrix
    Returns: (N, 3) points in camera coordinate frame
    """
    depths = depths.flatten()
    u = pixels[:, 0]
    v = pixels[:, 1]
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x = (u - cx) * depths / fx
    y = (v - cy) * depths / fy
    z = depths
    
    return np.column_stack([x, y, z])
