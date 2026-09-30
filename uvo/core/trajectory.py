import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

from .geometry import SE3
from .sensors import GeoCoordinateTransformer


@dataclass
class TrajectoryPoint:
    """
    A single timestamped state along the vehicle / camera trajectory.
    """
    timestamp: float
    pose_wc: SE3                       # Camera or Body frame in World (P_world = pose_wc * P_local)
    velocity: Optional[np.ndarray] = None   # [vx, vy, vz] in m/s
    status: str = "TRACKED"            # TRACKED, KEYFRAME, LOOP_CLOSED, DEGRADED
    covariance: Optional[np.ndarray] = None # (6, 6) pose uncertainty
    frame_id: Optional[int] = None
    ref_kf_id: Optional[int] = None
    pose_rel_kf: Optional[SE3] = None

    @property
    def position(self) -> np.ndarray:
        return self.pose_wc.t

    @property
    def rotation(self) -> np.ndarray:
        return self.pose_wc.R

    @property
    def quaternion(self) -> np.ndarray:
        """[qx, qy, qz, qw]"""
        return self.pose_wc.to_quat()


class Trajectory:
    """
    Manager and exporter for visual-inertial and visual odometry trajectories.
    Supports TUM format, CSV, GeoJSON (WGS84), and ATE/RPE metric evaluation.
    """
    def __init__(self, name: str = "uvo_trajectory"):
        self.name = name
        self.points: List[TrajectoryPoint] = []

    def __len__(self) -> int:
        return len(self.points)

    def add_point(self, point: TrajectoryPoint):
        self.points.append(point)

    def add_pose(
        self,
        timestamp: float,
        pose_wc: SE3,
        velocity: Optional[np.ndarray] = None,
        status: str = "TRACKED",
        frame_id: Optional[int] = None,
        ref_kf_id: Optional[int] = None,
        pose_rel_kf: Optional[SE3] = None
    ):
        self.points.append(TrajectoryPoint(
            timestamp=timestamp,
            pose_wc=pose_wc,
            velocity=velocity,
            status=status,
            frame_id=frame_id,
            ref_kf_id=ref_kf_id,
            pose_rel_kf=pose_rel_kf
        ))

    def get_timestamps(self) -> np.ndarray:
        return np.array([p.timestamp for p in self.points], dtype=np.float64)

    def get_positions(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([p.position for p in self.points], dtype=np.float64)

    def get_poses_matrix(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, 4, 4), dtype=np.float64)
        return np.array([p.pose_wc.to_matrix() for p in self.points], dtype=np.float64)

    def save_tum(self, filepath: str):
        """
        Saves trajectory in standard TUM format:
        timestamp tx ty tz qx qy qz qw
        """
        with open(filepath, "w") as f:
            for p in self.points:
                t = p.position
                q = p.quaternion
                f.write(f"{p.timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    def save_csv(self, filepath: str):
        """
        Saves trajectory as comprehensive CSV with position, orientation (quaternion and Euler angles), and status.
        """
        with open(filepath, "w") as f:
            f.write("timestamp,x,y,z,qx,qy,qz,qw,roll_deg,pitch_deg,yaw_deg,status\n")
            for p in self.points:
                t = p.position
                q = p.quaternion
                
                # Euler angles from rotation matrix (ZYX convention)
                R = p.rotation
                sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6
                if not singular:
                    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    pitch = math.degrees(math.atan2(-R[2, 0], sy))
                    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                else:
                    roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
                    pitch = math.degrees(math.atan2(-R[2, 0], sy))
                    yaw = 0.0

                f.write(f"{p.timestamp:.6f},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f},"
                        f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},"
                        f"{roll:.3f},{pitch:.3f},{yaw:.3f},{p.status}\n")

    def save_geojson(self, filepath: str, geo_transformer: GeoCoordinateTransformer):
        """
        Converts local metric trajectory to WGS84 coordinates and exports as GeoJSON FeatureCollection.
        Compatible with QGIS, Google Earth, Leaflet, and standard mission planning software.
        """
        coords_geojson = []
        features = []

        for p in self.points:
            t = p.position
            # In UVO: x = East, y = North (or z depending on optical/body convention),
            # here we treat [t[0], t[1], t[2]] as local ENU meters
            lat, lon, alt = geo_transformer.enu_to_geodetic(t[0], t[1], t[2])
            coords_geojson.append([lon, lat, alt])

        # 1. Trajectory line feature
        line_feature = {
            "type": "Feature",
            "properties": {
                "name": self.name,
                "num_points": len(self.points),
                "duration_sec": (self.points[-1].timestamp - self.points[0].timestamp) if len(self.points) > 1 else 0.0
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords_geojson
            }
        }
        features.append(line_feature)

        geojson_doc = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(filepath, "w") as f:
            json.dump(geojson_doc, f, indent=2)

    def compute_ate(
        self,
        gt_trajectory: "Trajectory",
        max_time_diff: float = 0.05,
        align_scale: bool = False
    ) -> Dict[str, float]:
        """
        Computes Absolute Trajectory Error (ATE) RMSE against ground-truth reference trajectory.
        Uses nearest-neighbor timestamp matching and Umeyama rigid/similarity alignment.
        """
        if len(self.points) == 0 or len(gt_trajectory.points) == 0:
            return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "matches": 0}

        est_times = self.get_timestamps()
        gt_times = gt_trajectory.get_timestamps()
        est_pos = self.get_positions()
        gt_pos = gt_trajectory.get_positions()

        # If timestamps have a large offset (> 1000s), align by relative time from start
        if abs(gt_times[0] - est_times[0]) > 100.0:
            est_t_search = est_times - est_times[0]
            gt_t_search = gt_times - gt_times[0]
        else:
            est_t_search = est_times
            gt_t_search = gt_times

        matched_est = []
        matched_gt = []

        for i, t_est in enumerate(est_t_search):
            idx = np.argmin(np.abs(gt_t_search - t_est))
            if abs(gt_t_search[idx] - t_est) <= max(max_time_diff, 1.0):
                matched_est.append(est_pos[i])
                matched_gt.append(gt_pos[idx])

        if len(matched_est) < 3:
            return {"rmse": -1.0, "mean": -1.0, "median": -1.0, "std": -1.0, "max": -1.0, "scale": 1.0, "matches": len(matched_est)}

        P = np.array(matched_est)  # (N, 3)
        Q = np.array(matched_gt)   # (N, 3)

        # Umeyama alignment (Find R, t, s such that s * R * P + t ≈ Q)
        R_align, t_align, scale_align = self._umeyama_alignment(P, Q, align_scale=align_scale)
        P_aligned = (scale_align * (R_align @ P.T)).T + t_align

        # Errors
        errors = np.linalg.norm(P_aligned - Q, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        std_err = float(np.std(errors))
        max_err = float(np.max(errors))

        return {
            "rmse": rmse,
            "mean": mean_err,
            "median": median_err,
            "std": std_err,
            "max": max_err,
            "scale": scale_align,
            "matches": len(matched_est)
        }

    @staticmethod
    def _umeyama_alignment(P: np.ndarray, Q: np.ndarray, align_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Standard Umeyama 3D-3D point set registration.
        P: source (N, 3)
        Q: target (N, 3)
        """
        n = P.shape[0]
        mu_p = np.mean(P, axis=0)
        mu_q = np.mean(Q, axis=0)

        P_centered = P - mu_p
        Q_centered = Q - mu_q

        sigma_p = np.sum(P_centered ** 2) / n
        H = (P_centered.T @ Q_centered) / n

        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        d = np.linalg.det(V @ U.T)
        S_diag = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])

        R = V @ S_diag @ U.T
        scale = 1.0
        if align_scale and sigma_p > 1e-8:
            scale = float(np.trace(S_diag @ np.diag(S)) / sigma_p)

        t = mu_q - scale * (R @ mu_p)
        return R, t, scale
