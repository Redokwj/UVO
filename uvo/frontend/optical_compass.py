import math
from typing import Optional, Tuple
import numpy as np

from ..core.frame import CameraIntrinsics


class OpticalCompass:
    """
    Optical Compass & Focus of Expansion (FOE) Heading Anchor.
    
    Extracts the visual Focus of Expansion (FOE) and vanishing heading from dense/sparse
    optical flow vectors. Anchors the vehicle heading (yaw) on straight tracks, eliminating
    micro-yaw drift integration without requiring physical magnetometer or IMU hardware.
    """
    def __init__(
        self,
        min_flow_magnitude_px: float = 1.0,
        max_flow_magnitude_px: float = 60.0,
        ransac_iterations: int = 100,
        inlier_distance_px: float = 3.0,
        min_inliers: int = 25
    ):
        self.min_flow_magnitude_px = min_flow_magnitude_px
        self.max_flow_magnitude_px = max_flow_magnitude_px
        self.ransac_iterations = ransac_iterations
        self.inlier_distance_px = inlier_distance_px
        self.min_inliers = min_inliers
        self.last_valid_heading: Optional[float] = None
        self.heading_history = []

    def estimate_foe(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask_bottom_ratio: float = 0.18
    ) -> Tuple[Optional[np.ndarray], Optional[float], float]:
        """
        Estimates the Focus of Expansion (FOE) [u_foe, v_foe] in pixel coordinates
        and the corresponding optical heading angle (radians) relative to camera principal axis.

        Returns:
            foe_coords: (2,) [u_foe, v_foe] or None
            heading_yaw_rad: optical heading angle in radians or None
            confidence: inlier ratio [0.0, 1.0]
        """
        if len(pts0) < self.min_inliers or len(pts1) < self.min_inliers:
            return None, None, 0.0

        # Filter out hood/bumper region
        H, W = intrinsics.height, intrinsics.width
        max_y = H * (1.0 - mask_bottom_ratio)
        valid_y = (pts0[:, 1] < max_y) & (pts1[:, 1] < max_y)
        p0 = pts0[valid_y]
        p1 = pts1[valid_y]

        if len(p0) < self.min_inliers:
            return None, None, 0.0

        # Motion vectors
        d = p1 - p0
        norms = np.linalg.norm(d, axis=1)
        valid_motion = (norms >= self.min_flow_magnitude_px) & (norms <= self.max_flow_magnitude_px)
        
        p0 = p0[valid_motion]
        p1 = p1[valid_motion]
        d = d[valid_motion]
        norms = norms[valid_motion]

        if len(p0) < self.min_inliers:
            return None, None, 0.0

        # Unit normal vectors to motion lines: n = [-dy, dx] / norm
        normals = np.column_stack([-d[:, 1], d[:, 0]]) / norms[:, None]
        # Line equation: n_x * x + n_y * y = c where c = n_x * x0 + n_y * y0
        c = np.sum(normals * p0, axis=1)

        # RANSAC for line intersection (Focus of Expansion)
        best_inliers = 0
        best_foe = None
        num_pts = len(p0)

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(num_pts, size=2, replace=False)
            A_sample = normals[idx]
            b_sample = c[idx]

            det = A_sample[0, 0] * A_sample[1, 1] - A_sample[0, 1] * A_sample[1, 0]
            if abs(det) < 1e-4:
                continue

            foe_candidate = np.linalg.solve(A_sample, b_sample)

            # Plausibility check: FOE should be within or near the image frame
            if not (-W * 0.5 <= foe_candidate[0] <= W * 1.5 and -H * 0.5 <= foe_candidate[1] <= H * 1.5):
                continue

            # Compute perpendicular distances of all lines to candidate point
            dist = np.abs(np.sum(normals * foe_candidate, axis=1) - c)
            inliers = (dist <= self.inlier_distance_px).sum()

            if inliers > best_inliers:
                best_inliers = inliers
                best_foe = foe_candidate

        if best_foe is None or best_inliers < self.min_inliers:
            return None, None, 0.0

        # Refine FOE on inliers via least squares
        dist = np.abs(np.sum(normals * best_foe, axis=1) - c)
        inlier_mask = dist <= self.inlier_distance_px
        A_inliers = normals[inlier_mask]
        b_inliers = c[inlier_mask]

        foe_refined, _, _, _ = np.linalg.lstsq(A_inliers, b_inliers, rcond=None)
        confidence = float(best_inliers) / float(num_pts)

        # Optical heading: angle of FOE relative to camera principal point
        dx = foe_refined[0] - intrinsics.cx
        heading_yaw_rad = math.atan2(dx, intrinsics.fx)

        # Smooth heading history
        if abs(heading_yaw_rad) < 0.25: # Within ~14 degrees of forward
            self.last_valid_heading = heading_yaw_rad
            self.heading_history.append(heading_yaw_rad)
            if len(self.heading_history) > 30:
                self.heading_history.pop(0)

        return foe_refined, heading_yaw_rad, confidence

    def get_smoothed_heading(self) -> Optional[float]:
        """Returns the median optical heading over recent inlier frames."""
        if not self.heading_history:
            return self.last_valid_heading
        return float(np.median(self.heading_history))
