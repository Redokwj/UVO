import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from ..core.geometry import SE3
from ..core.frame import Keyframe
from ..frontend.base import BaseTracker


@dataclass
class LoopVerificationResult:
    """
    Result of metric 3D-3D geometric loop verification.
    """
    is_verified: bool
    candidate_kf_id: int
    current_kf_id: int
    T_curr_cand: Optional[SE3] = None      # Rigid relative transform: P_curr = T_curr_cand * P_cand
    num_matches: int = 0
    num_inliers_3d: int = 0
    inlier_ratio_3d: float = 0.0
    scale_consistency: float = 1.0        # Ratio should be ~1.0 since both frames have metric depth
    rmse_3d_meters: float = 0.0


class LoopVerifier:
    """
    Metric 3D-3D Geometric Loop Verifier.
    Leverages physical 3D point clouds from UniDepth V2 to perform Umeyama RANSAC,
    preventing perceptual aliasing and false positive loops in natural terrain.
    """
    def __init__(
        self,
        min_inliers_3d: int = 50,
        max_inlier_dist_m: float = 0.45,
        max_scale_drift_pct: float = 18.0,
        ransac_iterations: int = 200
    ):
        self.min_inliers_3d = min_inliers_3d
        self.max_inlier_dist_m = max_inlier_dist_m
        self.max_scale_drift_pct = max_scale_drift_pct
        self.ransac_iterations = ransac_iterations

    def verify_loop(
        self,
        candidate_kf: Keyframe,
        current_kf: Keyframe,
        tracker: BaseTracker
    ) -> LoopVerificationResult:
        """
        Verifies loop hypothesis between candidate and current keyframe using 3D-3D RANSAC.
        """
        fail_res = LoopVerificationResult(
            is_verified=False,
            candidate_kf_id=candidate_kf.frame_id,
            current_kf_id=current_kf.frame_id
        )

        if candidate_kf.depth_map is None or current_kf.depth_map is None:
            return fail_res

        # 1. Feature matching via Front-End Tracker (XFeat for wide baseline)
        if hasattr(tracker, "xfeat"):
            track_res = tracker.xfeat.track(candidate_kf.frame, current_kf.frame)
        else:
            track_res = tracker.track(candidate_kf.frame, current_kf.frame)
        if track_res.num_matches < self.min_inliers_3d:
            return fail_res

        pts_cand_2d = track_res.matched_kpts0
        pts_curr_2d = track_res.matched_kpts1

        # 2. Extract metric 3D coordinates from UniDepth V2 depth maps
        pts_cand_3d = candidate_kf.extract_metric_points(pts_cand_2d)
        pts_curr_3d = current_kf.extract_metric_points(pts_curr_2d)

        if len(pts_cand_3d) < self.min_inliers_3d:
            return fail_res

        # Filter out points outside reliable depth bounds (0.3m to 60.0m)
        valid_mask = (
            (pts_cand_3d[:, 2] > 0.3) & (pts_cand_3d[:, 2] < 60.0) &
            (pts_curr_3d[:, 2] > 0.3) & (pts_curr_3d[:, 2] < 60.0)
        )
        
        P_cand = pts_cand_3d[valid_mask]
        P_curr = pts_curr_3d[valid_mask]

        num_pairs = len(P_cand)
        if num_pairs < self.min_inliers_3d:
            return fail_res

        # 3. 3D-3D RANSAC Umeyama Point Set Registration
        best_inliers_mask = None
        best_inliers_count = 0
        best_R = None
        best_t = None
        best_scale = 1.0

        np.random.seed(42)
        for _ in range(self.ransac_iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(num_pairs, 3, replace=False)
            P_samp = P_cand[sample_idx]
            Q_samp = P_curr[sample_idx]

            R_cand, t_cand, s_cand = self._solve_umeyama(P_samp, Q_samp, estimate_scale=True)
            if R_cand is None:
                continue

            # Check scale: since both depth maps are metric, scale must be close to 1.0
            scale_err_pct = abs(1.0 - s_cand) * 100.0
            if scale_err_pct > self.max_scale_drift_pct:
                continue

            # Count inliers in physical 3D space: P_proj = R * P_cand + t
            # Using rigid model (scale=1.0) to enforce true metric physics!
            P_transformed = (R_cand @ P_cand.T).T + t_cand
            dists_3d = np.linalg.norm(P_transformed - P_curr, axis=1)

            inliers = dists_3d < self.max_inlier_dist_m
            count = int(inliers.sum())

            if count > best_inliers_count:
                best_inliers_count = count
                best_inliers_mask = inliers
                best_R = R_cand
                best_t = t_cand
                best_scale = s_cand

        # 4. Verification Check
        if best_inliers_count < self.min_inliers_3d or best_inliers_mask is None:
            return fail_res

        # Re-fit final rigid SE(3) transform on all inliers
        P_in = P_cand[best_inliers_mask]
        Q_in = P_curr[best_inliers_mask]
        R_final, t_final, s_final = self._solve_umeyama(P_in, Q_in, estimate_scale=False)

        if R_final is None:
            return fail_res

        # Compute 3D RMSE
        P_final_trans = (R_final @ P_in.T).T + t_final
        rmse_3d = float(np.sqrt(np.mean(np.sum((P_final_trans - Q_in) ** 2, axis=1))))
        inlier_ratio = float(best_inliers_count / num_pairs * 100.0)

        T_curr_cand = SE3(R=R_final, t=t_final)

        return LoopVerificationResult(
            is_verified=True,
            candidate_kf_id=candidate_kf.frame_id,
            current_kf_id=current_kf.frame_id,
            T_curr_cand=T_curr_cand,
            num_matches=num_pairs,
            num_inliers_3d=best_inliers_count,
            inlier_ratio_3d=inlier_ratio,
            scale_consistency=float(s_final),
            rmse_3d_meters=rmse_3d
        )

    @staticmethod
    def _solve_umeyama(
        P: np.ndarray,
        Q: np.ndarray,
        estimate_scale: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Solves Umeyama 3D-3D point registration: Q = s * R * P + t
        """
        n = P.shape[0]
        if n < 3:
            return None, None, 1.0

        mu_p = np.mean(P, axis=0)
        mu_q = np.mean(Q, axis=0)

        P_cent = P - mu_p
        Q_cent = Q - mu_q

        sigma_p = np.sum(P_cent ** 2) / n
        if sigma_p < 1e-8:
            return None, None, 1.0

        H = (P_cent.T @ Q_cent) / n
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return None, None, 1.0

        V = Vt.T
        det_sign = np.linalg.det(V @ U.T)
        S_diag = np.diag([1.0, 1.0, 1.0 if det_sign > 0 else -1.0])

        R = V @ S_diag @ U.T
        scale = 1.0
        if estimate_scale:
            scale = float(np.trace(S_diag @ np.diag(S)) / sigma_p)

        t = mu_q - scale * (R @ mu_p)
        return R, t, scale
