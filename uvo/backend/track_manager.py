from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core.geometry import SE3, project_points
from ..core.frame import Keyframe
from .base import VisualTrack, VisualObservation


class TrackManager:
    """
    Manages multi-view visual landmark tracks across keyframes for Bundle Adjustment.
    """
    def __init__(self, max_tracks: int = 500, min_observations: int = 2):
        self.max_tracks = max_tracks
        self.min_observations = min_observations
        self.tracks: Dict[int, VisualTrack] = {}
        self.next_track_id: int = 1

    def add_keyframe_observations(
        self,
        keyframe: Keyframe,
        matched_pts_prev: Optional[np.ndarray] = None,
        matched_pts_curr: Optional[np.ndarray] = None,
        prev_kf_id: Optional[int] = None
    ):
        """
        Adds 2D observations and metric 3D depth priors from a newly created Keyframe.
        """
        kpts = keyframe.frame.keypoints
        if kpts is None or len(kpts) == 0:
            return

        # Extract metric 3D points from keyframe if depth map exists
        pts_3d_cam = keyframe.extract_metric_points(kpts) if keyframe.depth_map is not None else None
        
        # Transform points from camera frame to world frame: P_world = R_wc * P_cam + t_wc
        T_wc = keyframe.pose_wc
        pts_3d_world = (T_wc.R @ pts_3d_cam.T).T + T_wc.t if (pts_3d_cam is not None and len(pts_3d_cam) > 0) else None

        # Create or update tracks
        # If matches with previous keyframe provided, link them
        num_kpts = len(kpts)
        for i in range(num_kpts):
            if len(self.tracks) >= self.max_tracks:
                # Remove stale tracks that have only 1 observation
                stale_ids = [tid for tid, tr in self.tracks.items() if len(tr.observations) < 2]
                for sid in stale_ids[:50]:
                    del self.tracks[sid]

            # Get metric depth prior if valid
            d_prior = float(pts_3d_cam[i, 2]) if (pts_3d_cam is not None and i < len(pts_3d_cam)) else None
            if d_prior is not None and (d_prior <= 0.2 or d_prior >= 80.0):
                d_prior = None

            obs = VisualObservation(
                keyframe_id=keyframe.frame_id,
                pixel_uv=kpts[i],
                depth_prior=d_prior,
                depth_weight=1.0
            )

            p_world = pts_3d_world[i] if (pts_3d_world is not None and i < len(pts_3d_world)) else None

            # New track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            tr = VisualTrack(track_id=track_id, point_3d_world=p_world)
            tr.add_observation(obs)
            self.tracks[track_id] = tr

    def get_active_tracks(self, active_kf_ids: List[int]) -> List[VisualTrack]:
        """
        Returns all tracks with at least min_observations in the active keyframe window.
        """
        active_set = set(active_kf_ids)
        result = []
        for tr in self.tracks.values():
            if tr.is_outlier or tr.point_3d_world is None:
                continue
            obs_in_window = [o for o in tr.observations if o.keyframe_id in active_set]
            if len(obs_in_window) >= self.min_observations or any(o.depth_prior is not None for o in obs_in_window):
                result.append(tr)
        return result
