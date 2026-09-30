import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core.geometry import SE3
from ..core.frame import Keyframe
from ..frontend.base import BaseTracker
from .vpr import VPREngine, VPRDatabase
from .verifier import LoopVerifier, LoopVerificationResult
from .pgo import PoseGraphOptimizer, PoseGraphEdge


@dataclass
class LoopEvent:
    """
    Event record generated when a validated loop closure is closed and optimized.
    """
    candidate_kf_id: int
    current_kf_id: int
    num_inliers_3d: int
    rmse_3d_m: float
    correction_norm_m: float
    elapsed_time_sec: float
    timestamp: float


class LoopClosureManager:
    """
    Module 4 Manager: Coordinates Place Recognition (DINOv2),
    Metric 3D-3D Geometric Verification, and Global Pose Graph Optimization.
    """
    def __init__(
        self,
        min_gap_distance_m: float = 45.0,
        covisibility_window_size: int = 15,
        min_gap_frames: int = 25,
        min_gap_time_sec: float = 40.0,
        min_vpr_similarity: float = 0.85,
        min_inliers_3d: int = 50,
        cooldown_keyframes: int = 4,
        device: Optional[str] = None,
        fp16: bool = True
    ):
        self.min_gap_distance_m = min_gap_distance_m
        self.covisibility_window_size = covisibility_window_size
        self.min_gap_frames = min_gap_frames
        self.min_gap_time_sec = min_gap_time_sec
        self.min_vpr_similarity = min_vpr_similarity
        self.cooldown_keyframes = cooldown_keyframes
        self.last_loop_kf_idx = -9999
        self.last_matched_cand_id: Optional[int] = None
        self.corridor_tracking_active: bool = False
        
        self.vpr_engine = VPREngine(device=device, fp16=fp16)
        self.vpr_db = VPRDatabase(vpr_engine=self.vpr_engine)
        self.verifier = LoopVerifier(min_inliers_3d=min_inliers_3d)
        self.pgo = PoseGraphOptimizer(device=device)

        self.keyframes_history: List[Keyframe] = []
        self.keyframes_map: Dict[int, Keyframe] = {}
        self.closed_loops: List[LoopEvent] = []

    def process_keyframe(
        self,
        keyframe: Keyframe,
        tracker: BaseTracker
    ) -> Optional[LoopEvent]:
        """
        Processes a newly created keyframe: adds to database, searches for loops,
        verifies candidates in metric 3D space, and executes global PGO.
        Features Continuous Return Covisibility Corridor for tight return trajectory locking.
        """
        t0 = time.perf_counter()
        kf_id = keyframe.frame_id
        
        # Compute speed-independent accumulated odometric arc-length
        if len(self.keyframes_history) > 0:
            prev_kf = self.keyframes_history[-1]
            step_dist = float(np.linalg.norm(keyframe.pose_wc.t - prev_kf.pose_wc.t))
            keyframe.accumulated_distance_m = prev_kf.accumulated_distance_m + step_dist
        else:
            keyframe.accumulated_distance_m = 0.0

        # 1. Add sequential odometry edge from previous keyframe
        if len(self.keyframes_history) > 0:
            prev_kf = self.keyframes_history[-1]
            T_curr_prev = keyframe.pose_cw @ prev_kf.pose_cw.inv()
            self.pgo.add_edge(PoseGraphEdge(
                kf_i=prev_kf.frame_id,
                kf_j=keyframe.frame_id,
                T_ji_measured=T_curr_prev,
                is_loop_closure=False,
                weight=2.0
            ))

        # 2. Extract DINOv2 descriptor and register in VPR database
        desc = self.vpr_db.add_keyframe(keyframe)
        self.keyframes_history.append(keyframe)
        self.keyframes_map[kf_id] = keyframe
        curr_idx = len(self.keyframes_history) - 1

        # Check cooldown (shorter cooldown if actively tracking along the return corridor)
        active_cooldown = 2 if self.corridor_tracking_active else self.cooldown_keyframes
        if (curr_idx - self.last_loop_kf_idx) < active_cooldown:
            return None

        # Build covisibility exclusion set (recent local sliding window + connected neighbors)
        covis_exclude = set()
        recent_window = self.keyframes_history[-self.covisibility_window_size:] if len(self.keyframes_history) > self.covisibility_window_size else self.keyframes_history
        for k in recent_window:
            covis_exclude.add(k.frame_id)
            covis_exclude.update(k.connected_keyframes.keys())

        # 3. Query place recognition candidates with speed-independent distance gating
        candidates = self.vpr_db.query(
            query_desc=desc,
            current_kf_id=kf_id,
            current_timestamp=keyframe.timestamp,
            current_accum_dist_m=keyframe.accumulated_distance_m,
            min_gap_distance_m=self.min_gap_distance_m,
            covisibility_exclude_ids=covis_exclude,
            min_gap_frames=self.min_gap_frames,
            min_gap_time_sec=self.min_gap_time_sec,
            top_k=4,
            min_similarity=self.min_vpr_similarity
        )

        # 3b. Continuous Return Corridor Covisibility Candidates
        corridor_candidates: List[Tuple[int, float]] = []
        if self.corridor_tracking_active and self.last_matched_cand_id is not None:
            # Expected reverse traversal: candidate keyframes move backwards along the outbound track
            for delta in [-1, -2, 0, -3, 1]:
                cand_k_id = self.last_matched_cand_id + delta
                if cand_k_id in self.keyframes_map and cand_k_id not in covis_exclude and abs(cand_k_id - kf_id) >= self.min_gap_frames:
                    if not any(c[0] == cand_k_id for c in candidates):
                        corridor_candidates.append((cand_k_id, 0.99)) # Priority candidate

        all_candidates = corridor_candidates + candidates
        if not all_candidates:
            return None

        # 4. Metric 3D-3D Geometric Verification
        best_loop_event: Optional[LoopEvent] = None
        for cand_id, sim in all_candidates:
            cand_kf = self.keyframes_map.get(cand_id)
            if cand_kf is None:
                continue

            ver_res = self.verifier.verify_loop(cand_kf, keyframe, tracker)
            if not ver_res.is_verified or ver_res.T_curr_cand is None:
                continue

            # 5. Add Loop Closure Edge to Pose Graph
            self.pgo.add_edge(PoseGraphEdge(
                kf_i=cand_id,
                kf_j=kf_id,
                T_ji_measured=ver_res.T_curr_cand,
                is_loop_closure=True,
                weight=25.0
            ))

            # 6. Execute Global Pose Graph Optimization (PGO)
            all_kf_ids = [k.frame_id for k in self.keyframes_history]
            initial_poses = {k.frame_id: k.pose_cw for k in self.keyframes_history}
            
            old_pos = keyframe.pose_wc.t.copy()
            optimized_poses = self.pgo.optimize_loop(
                all_kf_ids, initial_poses, cand_id, kf_id, ver_res.T_curr_cand
            )

            # Update all keyframe poses across the trajectory
            for k_id, opt_pose in optimized_poses.items():
                if k_id in self.keyframes_map:
                    self.keyframes_map[k_id].frame.pose_cw = opt_pose

            new_pos = keyframe.pose_wc.t
            correction_dist = float(np.linalg.norm(new_pos - old_pos))

            elapsed = time.perf_counter() - t0
            best_loop_event = LoopEvent(
                candidate_kf_id=cand_id,
                current_kf_id=kf_id,
                num_inliers_3d=ver_res.num_inliers_3d,
                rmse_3d_m=ver_res.rmse_3d_meters,
                correction_norm_m=correction_dist,
                elapsed_time_sec=elapsed,
                timestamp=keyframe.timestamp
            )
            self.closed_loops.append(best_loop_event)
            self.last_loop_kf_idx = curr_idx
            self.last_matched_cand_id = cand_id
            self.corridor_tracking_active = True
            break # Accepted top verified loop

        return best_loop_event

    def finalize_global_optimization(self, enable_crawler_constraints: bool = True) -> Dict[int, SE3]:
        """
        Runs full global covisibility pose graph BA over all keyframes across the entire run.
        Simultaneously satisfies all odometry constraints, loop closures, and crawler kinematics.
        """
        all_kf_ids = [k.frame_id for k in self.keyframes_history]
        if len(all_kf_ids) < 4 or len(self.pgo.edges) == 0 or len(self.closed_loops) == 0:
            return {k.frame_id: k.pose_cw for k in self.keyframes_history}

        initial_poses = {k.frame_id: k.pose_cw for k in self.keyframes_history}
        optimized_poses = self.pgo.optimize(
            all_kf_ids,
            initial_poses,
            enable_crawler_constraints=enable_crawler_constraints
        )

        for k_id, opt_pose in optimized_poses.items():
            if k_id in self.keyframes_map:
                self.keyframes_map[k_id].frame.pose_cw = opt_pose

        return optimized_poses
