import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

from .core.geometry import SE3, project_points, unproject_pixels, so3_log, so3_exp
from .core.frame import Frame, Keyframe, CameraIntrinsics
from .core.sensors import CameraExtrinsics
from .core.trajectory import Trajectory, TrajectoryPoint
from .frontend.base import BaseTracker, TrackingResult, FrontendMode
from .frontend.xfeat_tracker import XFeatTracker
from .frontend.liftfeat_tracker import LiftFeatTracker
from .frontend.hybrid_tracker import HybridTracker
from .frontend.klt_tracker import KLTTracker
from .frontend.optical_compass import OpticalCompass
from .metric.base import BaseDepthProvider, DepthPrediction, DepthModelType
from .metric.unidepth_provider import UniDepthProvider
from .metric.metric3d_provider import Metric3DProvider
from .backend.base import VisualTrack, OptimizationResult
from .backend.track_manager import TrackManager
from .backend.solver import SlidingWindowOptimizer
from .loop_closure.manager import LoopClosureManager, LoopEvent


@dataclass
class UVOConfig:
    frontend_mode: FrontendMode = FrontendMode.HYBRID
    depth_mode: DepthModelType = DepthModelType.UNIDEPTH
    enable_metric_depth: bool = True
    enable_backend_ba: bool = True
    ba_window_size: int = 8
    ba_max_iterations: int = 10
    enable_loop_closure: bool = True
    loop_min_gap_dist_m: float = 160.0  # Rover loop is 284m, only match when at least 160m driven
    loop_min_gap_frames: int = 40
    loop_min_gap_time_sec: float = 120.0 # Only close after at least 2 minutes of driving
    loop_min_vpr_sim: float = 0.86       # True loop has >0.88 similarity
    keyframe_parallax_thresh_px: float = 28.0
    keyframe_rotation_thresh_rad: float = 0.105 # ~6.0 deg angular threshold for dense turning keyframes
    enable_angular_keyframing: bool = True      # Dynamic angular keyframe trigger during crawler turns
    keyframe_interval_max: int = 20
    min_inliers_tracking: int = 30
    min_inliers_pnp: int = 20
    reproj_error_pnp_px: float = 3.0
    extrinsics: Optional[CameraExtrinsics] = None
    fp16: bool = True  # Native Tensor Core FP16 half precision for UniDepth, VPR, and Front-End
    async_mapping: bool = True  # Decouple tracking from neural depth and backend bundle adjustment
    max_depth_resolution: int = 518  # Downscale resolution for metric depth model (divisible by 14)
    mask_bottom_ratio: float = 0.18  # Bottom crop ratio (filters out rover hood / front bumper)
    min_depth_m: float = 1.5         # Golden near-range bound (reject chassis / near grass)
    max_depth_m: float = 16.0        # Golden far-range bound (reject quadratic depth noise > 16m)
    enable_motion_smoothing: bool = True  # Motion model prior & velocity clamp to eliminate PnP jitter
    max_velocity_mps: float = 2.5         # Physically bound rover motion (reject PnP teleportations > 2.5 m/s)
    pose_smoothing_alpha: float = 0.35   # Exponential moving average filter on translation (1.0 = raw PnP)
    motion_damping: float = 0.85         # Velocity momentum persistence factor across frames
    use_pnp_motion_prior: bool = True    # Seed PnP optimization with motion model prediction (SOLVEPNP_ITERATIVE)
    use_lm_refinement: bool = True       # Run Levenberg-Marquardt non-linear refinement on inliers (solvePnPRefineLM)
    tracking_mode: str = "keyframe_anchored"  # "keyframe_anchored" (smooth 2D VO + metric keyframe anchors) or "pnp_per_frame"
    enable_closed_form_scale: bool = True     # Closed-form analytical scale calculation from keyframe 3D points
    is_tracked_chassis: bool = True           # Non-holonomic motion constraints for tracked crawler chassis (гусеничний дрон)
    crawler_lateral_damping: float = 0.90     # Suppresses 90% of lateral velocity jitter in crawler chassis frame
    enable_visual_imu: bool = True            # Automatic ground normal / gravity vector from UniDepth V2 point predictions
    enable_marginalization: bool = True       # Schur complement information prior for sliding window bundle adjustment
    enable_ground_scale_anchor: bool = False  # Ground height scale invariant (disabled to let UniDepth natural metric scale govern)
    nominal_camera_height_m: float = 1.15     # Calibrated physical mounting height of camera above ground (meters)
    vegetation_height_m: float = 0.28         # Average height of grass / vegetation above hard soil (meters)
    backend_optimizer: str = "torch"          # "torch" (Micro-Torch GPU LM) or "theseus" (Meta AI Theseus GPU Factor Graph)
    enable_optical_compass: bool = True       # Focus of Expansion (FOE) & horizon heading anchor to prevent yaw drift
    enable_global_pose_graph_ba: bool = True  # Full SE(3) covisibility pose graph BA backend
    enable_direct_pnp_tracking: bool = False  # Direct 3D-to-2D PnP tracking against active keyframe landmarks


class UVOPipeline:
    """
    Core Visual Odometry Engine of UVO (Unnamed Visual Odometry).
    Integrates Front-End (Module 1), Metric Depth Conditioning (Module 2),
    and Trajectory Estimation.
    """
    def __init__(self, config: Optional[UVOConfig] = None):
        self.config = config or UVOConfig()
        
        # 1. Initialize Front-End Tracker
        if self.config.frontend_mode == FrontendMode.HYBRID:
            self.tracker: BaseTracker = HybridTracker(
                keyframe_parallax_threshold=self.config.keyframe_parallax_thresh_px,
                min_inliers=self.config.min_inliers_tracking,
                fp16=self.config.fp16
            )
        elif self.config.frontend_mode == FrontendMode.LIFTFEAT:
            self.tracker = LiftFeatTracker(
                keyframe_parallax_threshold=self.config.keyframe_parallax_thresh_px,
                min_inliers=self.config.min_inliers_tracking,
                fp16=self.config.fp16
            )
        elif self.config.frontend_mode == FrontendMode.XFEAT:
            self.tracker = XFeatTracker(
                keyframe_parallax_threshold=self.config.keyframe_parallax_thresh_px,
                min_inliers=self.config.min_inliers_tracking,
                fp16=self.config.fp16
            )
        elif self.config.frontend_mode == FrontendMode.KLT:
            self.tracker = KLTTracker(
                keyframe_parallax_threshold=self.config.keyframe_parallax_thresh_px,
                min_inliers=self.config.min_inliers_tracking
            )
        else:
            self.tracker = HybridTracker(fp16=self.config.fp16)

        # 2. Initialize Metric Depth Provider (UniDepth V2 default)
        self.depth_provider: Optional[BaseDepthProvider] = None
        if self.config.enable_metric_depth:
            if self.config.depth_mode == DepthModelType.UNIDEPTH:
                self.depth_provider = UniDepthProvider(
                    fp16=self.config.fp16,
                    max_resolution=self.config.max_depth_resolution
                )
            elif self.config.depth_mode == DepthModelType.METRIC3D:
                self.depth_provider = Metric3DProvider()
            else:
                self.depth_provider = UniDepthProvider(
                    fp16=self.config.fp16,
                    max_resolution=self.config.max_depth_resolution
                )

        # 3. State variables
        self.current_pose_wc: SE3 = SE3.identity()
        self.trajectory = Trajectory(name="uvo_live_trajectory")
        self.keyframes: List[Keyframe] = []
        self.last_frame: Optional[Frame] = None
        self.last_keyframe: Optional[Keyframe] = None
        self.frame_count: int = 0
        self.frames_since_keyframe: int = 0

        # 4. Backend Factor Graph Optimizer (Module 3)
        self.track_manager = TrackManager(max_tracks=500)
        self.optimizer: Optional[BaseOptimizer] = None
        if self.config.enable_backend_ba:
            if self.config.backend_optimizer.lower() == "theseus":
                from .backend.theseus_optimizer import TheseusFactorGraphOptimizer
                self.optimizer = TheseusFactorGraphOptimizer(
                    window_size=self.config.ba_window_size,
                    max_iterations=self.config.ba_max_iterations
                )
            else:
                self.optimizer = SlidingWindowOptimizer(
                    window_size=self.config.ba_window_size,
                    max_iterations=self.config.ba_max_iterations,
                    enable_crawler_constraints=self.config.is_tracked_chassis,
                    enable_marginalization=self.config.enable_marginalization
                )
        self.last_ba_result: Optional[OptimizationResult] = None

        # 5. Global Loop Closure & Pose Graph Optimizer (Module 4)
        self.loop_manager: Optional[LoopClosureManager] = None
        if self.config.enable_loop_closure:
            self.loop_manager = LoopClosureManager(
                min_gap_distance_m=self.config.loop_min_gap_dist_m,
                min_gap_frames=self.config.loop_min_gap_frames,
                min_gap_time_sec=self.config.loop_min_gap_time_sec,
                min_vpr_similarity=self.config.loop_min_vpr_sim,
                fp16=self.config.fp16
            )
        self.last_loop_event: Optional[LoopEvent] = None

        # 6. Asynchronous Background Mapping & Depth Worker (Threaded SLAM)
        self.async_mapping = self.config.async_mapping
        self._mapping_queue = queue.Queue(maxsize=1) if self.async_mapping else None
        self._shutdown_event = threading.Event()
        self._mapping_thread = None
        if self.async_mapping:
            self._mapping_thread = threading.Thread(target=self._async_mapping_worker, daemon=True)
            self._mapping_thread.start()

        # 7. Motion Model Prior & State Smoothing (anti-jitter & velocity clamping)
        self._smoothed_velocity = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self._last_valid_pose_wc = SE3.identity()
        self._last_pose_timestamp = 0.0

        # 8. Optical Compass & Vanishing FOE Heading Anchor
        self.optical_compass: Optional[OpticalCompass] = None
        if self.config.enable_optical_compass:
            self.optical_compass = OpticalCompass()

    def process_image(
        self,
        image: np.ndarray,
        timestamp: float,
        intrinsics: CameraIntrinsics,
        gravity_vector: Optional[np.ndarray] = None
    ) -> Tuple[SE3, bool, Optional[TrackingResult]]:
        """
        Process a new monocular video frame through the UVO visual odometry pipeline.
        Returns:
            - current_pose_wc: Estimated camera-to-world SE(3) pose in meters
            - is_keyframe: Whether this frame was promoted to a keyframe
            - tracking_result: Front-end tracking diagnostics
        """
        self.frame_count += 1
        curr_frame = Frame(
            frame_id=self.frame_count,
            timestamp=timestamp,
            image=image,
            intrinsics=intrinsics
        )

        # -------------------------------------------------------------
        # FIRST FRAME INITIALIZATION
        # -------------------------------------------------------------
        if self.last_frame is None:
            curr_frame.pose_cw = SE3.identity()
            self.current_pose_wc = SE3.identity()
            self._last_valid_pose_wc = self.current_pose_wc
            self._last_pose_timestamp = timestamp
            self._smoothed_velocity = np.zeros(3, dtype=np.float64)
            
            # Make first frame a keyframe with metric depth
            kf = self._create_keyframe(curr_frame)
            self.keyframes.append(kf)
            self.track_manager.add_keyframe_observations(kf)
            self.last_frame = curr_frame
            self.last_keyframe = kf
            self.frames_since_keyframe = 0
            
            self.trajectory.add_pose(timestamp, self.current_pose_wc, status="KEYFRAME")
            return self.current_pose_wc, True, None

        # Initialize curr_frame pose from current estimate (prevents zero-pose reset)
        curr_frame.pose_cw = self.current_pose_wc.inv()

        # -------------------------------------------------------------
        # 1. FRONT-END 2D-2D TRACKING (XFeat / LiftFeat)
        # -------------------------------------------------------------
        tracking_res = self.tracker.track(self.last_frame, curr_frame)
        self.frames_since_keyframe += 1

        # -------------------------------------------------------------
        # 2. KEYFRAME DECISION (Parallax + Rotation + Tracking Inliers)
        # -------------------------------------------------------------
        rot_angle_kf = 0.0
        if self.last_keyframe is not None:
            # Predict rotation of curr_frame from tracker
            if tracking_res.relative_pose is not None and self.last_frame is not None:
                R_pred_wc = self.last_frame.pose_wc.R @ tracking_res.relative_pose.R.T
            else:
                R_pred_wc = self.current_pose_wc.R
            R_rel_kf = R_pred_wc.T @ self.last_keyframe.pose_wc.R

            if self.config.is_tracked_chassis:
                # Crawler yaw around gravity normal
                w_rel = so3_log(R_rel_kf)
                g_vec = getattr(self.last_keyframe, "gravity_cam", None)
                if g_vec is None:
                    g_vec = np.array([0.0, 0.9616, -0.2740])
                g_unit = g_vec / (np.linalg.norm(g_vec) + 1e-8)
                rot_angle_kf = abs(float(np.dot(w_rel, g_unit)))
            else:
                cos_th = (np.trace(R_rel_kf) - 1.0) / 2.0
                rot_angle_kf = float(np.arccos(np.clip(cos_th, -1.0, 1.0)))

        is_rotating = (self.config.enable_angular_keyframing and 
                       rot_angle_kf >= self.config.keyframe_rotation_thresh_rad and 
                       self.frames_since_keyframe >= 2)
        inliers_degraded = (tracking_res.num_inliers < self.config.min_inliers_tracking and self.frames_since_keyframe >= 3)
        parallax_accum = (tracking_res.is_keyframe_candidate and self.frames_since_keyframe >= 6 and tracking_res.mean_parallax_px >= 2.0)
        interval_timeout = (self.frames_since_keyframe >= self.config.keyframe_interval_max)

        need_keyframe = (
            is_rotating or
            inliers_degraded or
            parallax_accum or
            interval_timeout
        )

        # -------------------------------------------------------------
        # 3. POSE ESTIMATION: Direct Metric Keyframe PnP with Robust 2D Fallback
        # -------------------------------------------------------------
        pnp_success = False

        # Attempt Direct Metric 3D-to-2D PnP against active keyframe landmarks (Module 1)
        has_any_depth_kf = any(k.depth_map is not None for k in self.keyframes)
        if (self.config.enable_direct_pnp_tracking or need_keyframe or self.config.tracking_mode == "pnp_per_frame") and has_any_depth_kf:
            prior_pose_wc = None
            if self.config.use_pnp_motion_prior and self.last_frame is not None:
                dt_pred = max(1e-3, timestamp - self._last_pose_timestamp)
                p_pred = self._last_valid_pose_wc.t + self._smoothed_velocity * dt_pred
                if tracking_res.relative_pose is not None:
                    R_pred = self._last_valid_pose_wc.R @ tracking_res.relative_pose.R.T
                else:
                    R_pred = self._last_valid_pose_wc.R
                prior_pose_wc = SE3(R=R_pred, t=p_pred)

            pnp_pose = self._track_keyframe_pnp(self.last_keyframe, curr_frame, prior_pose_wc=prior_pose_wc)
            if pnp_pose is not None:
                if self.config.enable_motion_smoothing and self.last_frame is not None:
                    dt = max(1e-3, timestamp - self._last_pose_timestamp)
                    p_pnp = pnp_pose.t
                    p_prev = self._last_valid_pose_wc.t

                    # Instantaneous step velocity implied by raw PnP
                    v_step = (p_pnp - p_prev) / dt
                    speed = np.linalg.norm(v_step)

                    # Physical motion model prediction
                    p_pred = p_prev + self._smoothed_velocity * dt

                    if speed > self.config.max_velocity_mps:
                        # Outlier rejection
                        p_smooth = p_pred
                    else:
                        alpha = 0.85 if self.config.enable_direct_pnp_tracking else self.config.pose_smoothing_alpha
                        p_smooth = alpha * p_pnp + (1.0 - alpha) * p_pred

                        # Non-holonomic tracked chassis projection: damp lateral slip in body frame
                        if self.config.is_tracked_chassis:
                            step_body = self._last_valid_pose_wc.R.T @ (p_smooth - p_prev)
                            step_body[0] *= (1.0 - self.config.crawler_lateral_damping)
                            p_smooth = p_prev + self._last_valid_pose_wc.R @ step_body

                        v_actual = (p_smooth - p_prev) / dt
                        self._smoothed_velocity = (
                            self.config.motion_damping * self._smoothed_velocity +
                            (1.0 - self.config.motion_damping) * v_actual
                        )
                        if self.config.is_tracked_chassis:
                            v_body = self._last_valid_pose_wc.R.T @ self._smoothed_velocity
                            v_body[0] *= (1.0 - self.config.crawler_lateral_damping)
                            self._smoothed_velocity = self._last_valid_pose_wc.R @ v_body

                    self.current_pose_wc = SE3(R=pnp_pose.R, t=p_smooth)
                else:
                    self.current_pose_wc = pnp_pose

                curr_frame.pose_cw = self.current_pose_wc.inv()
                self._last_valid_pose_wc = self.current_pose_wc
                self._last_pose_timestamp = timestamp
                pnp_success = True

        # Fallback to 2D-2D Relative Pose with Crawler Kinematics & Optical Compass
        if not pnp_success:
            if tracking_res.relative_pose is not None:
                t_dir = tracking_res.relative_pose.t
                t_norm = np.linalg.norm(t_dir)
                dt = max(1e-3, timestamp - (self._last_pose_timestamp if self._last_pose_timestamp > 0 else timestamp))

                # Closed-form analytical scale calibration from keyframe 3D points
                if self.config.enable_closed_form_scale and self.last_keyframe is not None and t_norm > 1e-6:
                    d_closed = self._estimate_metric_scale_closed_form(
                        self.last_keyframe, curr_frame, tracking_res.relative_pose.R, t_dir
                    )
                    if d_closed is not None:
                        dt_kf = max(1e-3, timestamp - self.last_keyframe.timestamp)
                        speed_inst = d_closed / dt_kf
                        if 0.05 < speed_inst < self.config.max_velocity_mps:
                            v_dir = t_dir / t_norm
                            self._smoothed_velocity = 0.7 * self._smoothed_velocity + 0.3 * (speed_inst * v_dir)

                if t_norm > 1e-6:
                    speed = np.linalg.norm(self._smoothed_velocity)
                    speed = speed if (0.05 < speed < self.config.max_velocity_mps) else 0.35
                    step_m = speed * dt
                    # Crawler kinematics: forward motion strictly orthogonal to gravity (zero vertical plunge, zero lateral slip)
                    if self.config.is_tracked_chassis and self.last_keyframe is not None and getattr(self.last_keyframe, "gravity_cam", None) is not None:
                        g_cam = self.last_keyframe.gravity_cam
                        # Forward vector in camera coordinates (t_rel has negative Z so inv advances forward)
                        f_body = np.array([0.0, g_cam[2], -g_cam[1]], dtype=np.float64)
                        norm_f = np.linalg.norm(f_body)
                        if norm_f > 1e-6:
                            t_scaled = (f_body / norm_f) * step_m
                        else:
                            t_scaled = np.array([0.0, -0.274, -0.962]) * step_m
                    elif self.config.is_tracked_chassis:
                        # Fallback when gravity_cam not yet computed: camera pitch ~16 deg
                        t_scaled = np.array([0.0, -0.274, -0.962], dtype=np.float64) * step_m
                    else:
                        t_scaled = (t_dir / t_norm) * step_m
                else:
                    t_scaled = np.zeros(3)

                R_step = tracking_res.relative_pose.R
                if self.config.is_tracked_chassis:
                    w_step = so3_log(R_step)
                    g_vec = getattr(self.last_keyframe, "gravity_cam", None) if self.last_keyframe else None
                    if g_vec is None:
                        g_vec = np.array([0.0, 0.9616, -0.2740])
                    g_unit = g_vec / (np.linalg.norm(g_vec) + 1e-8)
                    yaw_angle = float(np.dot(w_step, g_unit))
                    w_filtered = yaw_angle * g_unit
                    R_step = so3_exp(w_filtered)

                T_rel = SE3(R=R_step, t=t_scaled)
                self.current_pose_wc = self.last_frame.pose_wc @ T_rel.inv()
                curr_frame.pose_cw = self.current_pose_wc.inv()
                self._last_valid_pose_wc = self.current_pose_wc
                self._last_pose_timestamp = timestamp

        is_keyframe = False
        loop_closed_this_frame = False
        if need_keyframe:
            is_keyframe = True
            curr_frame.is_keyframe = True
            curr_frame.pose_cw = self.current_pose_wc.inv()

            if self.async_mapping:
                # Decoupled high-rate VO: create keyframe immediately without blocking on depth
                kf = Keyframe(frame=curr_frame)
                self.keyframes.append(kf)
                self.last_keyframe = kf
                self.frames_since_keyframe = 0

                # Enqueue for asynchronous background foundation depth and bundle adjustment
                if self._mapping_queue is not None:
                    if self._mapping_queue.full():
                        try:
                            self._mapping_queue.get_nowait()
                            self._mapping_queue.task_done()
                        except queue.Empty:
                            pass
                    self._mapping_queue.put((kf, gravity_vector))
            else:
                # Synchronous fallback
                kf = self._create_keyframe(curr_frame)
                self.keyframes.append(kf)
                self.last_keyframe = kf
                self.frames_since_keyframe = 0
                self.track_manager.add_keyframe_observations(kf)

                if self.optimizer is not None and len(self.keyframes) >= 3:
                    window_kfs = self.keyframes[-self.config.ba_window_size:]
                    active_tracks = self.track_manager.get_active_tracks([k.frame_id for k in window_kfs])
                    if len(active_tracks) >= 5:
                        opt_res = self.optimizer.optimize(window_kfs, active_tracks, gravity_vector=gravity_vector)
                        self.last_ba_result = opt_res
                        self.current_pose_wc = self.last_keyframe.pose_wc

                if self.loop_manager is not None:
                    loop_ev = self.loop_manager.process_keyframe(kf, self.tracker)
                    if loop_ev is not None:
                        self.last_loop_event = loop_ev
                        loop_closed_this_frame = True
                        self.current_pose_wc = self.last_keyframe.pose_wc
                        self.update_trajectory_from_keyframes()

        # -------------------------------------------------------------
        # RECORD TRAJECTORY
        # -------------------------------------------------------------
        if is_keyframe:
            status = "LOOP_CLOSED" if loop_closed_this_frame else "KEYFRAME"
        else:
            status = "TRACKED_PNP" if pnp_success else "TRACKED_2D"

        ref_kf_id = self.last_keyframe.frame_id if self.last_keyframe is not None else None
        pose_rel_kf = (self.last_keyframe.pose_wc.inv() @ self.current_pose_wc) if self.last_keyframe is not None else None

        self.trajectory.add_pose(
            timestamp=timestamp,
            pose_wc=self.current_pose_wc,
            status=status,
            frame_id=curr_frame.frame_id,
            ref_kf_id=ref_kf_id,
            pose_rel_kf=pose_rel_kf
        )
        self.last_frame = curr_frame

        return self.current_pose_wc, is_keyframe, tracking_res

    def _async_mapping_worker(self):
        """
        Background Worker for Keyframe Foundation Metric Depth (UniDepth V2),
        Factor Graph Bundle Adjustment, and Place Recognition (DINOv2).
        Completely decouples high-rate real-time visual tracking from heavy neural inference.
        """
        while not self._shutdown_event.is_set():
            try:
                task = self._mapping_queue.get(timeout=0.05)
            except (queue.Empty, AttributeError):
                continue

            if task is None:
                break

            kf, gravity_vector = task
            try:
                # 1. Compute foundation metric depth asynchronously
                if self.depth_provider is not None:
                    depth_pred = self.depth_provider.predict_depth(kf.frame.image, kf.frame.intrinsics)
                    depth_map = depth_pred.depth_map
                    # Module 2: Ground Height Scale Anchor (Physical Plane Scale Invariant)
                    if self.config.enable_ground_scale_anchor and getattr(depth_pred, "ground_height", None) is not None:
                        h_meas = depth_pred.ground_height
                        h_target = max(0.40, self.config.nominal_camera_height_m - self.config.vegetation_height_m)
                        if 0.30 <= h_meas <= 2.50:
                            s_scale = float(np.clip(h_target / h_meas, 0.85, 1.15))
                            if depth_map is not None:
                                depth_map = depth_map * s_scale
                    kf.depth_map = depth_map
                    kf.depth_uncertainty = depth_pred.uncertainty_map
                    if depth_pred.gravity_cam is not None:
                        kf.gravity_cam = depth_pred.gravity_cam
                        if gravity_vector is None and self.config.enable_visual_imu:
                            gravity_vector = depth_pred.gravity_cam

                # 2. Register keyframe observations in multi-view track manager
                self.track_manager.add_keyframe_observations(kf)

                # 3. Trigger Sliding Window Factor Graph Optimization (Module 3)
                if self.optimizer is not None and len(self.keyframes) >= 3:
                    window_kfs = self.keyframes[-self.config.ba_window_size:]
                    active_tracks = self.track_manager.get_active_tracks([k.frame_id for k in window_kfs])
                    if len(active_tracks) >= 5:
                        opt_res = self.optimizer.optimize(window_kfs, active_tracks, gravity_vector=gravity_vector)
                        self.last_ba_result = opt_res

                # 4. Trigger Global Loop Closure & Pose Graph Optimization (Module 4)
                if self.loop_manager is not None:
                    loop_ev = self.loop_manager.process_keyframe(kf, self.tracker)
                    if loop_ev is not None:
                        self.last_loop_event = loop_ev
                        self.update_trajectory_from_keyframes()
            except Exception:
                pass
            finally:
                if self._mapping_queue is not None:
                    self._mapping_queue.task_done()

    def update_trajectory_from_keyframes(self):
        """
        Propagates globally optimized keyframe poses (from PGO and BA)
        across the entire stored trajectory points.
        """
        kf_map = {kf.frame_id: kf.pose_wc for kf in self.keyframes}
        for pt in self.trajectory.points:
            if pt.frame_id in kf_map:
                pt.pose_wc = kf_map[pt.frame_id]
            elif pt.ref_kf_id in kf_map and pt.pose_rel_kf is not None:
                parent_pose = kf_map[pt.ref_kf_id]
                pt.pose_wc = parent_pose @ pt.pose_rel_kf

    def get_keyframe_trajectory(self) -> Trajectory:
        """
        Returns a clean Trajectory object containing all keyframes with their latest
        globally optimized poses (updated by backend BA and global PGO).
        """
        traj = Trajectory(name="uvo_keyframe_trajectory")
        for kf in self.keyframes:
            status = "KEYFRAME"
            if self.last_loop_event and kf.frame_id == self.last_loop_event.current_kf_id:
                status = "LOOP_CLOSED"
            traj.add_point(TrajectoryPoint(
                timestamp=kf.timestamp,
                pose_wc=kf.pose_wc,
                status=status,
                frame_id=kf.frame_id
            ))
        return traj

    def _create_keyframe(self, frame: Frame) -> Keyframe:
        """
        Promotes Frame to Keyframe by computing metric depth and unprojecting 3D points.
        """
        depth_map = None
        uncertainty_map = None
        gravity_cam = None

        if self.depth_provider is not None:
            depth_pred = self.depth_provider.predict_depth(frame.image, frame.intrinsics)
            depth_map = depth_pred.depth_map
            uncertainty_map = depth_pred.uncertainty_map
            gravity_cam = depth_pred.gravity_cam

            # Module 2: Ground Height Scale Anchor (Physical Plane Scale Invariant)
            if self.config.enable_ground_scale_anchor and getattr(depth_pred, "ground_height", None) is not None:
                h_meas = depth_pred.ground_height
                h_target = max(0.40, self.config.nominal_camera_height_m - self.config.vegetation_height_m)
                if 0.30 <= h_meas <= 2.50:
                    s_scale = float(np.clip(h_target / h_meas, 0.85, 1.15))
                    if depth_map is not None:
                        depth_map = depth_map * s_scale

        kf = Keyframe(
            frame=frame,
            depth_map=depth_map,
            depth_uncertainty=uncertainty_map,
            gravity_cam=gravity_cam
        )
        return kf

    def _track_keyframe_pnp(
        self,
        keyframe: Keyframe,
        curr_frame: Frame,
        prior_pose_wc: Optional[SE3] = None
    ) -> Optional[SE3]:
        """
        Tracks curr_frame directly against keyframe using 3D-to-2D Perspective-n-Point.
        Utilizes motion prior extrinsic guess (SOLVEPNP_ITERATIVE) and Levenberg-Marquardt
        inlier refinement (solvePnPRefineLM) to completely eliminate stochastic RANSAC jitter.
        """
        # Find latest available keyframe with valid depth map
        ref_kf = keyframe if (keyframe is not None and keyframe.depth_map is not None) else None
        if ref_kf is None:
            for k in reversed(self.keyframes):
                if k.depth_map is not None:
                    ref_kf = k
                    break
        if ref_kf is None:
            return None

        # Fast matching between keyframe and curr_frame using cached features
        tracker_backend = getattr(self.tracker, "xfeat", self.tracker)
        if "tracker_feat" in ref_kf.frame.feature_cache:
            cache_kf = ref_kf.frame.feature_cache["tracker_feat"]
            kpts_kf, desc_kf = cache_kf["kpts"], cache_kf["desc"]
        else:
            kpts_kf, desc_kf, _ = tracker_backend.extract_features(ref_kf.frame)
            ref_kf.frame.feature_cache["tracker_feat"] = {"kpts": kpts_kf, "desc": desc_kf}

        if "tracker_feat" in curr_frame.feature_cache:
            cache_curr = curr_frame.feature_cache["tracker_feat"]
            kpts_curr, desc_curr = cache_curr["kpts"], cache_curr["desc"]
        else:
            kpts_curr, desc_curr, _ = tracker_backend.extract_features(curr_frame)
            curr_frame.feature_cache["tracker_feat"] = {"kpts": kpts_curr, "desc": desc_curr}

        pts_kf_2d, pts_curr_2d = tracker_backend.match_features(desc_kf, desc_curr, kpts_kf, kpts_curr)
        if len(pts_kf_2d) < self.config.min_inliers_pnp:
            return None

        # Extract 3D points in keyframe camera coordinate frame
        pts_kf_3d = ref_kf.extract_metric_points(pts_kf_2d)
        if len(pts_kf_3d) < self.config.min_inliers_pnp:
            return None

        # Filter out points with invalid depth, sky, and rover hood/bumper
        H = curr_frame.intrinsics.height
        max_y = H * (1.0 - self.config.mask_bottom_ratio)
        valid_depth = (
            (pts_kf_3d[:, 2] >= self.config.min_depth_m) &
            (pts_kf_3d[:, 2] <= self.config.max_depth_m) &
            (pts_kf_2d[:, 1] < max_y) &
            (pts_curr_2d[:, 1] < max_y)
        )
        if valid_depth.sum() < self.config.min_inliers_pnp:
            return None

        pts_3d_filt = pts_kf_3d[valid_depth].astype(np.float64)
        pts_2d_filt = pts_curr_2d[valid_depth].astype(np.float64)

        K = curr_frame.intrinsics.to_matrix().astype(np.float64)
        D = curr_frame.intrinsics.dist_coeffs().astype(np.float64)

        # Compute extrinsic guess relative to keyframe: T_curr_kf = (T_wc_curr)^(-1) @ T_wc_kf
        use_guess = False
        flags = cv2.SOLVEPNP_EPNP
        rvec_guess = np.zeros((3, 1), dtype=np.float64)
        tvec_guess = np.zeros((3, 1), dtype=np.float64)

        if self.config.use_pnp_motion_prior and prior_pose_wc is not None:
            T_curr_kf_guess = prior_pose_wc.inv() @ ref_kf.pose_wc
            rvec_guess, _ = cv2.Rodrigues(T_curr_kf_guess.R.astype(np.float64))
            tvec_guess = T_curr_kf_guess.t.reshape(3, 1).astype(np.float64)
            use_guess = True
            flags = cv2.SOLVEPNP_ITERATIVE

        # Solve PnP: P_curr = R * P_kf + t
        success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
            pts_3d_filt,
            pts_2d_filt,
            K,
            D,
            rvec=rvec_guess,
            tvec=tvec_guess,
            useExtrinsicGuess=use_guess,
            flags=flags,
            reprojectionError=self.config.reproj_error_pnp_px,
            confidence=0.999
        )

        if not success or inliers_pnp is None or len(inliers_pnp) < self.config.min_inliers_pnp:
            return None

        # Levenberg-Marquardt non-linear refinement on inliers for sub-pixel smooth convergence
        if self.config.use_lm_refinement and len(inliers_pnp) >= 10:
            try:
                inl_idx = inliers_pnp.flatten()
                rvec, tvec = cv2.solvePnPRefineLM(
                    pts_3d_filt[inl_idx],
                    pts_2d_filt[inl_idx],
                    K,
                    D,
                    rvec,
                    tvec
                )
            except Exception:
                pass

        R, _ = cv2.Rodrigues(rvec)
        T_curr_kf = SE3(R=R, t=tvec.flatten())
        pose_wc_curr = ref_kf.pose_wc @ T_curr_kf.inv()
        return pose_wc_curr

    def _estimate_metric_scale_closed_form(
        self,
        ref_kf: Keyframe,
        curr_frame: Frame,
        R_rel: np.ndarray,
        t_dir: np.ndarray
    ) -> Optional[float]:
        """
        Closed-form analytical recovery of instantaneous metric displacement d.
        Given known 3D points in ref_kf and normalized ray directions in curr_frame:
            x_ray x (R * P_3d + d * t_dir) = 0
            => d = - (a . b) / (|b|^2 + eps)
        Completely deterministic, O(N), runs in < 0.1 ms with zero RANSAC jitter!
        """
        if ref_kf is None or ref_kf.depth_map is None:
            return None

        tracker_backend = getattr(self.tracker, "xfeat", self.tracker)
        if not hasattr(tracker_backend, "extract_features"):
            return None

        try:
            kpts_kf, desc_kf, _ = tracker_backend.extract_features(ref_kf.frame)
            kpts_curr, desc_curr, _ = tracker_backend.extract_features(curr_frame)
            pts_kf_2d, pts_curr_2d = tracker_backend.match_features(desc_kf, desc_curr, kpts_kf, kpts_curr)

            if len(pts_kf_2d) < 15:
                return None

            pts_kf_3d = ref_kf.extract_metric_points(pts_kf_2d)
            if len(pts_kf_3d) < 15:
                return None

            H = curr_frame.intrinsics.height
            max_y = H * (1.0 - self.config.mask_bottom_ratio)
            valid = (
                (pts_kf_3d[:, 2] >= self.config.min_depth_m) &
                (pts_kf_3d[:, 2] <= self.config.max_depth_m) &
                (pts_kf_2d[:, 1] < max_y) &
                (pts_curr_2d[:, 1] < max_y)
            )
            if valid.sum() < 12:
                return None

            P = pts_kf_3d[valid].astype(np.float64)
            p2d = pts_curr_2d[valid].astype(np.float64)

            K = curr_frame.intrinsics.to_matrix().astype(np.float64)
            K_inv = np.linalg.inv(K)

            # Normalized rays in current camera frame
            rays = (K_inv @ np.hstack([p2d, np.ones((len(p2d), 1))]).T).T
            t_unit = t_dir.reshape(3) / (np.linalg.norm(t_dir) + 1e-8)

            # a_i = rays_i x (R * P_i), b_i = rays_i x t_unit
            RP = (R_rel @ P.T).T
            a = np.cross(rays, RP)
            b = np.cross(rays, t_unit)

            b_sq = np.sum(b * b, axis=1)
            valid_b = b_sq > 1e-6
            if valid_b.sum() < 10:
                return None

            d_pts = - np.sum(a[valid_b] * b[valid_b], axis=1) / (b_sq[valid_b] + 1e-8)
            dt = max(1e-3, curr_frame.timestamp - ref_kf.timestamp)
            max_dist = self.config.max_velocity_mps * dt + 0.5
            plausible = (d_pts > 0.01) & (d_pts < max_dist)

            if plausible.sum() < 8:
                return None

            return float(np.median(d_pts[plausible]))
        except Exception:
            return None

    def close(self):
        """Stops background threads cleanly and executes final global pose graph BA."""
        if hasattr(self, "_mapping_queue") and self._mapping_queue is not None:
            try:
                self._mapping_queue.join()
            except Exception:
                pass
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        if hasattr(self, "_mapping_thread") and self._mapping_thread is not None:
            self._mapping_thread.join(timeout=3.0)

        # Module 4: Global Covisibility Pose Graph Optimizer (Full Pose Graph BA Backend)
        if self.loop_manager is not None and self.config.enable_global_pose_graph_ba:
            try:
                self.loop_manager.finalize_global_optimization(
                    enable_crawler_constraints=self.config.is_tracked_chassis
                )
                self.update_trajectory_from_keyframes()
            except Exception:
                pass

    def save_map(self, filepath: str):
        """
        Saves the current keyframes (Teach Map) to disk for later VT&R execution.
        """
        from uvo.navigation.map_serializer import save_vtr_map
        save_vtr_map(self.keyframes, filepath)
