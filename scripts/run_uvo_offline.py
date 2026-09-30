"""
UVO Offline Visual Odometry Runner.
Runs the UVO modular pipeline on rover field video recordings,
estimates metric 3D camera trajectory, and exports TUM, CSV, and GeoJSON files.
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

# Add UVO root to python path
UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

import uvo
from uvo.core.frame import CameraIntrinsics
from uvo.core.geometry import SE3
from uvo.core.sensors import GeoCoordinateTransformer, CameraExtrinsics
from uvo.frontend.base import FrontendMode
from uvo.metric.base import DepthModelType
from uvo.pipeline import UVOPipeline, UVOConfig


def load_calibration(calib_path: str, width: int = 704, height: int = 576) -> CameraIntrinsics:
    """
    Loads camera calibration from calib text file.
    Expected format: fx fy cx cy [k1 k2 p1 p2 k3]
    """
    with open(calib_path, "r") as f:
        line = f.readline().strip()
    vals = [float(x) for x in line.split()]
    return CameraIntrinsics.from_tuple(tuple(vals), width=width, height=height)


def draw_hud(
    frame_bgr: np.ndarray,
    frame_id: int,
    fps: float,
    current_pose: SE3,
    tracking_res: Optional[uvo.TrackingResult],
    is_keyframe: bool,
    keyframe_count: int
) -> np.ndarray:
    """
    Renders high-contrast tactical HUD with inlier feature tracks, metric position, and telemetry.
    """
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # Draw matched inliers if available
    if tracking_res is not None and tracking_res.inlier_mask is not None:
        mask = tracking_res.inlier_mask
        pts_curr = tracking_res.matched_kpts1[mask]
        pts_prev = tracking_res.matched_kpts0[mask]
        
        # Subsample points for clean visualization
        n_pts = len(pts_curr)
        draw_count = min(n_pts, 120)
        if draw_count > 0:
            step = max(1, n_pts // draw_count)
            for i in range(0, n_pts, step):
                p0 = (int(round(pts_prev[i][0])), int(round(pts_prev[i][1])))
                p1 = (int(round(pts_curr[i][0])), int(round(pts_curr[i][1])))
                cv2.line(vis, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(vis, p1, 3, (0, 255, 255), -1)

    # Draw HUD overlay panel
    overlay = vis.copy()
    box_w = min(460, w - 30)
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    t = current_pose.t
    q = current_pose.to_quat()
    if is_keyframe and tracking_res and hasattr(tracking_res, "is_loop_closed") and tracking_res.is_loop_closed:
        kf_badge = "[LOOP CLOSED]"
        badge_color = (255, 255, 0) # Cyan
    elif is_keyframe:
        kf_badge = "[KEYFRAME]"
        badge_color = (0, 165, 255) # Orange
    else:
        kf_badge = "[TRACKING]"
        badge_color = (0, 255, 0) # Green

    cv2.putText(vis, f"UVO Visual Odometry (v{uvo.__version__})", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, f"State: {kf_badge} | Frame: #{frame_id:04d} | Keyframes: {keyframe_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_color, 1)
    cv2.putText(vis, f"Metric Pos: X={t[0]:+.2f}m  Y={t[1]:+.2f}m  Z={t[2]:+.2f}m", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    inliers_str = f"{tracking_res.num_inliers}/{tracking_res.num_matches} ({tracking_res.inlier_ratio:.1f}%)" if tracking_res else "N/A"
    tracker_name = tracking_res.tracker_name if tracking_res else "Init"
    cv2.putText(vis, f"Tracker: {tracker_name} | Inliers: {inliers_str}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    dist_traveled = float(np.linalg.norm(t))
    cv2.putText(vis, f"Dist Origin: {dist_traveled:.2f} m | FPS: {fps:.1f}", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="UVO Offline Visual Odometry Runner")
    parser.add_argument("--video", type=str, default="front_190610.mp4", help="Path to input video file")
    parser.add_argument("--calib", type=str, default="calib_dahua_704.txt", help="Path to calibration file")
    parser.add_argument("--start_frame", type=int, default=0, help="Initial frame index to start tracking from")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum number of frames to process (0 for full video)")
    parser.add_argument("--step", type=int, default=4, help="Frame step/stride")
    parser.add_argument("--frontend", type=str, default="hybrid", choices=["hybrid", "xfeat", "liftfeat"], help="Frontend tracker")
    parser.add_argument("--depth_mode", type=str, default="unidepth", choices=["unidepth", "metric3d", "none"], help="Metric depth conditioning")
    parser.add_argument("--out_dir", type=str, default="UVO/output", help="Directory to save outputs")
    parser.add_argument("--save_vis", action="store_true", help="Save annotated visualization video")
    parser.add_argument("--loop_gap_dist", type=float, default=75.0, help="Minimum accumulated odometric distance (meters) for loop closure (speed-independent)")
    parser.add_argument("--loop_gap_sec", type=float, default=0.0, help="Optional minimum temporal gap in seconds for loop closure")
    parser.add_argument("--loop_min_sim", type=float, default=0.76, help="Minimum VPR similarity for loop candidate")
    parser.add_argument("--disable_loop", action="store_true", help="Disable loop closure (pure VO mode)")
    parser.add_argument("--anchor_lat", type=float, default=50.4501, help="Reference origin latitude for GeoJSON")
    parser.add_argument("--anchor_lon", type=float, default=30.5234, help="Reference origin longitude for GeoJSON")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Open Video Stream
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file: {args.video}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 704
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576

    # 2. Load Calibration
    intrinsics = load_calibration(args.calib, width=vid_w, height=vid_h)

    print("=" * 80)
    print(f"UVO: Unnamed Visual Odometry (v{uvo.__version__})")
    print(f"  - Input Video:    {args.video} ({total_video_frames} frames @ {video_fps:.1f} FPS, {vid_w}x{vid_h})")
    print(f"  - Start Frame:    {args.start_frame}")
    print(f"  - Frame Step:     {args.step}")
    print(f"  - Calibration:    {args.calib} (fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f})")
    print(f"  - Frontend:       {args.frontend}")
    print(f"  - Metric Depth:   {args.depth_mode}")
    print(f"  - Loop Closure:   {'DISABLED' if args.disable_loop else f'ENABLED (min_dist={args.loop_gap_dist:.1f}m, min_sim={args.loop_min_sim:.2f})'}")
    print(f"  - Output Dir:     {out_path}")
    print("=" * 80)

    # 3. Configure Pipeline
    f_mode = FrontendMode(args.frontend)
    enable_depth = (args.depth_mode != "none")
    d_mode = DepthModelType(args.depth_mode) if enable_depth else DepthModelType.UNIDEPTH

    enable_loop = (not args.disable_loop)
    config = UVOConfig(
        frontend_mode=f_mode,
        depth_mode=d_mode,
        enable_metric_depth=enable_depth,
        enable_loop_closure=enable_loop,
        loop_min_gap_dist_m=args.loop_gap_dist,
        loop_min_gap_time_sec=args.loop_gap_sec,
        loop_min_vpr_sim=args.loop_min_sim,
        keyframe_parallax_thresh_px=28.0,
        keyframe_interval_max=15
    )
    pipeline = UVOPipeline(config=config)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # Video Writer if save_vis requested
    writer = None
    if args.save_vis:
        vis_file = str(out_path / "uvo_odometry_vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vis_file, fourcc, video_fps / args.step, (vid_w, vid_h))
        print(f"[+] Recording visualization to: {vis_file}")

    frames_remaining = (total_video_frames - args.start_frame) // args.step
    max_frames_to_process = args.max_frames if args.max_frames > 0 else frames_remaining

    processed_count = 0
    raw_frame_idx = args.start_frame
    t_pipeline_start = time.perf_counter()

    print(f"\n[*] Commencing Visual Odometry tracking loop ({max_frames_to_process} frames to process)...")
    while cap.isOpened() and processed_count < max_frames_to_process:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if raw_frame_idx % args.step != 0:
            raw_frame_idx += 1
            continue

        timestamp = raw_frame_idx / video_fps
        t_frame_start = time.perf_counter()

        # Execute UVO pipeline step
        curr_pose, is_kf, tracking_res = pipeline.process_image(
            image=frame_bgr,
            timestamp=timestamp,
            intrinsics=intrinsics
        )

        dt_frame = time.perf_counter() - t_frame_start
        fps_instant = 1.0 / max(1e-4, dt_frame)
        processed_count += 1
        raw_frame_idx += 1

        # Periodic console logging
        t_vec = curr_pose.t
        is_loop = (pipeline.last_loop_event is not None and pipeline.last_loop_event.timestamp == timestamp)
        if is_loop:
            kf_tag = "[LOOP]"
            ev = pipeline.last_loop_event
            print(f"  >>> LOOP CLOSED with Keyframe #{ev.candidate_kf_id}! (Inliers={ev.num_inliers_3d}, RMSE={ev.rmse_3d_m:.3f}m, Correction={ev.correction_norm_m:.3f}m)")
        elif is_kf:
            kf_tag = "[KF]  "
        else:
            kf_tag = "      "

        inliers_tag = f"Inliers={tracking_res.num_inliers:4d}" if tracking_res else "Init        "
        if is_kf or is_loop or (processed_count % 20 == 0) or (processed_count == 1):
            print(f"[{processed_count:04d}/{max_frames_to_process}] {kf_tag} "
                  f"Pos=[{t_vec[0]:+6.2f}, {t_vec[1]:+6.2f}, {t_vec[2]:+6.2f}] m | "
                  f"{inliers_tag} | {dt_frame*1000:5.1f} ms ({fps_instant:4.1f} FPS)")

        if writer is not None:
            vis_hud = draw_hud(
                frame_bgr=frame_bgr,
                frame_id=processed_count,
                fps=fps_instant,
                current_pose=curr_pose,
                tracking_res=tracking_res,
                is_keyframe=is_kf,
                keyframe_count=len(pipeline.keyframes)
            )
            writer.write(vis_hud)

    cap.release()
    if writer is not None:
        writer.release()

    total_time = time.perf_counter() - t_pipeline_start
    avg_fps = processed_count / max(1e-4, total_time)
    loops_count = len(pipeline.loop_manager.closed_loops) if pipeline.loop_manager else 0
    print("\n" + "=" * 80)
    print(f"[+] Tracking Complete! Processed {processed_count} frames in {total_time:.2f} s ({avg_fps:.2f} FPS)")
    print(f"[+] Total Keyframes generated: {len(pipeline.keyframes)}")
    print(f"[+] Total Loops Closed (PGO):  {loops_count}")

    # 4. Export Trajectories
    pipeline.update_trajectory_from_keyframes()
    tum_path = str(out_path / "trajectory_tum.txt")
    kf_tum_path = str(out_path / "trajectory_keyframes_tum.txt")
    csv_path = str(out_path / "trajectory_full.csv")
    geojson_path = str(out_path / "trajectory_map.geojson")

    pipeline.trajectory.save_tum(tum_path)
    pipeline.get_keyframe_trajectory().save_tum(kf_tum_path)
    pipeline.trajectory.save_csv(csv_path)

    geo_trans = GeoCoordinateTransformer(args.anchor_lat, args.anchor_lon, 150.0)
    pipeline.trajectory.save_geojson(geojson_path, geo_trans)

    print(f"[+] Saved TUM Trajectory:       {tum_path}")
    print(f"[+] Saved Keyframes TUM:        {kf_tum_path}")
    print(f"[+] Saved CSV Trajectory:       {csv_path}")
    print(f"[+] Saved GeoJSON Map:          {geojson_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
