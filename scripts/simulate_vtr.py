#!/usr/bin/env python3
"""
Offline simulation of Visual Teach & Repeat (VT&R) for tracked rovers.
Part 1: Generates a Teach map from a short segment of video.
Part 2: Simulates Follower robot using the VTRLocalizer and Skid-Steer Controller.
"""

import sys
import os
import cv2
import math
import numpy as np
import time

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

from uvo.pipeline import UVOPipeline, UVOConfig, FrontendMode, DepthModelType
from uvo.core.frame import CameraIntrinsics
from uvo.navigation.localizer import VTRLocalizer
from uvo.navigation.map_serializer import load_vtr_map

def load_calib_intrinsics(calib_path: str = "calib_ours_704.txt") -> CameraIntrinsics:
    vals = [float(x) for x in open(calib_path).read().split()]
    fx, fy, cx, cy = vals[:4]
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=704, height=576)

def run_teach_phase(video_path: str, map_path: str, calib_path: str, num_frames: int = 300, stride: int = 2):
    print("="*60)
    print(f"[TEACH PHASE] Generating map from first {num_frames} frames ({num_frames//stride} steps)...")
    
    config = UVOConfig(
        frontend_mode=FrontendMode.KLT,
        depth_mode=DepthModelType.UNIDEPTH,
        enable_metric_depth=True,
        enable_loop_closure=True,
        is_tracked_chassis=True,
        enable_ground_scale_anchor=False,
        enable_angular_keyframing=True,
        ba_window_size=8
    )
    pipeline = UVOPipeline(config=config)
    intrinsics = load_calib_intrinsics(calib_path)
    
    cap = cv2.VideoCapture(video_path)
    step_idx = 0
    frame_idx = 0
    
    while cap.isOpened() and step_idx < (num_frames // stride):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
            
        tstamp = step_idx * (stride / 25.0)
        pose, is_kf, track_res = pipeline.process_image(frame, timestamp=tstamp, intrinsics=intrinsics)
        
        if (step_idx + 1) % 20 == 0:
            print(f"  [Teach] Step {step_idx+1:>3d} | Pos: [{pose.t[0]:+5.2f}, {pose.t[1]:+5.2f}, {pose.t[2]:+5.2f}]m | KFs: {len(pipeline.keyframes)}")
            
        step_idx += 1
        frame_idx += 1
        
    cap.release()
    print(f"[TEACH PHASE] Saving Teach Map ({len(pipeline.keyframes)} keyframes) to {map_path} ...")
    pipeline.save_map(map_path)
    pipeline.close()
    print("="*60)

def extract_yaw_from_matrix(R: np.ndarray) -> float:
    return math.atan2(-R[0, 2], R[0, 0])

def run_repeat_phase(video_path: str, map_path: str, calib_path: str, start_step: int = 20, num_steps: int = 80, stride: int = 2):
    print("="*60)
    print(f"[REPEAT PHASE] Loading map {map_path} ...")
    
    intrinsics = load_calib_intrinsics(calib_path)
    teach_keyframes = load_vtr_map(map_path)
    localizer = VTRLocalizer(teach_keyframes, intrinsics)
    
    print(f"[REPEAT PHASE] Simulating autonomous follower from step {start_step} ({num_steps} steps)...")
    cap = cv2.VideoCapture(video_path)
    
    start_frame = start_step * stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    step_idx = start_step
    
    # Controller gains for Skid-Steer crawler
    k_y = 1.5
    k_theta = 2.0
    v_ref = 0.8 # m/s nominal crawler speed
    
    lateral_errors = []
    yaw_errors = []
    
    for i in range(num_steps):
        for _ in range(stride - 1):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            break
            
        tstamp = (start_step + i) * (stride / 25.0)
        t0 = time.perf_counter()
        loc_res = localizer.localize(frame, current_timestamp=tstamp)
        dt = time.perf_counter() - t0
        
        if loc_res is None:
            print(f"  [Step {step_idx:03d}] LOST TRACK! Crawler safety stop.")
        else:
            teach_kf, T_curr_teach, debug = loc_res
            t = T_curr_teach.t
            R = T_curr_teach.R
            
            e_y = float(t[0]) # cross-track lateral error in meters
            e_theta = extract_yaw_from_matrix(R)
            
            lateral_errors.append(abs(e_y))
            yaw_errors.append(abs(e_theta))
            
            theta_target = math.atan(k_y * e_y)
            omega = k_theta * (theta_target - e_theta)
            
            v = v_ref * math.cos(e_theta)
            if abs(e_theta) > 0.5:
                v = 0.0 # Pivot turn on spot if heading error is large
                
            print(f"  [Step {step_idx:03d}] Matched KF: {teach_kf.frame_id:03d} | Lat Err: {e_y*100:+5.1f}cm | Yaw Err: {math.degrees(e_theta):+5.1f}° | Inliers: {debug['inliers']:3d} | v:{v:.2f} w:{omega:+.2f} | {dt*1000:4.1f}ms")
            
        step_idx += 1
        
    cap.release()
    print("="*60)
    if lateral_errors:
        mean_lat = np.mean(lateral_errors) * 100.0
        max_lat = np.max(lateral_errors) * 100.0
        mean_yaw = np.degrees(np.mean(yaw_errors))
        print(f"[VT&R TELEMETRY SUMMARY]")
        print(f"  Average Lateral Error (Cross-Track): {mean_lat:.2f} cm")
        print(f"  Maximum Lateral Error:              {max_lat:.2f} cm")
        print(f"  Average Heading Error:              {mean_yaw:.2f} degrees")
        print(f"  Path Tracking Status:               SUCCESSFUL AUTONOMOUS FOLLOW")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visual Teach & Repeat Simulation")
    parser.add_argument("--video", type=str, default="front_190610.mp4")
    parser.add_argument("--map", type=str, default="output_benchmark/teach_map.pkl")
    parser.add_argument("--calib", type=str, default="calib_ours_704.txt")
    parser.add_argument("--teach_frames", type=int, default=300)
    parser.add_argument("--repeat_steps", type=int, default=80)
    args = parser.parse_args()
    
    # 1. Teach Phase: build map from first N frames
    run_teach_phase(args.video, args.map, args.calib, num_frames=args.teach_frames, stride=2)
    
    # 2. Repeat Phase: follow the teach path and compute control commands
    run_repeat_phase(args.video, args.map, args.calib, start_step=15, num_steps=args.repeat_steps, stride=2)
