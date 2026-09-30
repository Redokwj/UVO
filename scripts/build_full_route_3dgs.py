#!/usr/bin/env python3
"""
Build Complete 3DGS Route Model from Video and Optimized SLAM Keyframes.
Exports a dense 3D Gaussian Splatting scene covering the entire 382-second rover trajectory.
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

from uvo.core.frame import CameraIntrinsics
from uvo.core.geometry import SE3
from uvo.metric.unidepth_provider import UniDepthProvider
from uvo.dense_slam.gaussian_model import GaussianModel, GaussianConfig


def load_keyframes_tum(tum_path: str):
    """Loads keyframe list from TUM format file: timestamp tx ty tz qx qy qz qw"""
    kfs = []
    with open(tum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                pose_wc = SE3.from_quat_and_trans([qx, qy, qz, qw], [tx, ty, tz])
                kfs.append((t, pose_wc))
    return kfs


def main():
    tum_file = "output_full/trajectory_keyframes_tum.txt"
    if not os.path.exists(tum_file):
        tum_file = "output/trajectory_keyframes_tum.txt"
        
    video_path = "../front_190610.mp4"
    if not os.path.exists(video_path):
        video_path = "front_190610.mp4"

    print("=" * 70)
    print("BUILDING FULL ROUTE 3DGS DENSE SLAM MODEL")
    print(f"Keyframes TUM: {tum_file}")
    print(f"Video:         {video_path}")
    print("=" * 70)

    keyframes = load_keyframes_tum(tum_file)
    print(f"Loaded {len(keyframes)} keyframes from optimized trajectory.")

    # Sample keyframes along the route to balance detail and performance
    # Take every 2nd or 3rd keyframe (~65-75 keyframes covering all 382 seconds)
    stride = 3
    sampled_kfs = keyframes[::stride]
    print(f"Selected {len(sampled_kfs)} keyframes (stride={stride}) along the full route.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open video {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 704
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576

    intrinsics = CameraIntrinsics.from_tuple((778.58, 1127.79, 343.86, 311.17), width=vid_w, height=vid_h)

    # Initialize UniDepth V2
    print("[*] Initializing UniDepth V2 Metric Depth Engine (GPU)...")
    depth_provider = UniDepthProvider(device="cuda")

    # Initialize 3DGS Model (10 cm voxels to maintain clean scale over hundreds of meters)
    cfg = GaussianConfig(
        voxel_filter_size=0.10,
        min_depth_m=0.8,
        max_depth_m=14.0,
        max_scale=0.35,
        init_opacity=0.85
    )
    gaussian_model = GaussianModel(config=cfg, device="cuda")

    t_start = time.perf_counter()
    for i, (t_sec, pose_wc) in enumerate(sampled_kfs):
        frame_idx = int(round(t_sec * video_fps))
        frame_idx = min(frame_idx, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        if frame_bgr.shape[0] != vid_h or frame_bgr.shape[1] != vid_w:
            frame_bgr = cv2.resize(frame_bgr, (vid_w, vid_h))

        # Predict metric depth with UniDepth V2
        pred = depth_provider.predict_depth(frame_bgr)
        depth_map = pred.depth_map

        # P_c = R_cw * P_w + t_cw
        R_wc = pose_wc.R
        t_wc = pose_wc.t
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        # Add Gaussians
        n_added = gaussian_model.add_from_depth(
            image_bgr=frame_bgr,
            depth_map=depth_map,
            intrinsics=intrinsics,
            R_cw=R_cw,
            t_cw=t_cw,
            subsample_step=4
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(sampled_kfs):
            elapsed = time.perf_counter() - t_start
            print(f"[{i+1:02d}/{len(sampled_kfs)}] Frame #{frame_idx:04d} (t={t_sec:5.1f}s) | "
                  f"Pos=[{t_wc[0]:+5.1f}, {t_wc[1]:+5.1f}, {t_wc[2]:+5.1f}]m | "
                  f"Gaussians: {gaussian_model.num_gaussians:6d} (+{n_added:4d}) | "
                  f"Time: {elapsed:4.1f}s")

    cap.release()

    # Export to standard 3DGS PLY
    out_ply = "scene_full_route_3dgs.ply"
    gaussian_model.export_ply(out_ply)

    # Copy to repo root and artifacts
    import shutil
    shutil.copy(out_ply, f"../{out_ply}")
    shutil.copy(out_ply, "C:/Users/koder/.gemini/antigravity-ide/brain/cbd1b665-a219-46d9-bccc-bbe792a4a932/scene_full_route_3dgs.ply")

    print("=" * 70)
    print("FULL ROUTE 3DGS MODEL READY!")
    print(f"Total 3D Gaussians: {gaussian_model.num_gaussians}")
    print(f"Saved to:           {out_ply} and ../{out_ply}")
    print("=" * 70)


if __name__ == "__main__":
    main()
