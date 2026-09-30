"""
UVO Evaluation Script: Evaluate UVO Trajectory against Ground-Truth GPS Logs.
Computes Absolute Trajectory Error (ATE RMSE), scale consistency, and plots top-down trajectory.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

from uvo.core.geometry import SE3
from uvo.core.trajectory import Trajectory, TrajectoryPoint


def load_gps_csv(filepath: str) -> Trajectory:
    """
    Loads recorded rover GPS log from CSV file (format: t, x, y, path).
    """
    traj = Trajectory(name="gps_ground_truth")
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")
        t_idx = 0
        x_idx = 1
        y_idx = 2
        
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                t = float(parts[t_idx])
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = 0.0 # 2D ground plane GPS
                pose = SE3(t=[x, y, z])
                traj.add_point(TrajectoryPoint(timestamp=t, pose_wc=pose, status="GPS_GT"))
            except ValueError:
                continue
    return traj


def load_tum_file(filepath: str) -> Trajectory:
    """
    Loads estimated trajectory from TUM format: timestamp tx ty tz qx qy qz qw
    """
    traj = Trajectory(name="uvo_estimate")
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                pose = SE3.from_quat_and_trans([qx, qy, qz, qw], [tx, ty, tz])
                traj.add_point(TrajectoryPoint(timestamp=t, pose_wc=pose, status="EST"))
            except ValueError:
                continue
    return traj


def plot_trajectory_comparison(
    traj_est: Trajectory,
    traj_gt: Trajectory,
    ate_rigid: dict,
    ate_sim3: dict,
    out_file: str
):
    """
    Generates a clear 2D bird's-eye comparison plot with telemetry and metrics.
    """
    pos_est = traj_est.get_positions()
    pos_gt = traj_gt.get_positions()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Top-Down Bird's Eye View (X vs Y)
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], "k--", label="Ground Truth (GPS)", linewidth=2.0)
    ax1.plot(pos_est[:, 0], pos_est[:, 1], "r-", label="UVO Visual Odometry", linewidth=2.0)
    ax1.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], color="green", s=100, label="Start (0,0)", zorder=5)
    ax1.set_xlabel("X [meters] (East)")
    ax1.set_ylabel("Y [meters] (North)")
    ax1.set_title("Top-Down Trajectory: UVO vs Rover GPS")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axis("equal")

    # 2. Cumulative distance & Error text box
    ax2.axis("off")
    info_text = (
        "=== UVO ACCURACY EVALUATION METRICS ===\n\n"
        f"Ground Truth File:    {traj_gt.name}\n"
        f"Estimated Trajectory: {traj_est.name}\n"
        f"Evaluated Poses:      {len(pos_est)} vs {len(pos_gt)} GPS points\n\n"
        "--- Rigid ATE (True Metric Scale, SE3) ---\n"
        f"  * ATE RMSE:         {ate_rigid['rmse']:.3f} m\n"
        f"  * Mean Error:       {ate_rigid['mean']:.3f} m\n"
        f"  * Median Error:     {ate_rigid['median']:.3f} m\n"
        f"  * Max Deviation:    {ate_rigid['max']:.3f} m\n\n"
        "--- Sim3 Alignment (Scale Drift Analysis) ---\n"
        f"  * Sim3 RMSE:        {ate_sim3['rmse']:.3f} m\n"
        f"  * Fitted Scale:     {ate_sim3['scale']:.4f} (1.0 = zero drift)\n"
        f"  * Scale Error:      {abs(1.0 - ate_sim3['scale']) * 100.0:.2f} %\n\n"
        "Status: Monocular Metric Scale Anchored by UniDepth V2"
    )
    ax2.text(0.05, 0.5, info_text, fontsize=11, family="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="#888888"))

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[+] Saved evaluation comparison plot to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate UVO Trajectory against Rover GPS Log")
    parser.add_argument("--est", type=str, required=True, help="Path to estimated TUM trajectory file")
    parser.add_argument("--gps", type=str, default="gps_loop.csv", help="Path to ground truth GPS CSV")
    parser.add_argument("--out_plot", type=str, default="UVO/output_test/trajectory_vs_gps.png", help="Path to output plot image")
    args = parser.parse_args()

    print("=" * 80)
    print("UVO BENCHMARK EVALUATOR: VISUAL ODOMETRY VS GPS GROUND TRUTH")
    print(f"  - Estimate:     {args.est}")
    print(f"  - Ground Truth: {args.gps}")
    print("=" * 80)

    traj_est = load_tum_file(args.est)
    traj_gt = load_gps_csv(args.gps)

    print(f"[+] Loaded {len(traj_est)} estimated poses and {len(traj_gt)} GPS ground-truth points.")

    # 1. Evaluate with SE(3) Rigid Alignment (Measures true absolute metric scale!)
    ate_rigid = traj_est.compute_ate(traj_gt, align_scale=False)
    
    # 2. Evaluate with Sim(3) Similarity Alignment (Measures scale drift)
    ate_sim3 = traj_est.compute_ate(traj_gt, align_scale=True)

    print("\n" + "-" * 40)
    print("RESULTS (Rigid SE(3) Metric Scale):")
    print(f"  RMSE:      {ate_rigid['rmse']:.3f} meters")
    print(f"  Mean:      {ate_rigid['mean']:.3f} meters")
    print(f"  Median:    {ate_rigid['median']:.3f} meters")
    print(f"  Max Error: {ate_rigid['max']:.3f} meters")
    print("-" * 40)
    print("RESULTS (Sim(3) Scale Consistency):")
    print(f"  RMSE:      {ate_sim3['rmse']:.3f} meters")
    print(f"  Scale (s): {ate_sim3['scale']:.4f} (relative to 1.0)")
    scale_drift_pct = abs(1.0 - ate_sim3['scale']) * 100.0
    print(f"  Drift:     {scale_drift_pct:.2f}%")
    print("-" * 40)

    plot_trajectory_comparison(traj_est, traj_gt, ate_rigid, ate_sim3, args.out_plot)


if __name__ == "__main__":
    main()
