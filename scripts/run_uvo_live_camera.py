#!/usr/bin/env python3
"""
UVO Live Camera SLAM Runner for NVIDIA Jetson (Orin / Xavier).

Supports:
- Jetson CSI Camera (via hardware nvarguscamerasrc)
- USB UVC Webcam (/dev/video0 via V4L2)
- RTSP Network IP Cameras (Dahua / Hikvision via hardware nvv4l2decoder)
- Web streaming for tablet/browser monitoring (http://<jetson_ip>:8080)
- ROS 2 publication of Odometry and 2.5D Traversability Costmap
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

import threading
from http.server import HTTPServer

from uvo.core.frame import CameraIntrinsics
from uvo.core.geometry import SE3
from uvo.pipeline import UVOPipeline, UVOConfig, FrontendMode, DepthModelType
from uvo.navigation.costmap import TraversabilityCostmap, CostmapConfig
from uvo.navigation.geo_anchor import GeoAnchorManager, GeoPoint
from scripts.visualize_costmap_live import compose_dashboard, colorize_depth, MJPEGStreamHandler


def get_jetson_gstreamer_pipeline(
    camera_type: str,
    source: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30
) -> str:
    """
    Constructs hardware-accelerated GStreamer pipelines optimized for NVIDIA Jetson.
    """
    if camera_type == "csi":
        # Jetson onboard CSI camera (IMX219 / IMX477)
        sensor_id = int(source) if source.isdigit() else 0
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=1"
        )
    elif camera_type == "rtsp":
        # Network IP Camera (Dahua / Hikvision) with Jetson Hardware NVDEC H.264/H.265 Decoding
        return (
            f"rtspsrc location={source} latency=0 buffer-mode=auto ! "
            f"rtph264depay ! "
            f"nvv4l2decoder enable-max-performance=1 ! "
            f"nvvidconv ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=1"
        )
    else:
        # Standard USB V4L2 Webcam
        dev_idx = source if source.startswith("/dev/video") else f"/dev/video{source}"
        return dev_idx


class AsyncDashboardEncoder(threading.Thread):
    """
    Decoupled Web Dashboard Compositor and JPEG Encoder Thread.
    Composes and compresses tactical display at fixed 25-30 FPS in background,
    ensuring 0 ms blocking overhead in the core SLAM visual odometry loop.
    """
    def __init__(self, target_fps: int = 30):
        super().__init__(daemon=True)
        self.target_fps = target_fps
        self.running = True
        self.lock = threading.Lock()
        self.latest_data = None
        self.new_data_event = threading.Event()

    def update(self, **kwargs):
        with self.lock:
            self.latest_data = kwargs
        self.new_data_event.set()

    def run(self):
        target_dt = 1.0 / max(5, self.target_fps)
        while self.running:
            self.new_data_event.wait(timeout=0.1)
            self.new_data_event.clear()
            if not self.running:
                break

            with self.lock:
                data = self.latest_data
            if data is None:
                continue

            t0 = time.perf_counter()
            try:
                dashboard = compose_dashboard(
                    camera_img=data["frame"],
                    depth_colored=data["depth_colored"],
                    bev_costmap=data["bev_costmap"],
                    frame_idx=data["frame_count"],
                    fps=data["fps"],
                    pose=data["pose"],
                    tracking_res=data["tracking_res"],
                    is_keyframe=data["is_kf"],
                    costmap_res=data["costmap_res"]
                )
                _, jpeg_bytes = cv2.imencode('.jpg', dashboard, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with MJPEGStreamHandler.lock:
                    MJPEGStreamHandler.latest_jpeg = jpeg_bytes.tobytes()
                    MJPEGStreamHandler.frame_id = data["frame_count"]
            except Exception:
                pass

            dt = time.perf_counter() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

    def stop(self):
        self.running = False
        self.new_data_event.set()


def main():
    parser = argparse.ArgumentParser(description="UVO Jetson Live Camera SLAM Runner")
    parser.add_argument("--type", type=str, default="usb", choices=["usb", "csi", "rtsp", "video"], help="Camera interface type")
    parser.add_argument("--source", type=str, default="0", help="Camera device index, CSI sensor-id, RTSP URL, or video path")
    parser.add_argument("--video", type=str, default=None, help="Direct video file path (overrides --type/--source)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--frontend", type=str, default="xfeat", choices=["xfeat", "hybrid", "liftfeat"], help="Front-End Tracker (xfeat: 60+ FPS, hybrid: adaptive)")
    parser.add_argument("--fps", type=int, default=30, help="Capture framerate")
    parser.add_argument("--calib", type=str, default="calib_dahua_704.txt", help="Calibration file")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Enable Tensor Core FP16 acceleration (default: enabled)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 and run in FP32")
    parser.add_argument("--web", action="store_true", help="Start web dashboard at http://0.0.0.0:8080")
    parser.add_argument("--port", type=int, default=8080, help="Web streaming port")
    parser.add_argument("--anchor_lat", type=float, default=None, help="Optional initial latitude for geo-anchor")
    parser.add_argument("--anchor_lon", type=float, default=None, help="Optional initial longitude for geo-anchor")
    parser.add_argument("--bottom_mask", type=float, default=0.18, help="Bottom hood / bumper crop ratio (default: 0.18 for rover)")
    args = parser.parse_args()

    # Resolve Video / Camera Source
    video_path = args.video if args.video is not None else (args.source if args.type == "video" else None)
    if video_path is not None:
        if not os.path.exists(video_path):
            parent_candidate = os.path.join(os.path.dirname(UVO_ROOT), video_path)
            if os.path.exists(parent_candidate):
                video_path = parent_candidate
        print(f"[*] Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    elif args.type in ["csi", "rtsp"]:
        gst_pipe = get_jetson_gstreamer_pipeline(args.type, args.source, args.width, args.height, args.fps)
        print(f"[*] Opening Jetson GStreamer pipeline:\n    {gst_pipe}")
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
    else:
        dev_idx = int(args.source) if args.source.isdigit() else args.source
        print(f"[*] Opening USB camera: {dev_idx}")
        cap = cv2.VideoCapture(dev_idx)

    if not cap.isOpened():
        source_desc = video_path if video_path else f"{args.type} ({args.source})"
        print(f"[!] Error: Could not open video/camera source: {source_desc}")
        sys.exit(1)

    fe_mode_map = {
        "xfeat": FrontendMode.XFEAT,
        "hybrid": FrontendMode.HYBRID,
        "liftfeat": FrontendMode.LIFTFEAT
    }
    frontend_mode = fe_mode_map.get(args.frontend, FrontendMode.XFEAT)

    # Initialize UVO
    print(f"[*] Initializing UVO SLAM on NVIDIA Jetson (Frontend: {args.frontend.upper()} | FP16: {'ON' if args.fp16 else 'OFF'})...")
    config = UVOConfig(
        frontend_mode=frontend_mode,
        depth_mode=DepthModelType.UNIDEPTH,
        enable_metric_depth=True,
        enable_loop_closure=True,
        ba_window_size=8,
        fp16=args.fp16,
        mask_bottom_ratio=args.bottom_mask
    )
    pipeline = UVOPipeline(config=config)
    costmap = TraversabilityCostmap(CostmapConfig(subsample_step=2))
    
    geo_anchor = None
    if args.anchor_lat is not None and args.anchor_lon is not None:
        geo_anchor = GeoAnchorManager(ref_lat=args.anchor_lat, ref_lon=args.anchor_lon)
        geo_anchor.set_position_anchor([0, 0, 0], GeoPoint(args.anchor_lat, args.anchor_lon))
        print(f"[✓] Geo-Anchor locked to: {args.anchor_lat:.6f}, {args.anchor_lon:.6f}")

    # Camera Intrinsics
    calib_file = args.calib
    if not os.path.exists(calib_file):
        parent_calib = os.path.join(os.path.dirname(UVO_ROOT), calib_file)
        if os.path.exists(parent_calib):
            calib_file = parent_calib

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else args.width
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else args.height
    if vid_w <= 0 or vid_h <= 0:
        vid_w, vid_h = args.width, args.height

    if os.path.exists(calib_file):
        try:
            calib_data = np.loadtxt(calib_file)
            intrinsics = CameraIntrinsics(
                fx=float(calib_data[0]), fy=float(calib_data[1]),
                cx=float(calib_data[2]), cy=float(calib_data[3]),
                width=vid_w, height=vid_h
            )
            print(f"[✓] Loaded calibration: {calib_file} (fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, {vid_w}x{vid_h})")
        except Exception:
            intrinsics = CameraIntrinsics(fx=vid_w * 0.8, fy=vid_w * 0.8, cx=vid_w / 2.0, cy=vid_h / 2.0, width=vid_w, height=vid_h)
    else:
        intrinsics = CameraIntrinsics(fx=vid_w * 0.8, fy=vid_w * 0.8, cx=vid_w / 2.0, cy=vid_h / 2.0, width=vid_w, height=vid_h)

    # Setup Geo-Anchor Manager
    if geo_anchor is None:
        geo_anchor = GeoAnchorManager()
    MJPEGStreamHandler.geo_anchor_mgr = geo_anchor

    encoder = None
    if args.web:
        encoder = AsyncDashboardEncoder(target_fps=args.fps)
        encoder.start()
        server = HTTPServer(('0.0.0.0', args.port), MJPEGStreamHandler)
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        print(f"\n[WEB] Live Tactical Tablet Dashboard active: http://0.0.0.0:{args.port}\n")

    print("\n" + "=" * 65)
    print("  UVO SLAM IS RUNNING (THREADED DECOUPLED PIPELINE)!")
    print(f"  Camera Type: {args.type.upper()} ({vid_w}x{vid_h} @ {args.fps} FPS)")
    if args.web:
        print(f"  Tablet Web Dashboard: http://<jetson_ip>:{args.port}")
    print("=" * 65 + "\n")

    frame_count = 0
    t_start = time.perf_counter()
    last_depth_colored = None
    last_bev_costmap = np.full((500, 500, 3), 40, dtype=np.uint8)
    last_costmap_res = None

    try:
        while cap.isOpened():
            t_frame_0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            t_now = time.perf_counter() - t_start

            # High-speed Front-End Tracking (XFeat + PnP against latest ready keyframe)
            pose, is_kf, tracking_res = pipeline.process_image(frame, timestamp=t_now, intrinsics=intrinsics)
            dt = time.perf_counter() - t_frame_0
            instant_fps = 1.0 / max(1e-4, dt)

            # Check if any new keyframe metric depth finished in the background mapping thread
            for kf in reversed(pipeline.keyframes[-3:]):
                if kf.depth_map is not None and getattr(kf, "_costmap_computed", False) is False:
                    kf._costmap_computed = True
                    last_depth_colored = colorize_depth(kf.depth_map)
                    last_costmap_res = costmap.process_depth_map(kf.depth_map, intrinsics)
                    last_bev_costmap = last_costmap_res.render_bev_image(upscale_factor=4)
                    break

            # Send data to background web dashboard encoder (non-blocking, ~0.0001ms)
            if args.web and encoder is not None:
                with MJPEGStreamHandler.lock:
                    MJPEGStreamHandler.current_pose = pose
                    MJPEGStreamHandler.telemetry["x"] = float(pose.t[0])
                    MJPEGStreamHandler.telemetry["y"] = float(pose.t[1])
                    MJPEGStreamHandler.telemetry["z"] = float(pose.t[2])
                    MJPEGStreamHandler.telemetry["fps"] = float(instant_fps)
                    MJPEGStreamHandler.telemetry["inliers"] = int(tracking_res.num_inliers) if tracking_res else 0
                    MJPEGStreamHandler.telemetry["state"] = "KEYFRAME" if is_kf else "TRACKING"
                    if geo_anchor is not None and geo_anchor.is_anchored:
                        lat, lon, alt = geo_anchor.local_to_wgs84(pose.t)
                        MJPEGStreamHandler.telemetry["is_anchored"] = True
                        MJPEGStreamHandler.telemetry["lat"] = lat
                        MJPEGStreamHandler.telemetry["lon"] = lon
                        MJPEGStreamHandler.telemetry["alt"] = alt

                encoder.update(
                    frame=frame,
                    depth_colored=last_depth_colored,
                    bev_costmap=last_bev_costmap,
                    frame_count=frame_count,
                    fps=instant_fps,
                    pose=pose,
                    tracking_res=tracking_res,
                    is_kf=is_kf,
                    costmap_res=last_costmap_res
                )

            # Real-time pacing for recorded video files (smooth 30 FPS playback)
            is_video_file = (args.video is not None) or (args.type == "video")
            if is_video_file:
                target_dt = 1.0 / max(10, args.fps)
                dt_total = time.perf_counter() - t_frame_0
                if dt_total < target_dt:
                    time.sleep(target_dt - dt_total)

            if frame_count % 30 == 0:
                pos = pose.t
                print(f"[Frame #{frame_count:05d}] Pos=[{pos[0]:+5.2f}, {pos[1]:+5.2f}, {pos[2]:+5.2f}]m | "
                      f"Keyframes: {len(pipeline.keyframes)} | "
                      f"Inliers: {tracking_res.num_inliers if tracking_res else 0} | {instant_fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        pipeline.close()
        if encoder is not None:
            encoder.stop()
        cap.release()
        print("Camera released. UVO SLAM stopped.")


if __name__ == "__main__":
    main()
