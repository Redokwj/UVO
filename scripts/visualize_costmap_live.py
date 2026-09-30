#!/usr/bin/env python3
"""
UVO 2.5D Traversability Costmap & SLAM Live Visualizer.

Runs visual odometry on field video, computes real-time metric depth from UniDepth V2,
projects 3D points into a 2.5D Elevation & Traversability Costmap, and displays
a real-time dual-view tactical dashboard (Camera + Depth + 2.5D BEV Costmap).
"""

import os
import sys
import time
import argparse
import threading
from typing import Optional, List, Tuple
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import cv2
import numpy as np
import torch

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

import uvo
from uvo.core.frame import CameraIntrinsics
from uvo.core.geometry import SE3
from uvo.frontend.base import FrontendMode
from uvo.metric.base import DepthModelType
from uvo.pipeline import UVOPipeline, UVOConfig
from uvo.navigation.costmap import TraversabilityCostmap, CostmapConfig, CostmapResult
from uvo.navigation.geo_anchor import GeoAnchorManager, GeoPoint


def load_calibration(calib_path: str, width: int = 704, height: int = 576) -> CameraIntrinsics:
    with open(calib_path, "r") as f:
        line = f.readline().strip()
    vals = [float(x) for x in line.split()]
    return CameraIntrinsics.from_tuple(tuple(vals), width=width, height=height)


def colorize_depth(depth_map: np.ndarray, min_d: float = 0.5, max_d: float = 15.0) -> np.ndarray:
    """Colorizes metric depth map using Turbo colormap."""
    d_clipped = np.clip(depth_map, min_d, max_d)
    norm = ((d_clipped - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return colored


def compose_dashboard(
    camera_img: np.ndarray,
    depth_colored: Optional[np.ndarray],
    bev_costmap: np.ndarray,
    frame_idx: int,
    fps: float,
    pose: SE3,
    tracking_res,
    is_keyframe: bool,
    costmap_res: Optional[CostmapResult] = None
) -> np.ndarray:
    """
    Creates a unified tactical dashboard:
    [ Left: Camera + Depth Inset + Telemetry HUD ] | [ Right: 2.5D BEV Traversability Costmap ]
    Target dimensions: ~ 1280 x 576
    """
    h_cam, w_cam = camera_img.shape[:2]
    left_view = camera_img.copy()
    
    # Draw matched inliers if available
    if tracking_res is not None and tracking_res.inlier_mask is not None:
        mask = tracking_res.inlier_mask
        pts_curr = tracking_res.matched_kpts1[mask]
        pts_prev = tracking_res.matched_kpts0[mask]
        n_pts = len(pts_curr)
        draw_count = min(n_pts, 100)
        if draw_count > 0:
            step = max(1, n_pts // draw_count)
            for i in range(0, n_pts, step):
                p0 = (int(round(pts_prev[i][0])), int(round(pts_prev[i][1])))
                p1 = (int(round(pts_curr[i][0])), int(round(pts_curr[i][1])))
                cv2.line(left_view, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(left_view, p1, 3, (0, 255, 255), -1)

    # Inset Depth Map in the top-right corner of the left view
    if depth_colored is not None:
        inset_w = int(w_cam * 0.35)
        inset_h = int(h_cam * 0.35)
        depth_small = cv2.resize(depth_colored, (inset_w, inset_h))
        cv2.rectangle(depth_small, (0, 0), (inset_w - 1, inset_h - 1), (0, 255, 255), 2)
        cv2.putText(depth_small, "UniDepth Metric (m)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        left_view[15:15 + inset_h, w_cam - 15 - inset_w:w_cam - 15] = depth_small

    # HUD Overlay Box on left view
    overlay = left_view.copy()
    cv2.rectangle(overlay, (15, 15), (380, 165), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, left_view, 0.25, 0, left_view)

    t = pose.t
    badge = "[KEYFRAME]" if is_keyframe else "[TRACKING]"
    b_color = (0, 165, 255) if is_keyframe else (0, 255, 0)
    
    cv2.putText(left_view, "UVO VISUAL ODOMETRY & SLAM", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left_view, f"State: {badge} | Frame: #{frame_idx:04d}", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1, cv2.LINE_AA)
    cv2.putText(left_view, f"Pos: X={t[0]:+.2f}m  Y={t[1]:+.2f}m  Z={t[2]:+.2f}m", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    
    inliers_str = f"{tracking_res.num_inliers}/{tracking_res.num_matches}" if tracking_res else "Init"
    cv2.putText(left_view, f"Tracker: XFeat Hybrid | Inliers: {inliers_str}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(left_view, f"Pipeline FPS: {fps:.1f} FPS", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Resize views to unified height 576
    target_h = 576
    scale_l = target_h / float(h_cam)
    target_wl = int(w_cam * scale_l)
    left_resized = cv2.resize(left_view, (target_wl, target_h))

    h_bev, w_bev = bev_costmap.shape[:2]
    scale_r = target_h / float(h_bev)
    target_wr = int(w_bev * scale_r)
    right_resized = cv2.resize(bev_costmap, (target_wr, target_h))

    # Add legend at bottom right
    legend_y = target_h - 40
    cv2.rectangle(right_resized, (10, legend_y - 20), (target_wr - 10, target_h - 8), (20, 20, 20), -1)
    # Green = Free
    cv2.circle(right_resized, (25, legend_y), 5, (40, 160, 40), -1)
    cv2.putText(right_resized, "Free", (35, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # Yellow = Rough
    cv2.circle(right_resized, (85, legend_y), 5, (0, 200, 255), -1)
    cv2.putText(right_resized, "Rough", (95, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # Red = Obstacle
    cv2.circle(right_resized, (155, legend_y), 5, (0, 0, 240), -1)
    cv2.putText(right_resized, "Obstacle(+)", (165, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # Cyan = Hole/Trench
    cv2.circle(right_resized, (255, legend_y), 5, (255, 180, 0), -1)
    cv2.putText(right_resized, "Trench(-)", (265, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # Magenta = Rollover
    cv2.circle(right_resized, (340, legend_y), 5, (200, 0, 200), -1)
    cv2.putText(right_resized, "Slope>22deg", (350, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    combined = np.hstack([left_resized, right_resized])
    return combined


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    latest_jpeg = None
    frame_id = 0
    telemetry = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "lat": None, "lon": None, "alt": None,
        "fps": 0.0, "inliers": 0, "state": "INIT",
        "is_anchored": False
    }
    geo_anchor_mgr = None
    current_pose = None
    lock = threading.Lock()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <title>UVO Tactical Tablet Dashboard</title>
                <style>
                    :root { --bg: #0b0d13; --card: #151821; --border: #232733; --accent: #00ff88; --warn: #ffaa00; --danger: #ff3344; --text: #e1e4ea; --sub: #8b949e; }
                    body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 12px; }
                    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 12px; }
                    .header h1 { font-size: 1.1rem; margin: 0; color: var(--accent); letter-spacing: 0.5px; }
                    .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); margin-right: 6px; }
                    .stream-container { position: relative; width: 100%; border-radius: 10px; overflow: hidden; border: 2px solid var(--border); background: #000; box-shadow: 0 8px 30px rgba(0,0,0,0.7); }
                    .stream-container img { width: 100%; height: auto; display: block; }
                    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }
                    .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
                    .card .label { font-size: 0.72rem; text-transform: uppercase; color: var(--sub); letter-spacing: 0.5px; margin-bottom: 4px; }
                    .card .val { font-size: 1.15rem; font-weight: 600; font-family: monospace; color: #fff; }
                    .anchor-panel { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-top: 12px; }
                    .anchor-panel h3 { margin: 0 0 10px 0; font-size: 0.95rem; color: #58a6ff; }
                    .form-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
                    .input-group { display: flex; flex-direction: column; flex: 1; min-width: 110px; }
                    .input-group label { font-size: 0.7rem; color: var(--sub); margin-bottom: 2px; }
                    .input-group input { background: #0c0e14; border: 1px solid var(--border); color: #fff; padding: 8px; border-radius: 6px; font-family: monospace; }
                    button { background: #238636; color: #fff; border: none; padding: 9px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
                    button:hover { background: #2ea043; }
                    button.btn-gps { background: #1f6feb; }
                    button.btn-gps:hover { background: #388bfd; }
                    .toast { display: none; padding: 8px; border-radius: 6px; font-size: 0.85rem; margin-top: 8px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <div><span class="status-dot"></span><span style="font-weight:700;">UVO TACTICAL SLAM</span></div>
                    <div style="font-size:0.8rem; color:var(--sub);" id="fps-badge">-- FPS</div>
                </div>
                
                <div class="stream-container">
                    <img src="/stream.mjpg" alt="Live Tactical Feed" />
                </div>

                <div class="grid">
                    <div class="card">
                        <div class="label">Local SLAM Pos</div>
                        <div class="val" id="pos-val">X:+0.00 Y:+0.00</div>
                    </div>
                    <div class="card">
                        <div class="label">Real GPS (WGS84)</div>
                        <div class="val" id="gps-val" style="color:#58a6ff;">Not Anchored</div>
                    </div>
                    <div class="card">
                        <div class="label">Tracking State</div>
                        <div class="val" id="state-val" style="color:var(--accent);">TRACKING</div>
                    </div>
                    <div class="card">
                        <div class="label">Inliers / Points</div>
                        <div class="val" id="inliers-val">-- / --</div>
                    </div>
                </div>

                <div class="anchor-panel">
                    <h3>⚓ Optical Geo-Anchor (Field Coordinate Lock)</h3>
                    <div class="form-row">
                        <div class="input-group">
                            <label>Latitude</label>
                            <input type="number" id="in-lat" step="0.000001" placeholder="50.450123" />
                        </div>
                        <div class="input-group">
                            <label>Longitude</label>
                            <input type="number" id="in-lon" step="0.000001" placeholder="30.523456" />
                        </div>
                        <div class="input-group" style="max-width:90px;">
                            <label>Heading (°)</label>
                            <input type="number" id="in-hdg" step="1" placeholder="0" />
                        </div>
                        <div style="margin-top:14px; display:flex; gap:6px;">
                            <button onclick="setAnchor()">Lock Anchor</button>
                            <button class="btn-gps" onclick="useTabletGPS()">📍 My GPS</button>
                        </div>
                    </div>
                    <div id="toast" class="toast"></div>
                </div>

                <script>
                    async function updateTelemetry() {
                        try {
                            const res = await fetch('/api/telemetry');
                            const data = await res.json();
                            document.getElementById('pos-val').innerText = `X:${data.x>=0?'+':''}${data.x.toFixed(2)} Y:${data.y>=0?'+':''}${data.y.toFixed(2)}`;
                            if (data.is_anchored && data.lat) {
                                document.getElementById('gps-val').innerText = `${data.lat.toFixed(5)}, ${data.lon.toFixed(5)}`;
                            } else {
                                document.getElementById('gps-val').innerText = 'Unanchored';
                            }
                            document.getElementById('state-val').innerText = data.state;
                            document.getElementById('fps-badge').innerText = `${data.fps.toFixed(1)} FPS`;
                            document.getElementById('inliers-val').innerText = `${data.inliers} pts`;
                        } catch(e) {}
                    }
                    setInterval(updateTelemetry, 400);

                    async function setAnchor() {
                        const lat = parseFloat(document.getElementById('in-lat').value);
                        const lon = parseFloat(document.getElementById('in-lon').value);
                        const hdg = parseFloat(document.getElementById('in-hdg').value) || null;
                        if (!lat || !lon) {
                            showToast("Please enter valid Latitude & Longitude", false);
                            return;
                        }
                        try {
                            const res = await fetch('/api/anchor', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({lat, lon, heading: hdg})
                            });
                            const r = await res.json();
                            showToast(r.message, true);
                        } catch(e) {
                            showToast("Error sending anchor", false);
                        }
                    }

                    function useTabletGPS() {
                        if (!navigator.geolocation) {
                            showToast("Geolocation not supported on this device", false);
                            return;
                        }
                        showToast("Acquiring tablet GPS fix...", true);
                        navigator.geolocation.getCurrentPosition((pos) => {
                            document.getElementById('in-lat').value = pos.coords.latitude.toFixed(6);
                            document.getElementById('in-lon').value = pos.coords.longitude.toFixed(6);
                            if (pos.coords.heading) {
                                document.getElementById('in-hdg').value = Math.round(pos.coords.heading);
                            }
                            showToast(`GPS Acquired (±${pos.coords.accuracy.toFixed(0)}m). Tap 'Lock Anchor' to confirm.`, true);
                        }, (err) => {
                            showToast("Could not get GPS: " + err.message, false);
                        }, {enableHighAccuracy: true});
                    }

                    function showToast(msg, ok) {
                        const t = document.getElementById('toast');
                        t.style.display = 'block';
                        t.style.background = ok ? '#1f6feb22' : '#ff334422';
                        t.style.color = ok ? '#58a6ff' : '#ff7b72';
                        t.style.border = ok ? '1px solid #1f6feb66' : '1px solid #ff334466';
                        t.innerText = msg;
                    }
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with MJPEGStreamHandler.lock:
                data = json.dumps(MJPEGStreamHandler.telemetry).encode('utf-8')
            self.wfile.write(data)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            last_sent_id = -1
            try:
                while True:
                    jpg = None
                    with MJPEGStreamHandler.lock:
                        if MJPEGStreamHandler.frame_id != last_sent_id:
                            jpg = MJPEGStreamHandler.latest_jpeg
                            last_sent_id = MJPEGStreamHandler.frame_id
                    if jpg is not None:
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.01)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/anchor':
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode('utf-8'))
                lat = float(data['lat'])
                lon = float(data['lon'])
                heading = float(data['heading']) if data.get('heading') is not None else None
                
                with MJPEGStreamHandler.lock:
                    if MJPEGStreamHandler.geo_anchor_mgr is not None and MJPEGStreamHandler.current_pose is not None:
                        pos = MJPEGStreamHandler.current_pose.t
                        MJPEGStreamHandler.geo_anchor_mgr.set_position_anchor(pos, GeoPoint(lat, lon), heading_deg=heading)
                        MJPEGStreamHandler.telemetry["is_anchored"] = True
                        MJPEGStreamHandler.telemetry["lat"] = lat
                        MJPEGStreamHandler.telemetry["lon"] = lon
                        msg = f"Anchor locked at ({lat:.6f}, {lon:.6f})"
                    else:
                        msg = "Anchor manager or SLAM pose not ready"
                
                resp = json.dumps({"status": "ok", "message": msg}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


class GUIViewer:
    """
    Robust GUI viewer that tries OpenCV HighGUI first,
    and automatically falls back to Tkinter/Matplotlib if HighGUI is not compiled.
    """
    def __init__(self, title: str = "UVO 2.5D Costmap & SLAM"):
        self.title = title
        self.use_cv2 = True
        self.fig = None
        self.ax = None
        self.im_artist = None
        self.is_paused = False
        self.should_exit = False

    def on_key(self, event):
        if event.key in ['q', 'escape']:
            self.should_exit = True
        elif event.key == ' ':
            self.is_paused = not self.is_paused
            state = "PAUSED" if self.is_paused else "RESUMED"
            print(f"[{state}] (press SPACE to toggle)")

    def show(self, frame_bgr: np.ndarray) -> bool:
        if self.should_exit:
            return False

        if self.use_cv2:
            try:
                cv2.imshow(self.title, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    return False
                elif key == ord(' '):
                    while True:
                        k2 = cv2.waitKey(30) & 0xFF
                        if k2 in [ord(' '), ord('q'), 27]:
                            if k2 in [ord('q'), 27]:
                                return False
                            break
                return True
            except cv2.error:
                print("\n[!] OpenCV HighGUI not compiled in this Python environment.")
                print("[*] Automatically switching to native Windows GUI window (Tkinter / Matplotlib)...")
                self.use_cv2 = False
                import matplotlib.pyplot as plt
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(13, 6))
                self.fig.canvas.manager.set_window_title(self.title)
                self.fig.canvas.mpl_connect('key_press_event', self.on_key)
                self.ax.axis("off")
                self.fig.tight_layout(pad=0)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.im_artist = self.ax.imshow(frame_rgb)
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
                return True

        # Matplotlib path
        import matplotlib.pyplot as plt
        if not plt.fignum_exists(self.fig.number):
            return False
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.im_artist.set_data(frame_rgb)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

        while self.is_paused and not self.should_exit:
            if not plt.fignum_exists(self.fig.number):
                return False
            plt.pause(0.05)

        return not self.should_exit

    def close(self):
        if self.use_cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        elif self.fig is not None:
            import matplotlib.pyplot as plt
            try:
                plt.close(self.fig)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="UVO 2.5D Costmap & SLAM Live Visualizer")
    parser.add_argument("--video", type=str, default="front_190610.mp4", help="Video path")
    parser.add_argument("--calib", type=str, default="calib_dahua_704.txt", help="Calibration file")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--max_frames", type=int, default=150, help="Frames to process")
    parser.add_argument("--step", type=int, default=2, help="Frame step")
    parser.add_argument("--gui", action="store_true", help="Show interactive native desktop GUI window")
    parser.add_argument("--web", action="store_true", help="Start web stream at http://localhost:8080")
    parser.add_argument("--port", type=int, default=8080, help="Port for web stream")
    parser.add_argument("--save_video", type=str, default="output_costmap_demo.mp4", help="Path to save output video")
    parser.add_argument("--save_snapshots", action="store_true", help="Save keyframe snapshot images")
    args = parser.parse_args()

    # Search for video file if path relative to repo root
    video_path = args.video
    if not os.path.exists(video_path):
        for candidate in [os.path.join("..", video_path), os.path.join(UVO_ROOT, video_path), os.path.join(os.path.dirname(UVO_ROOT), video_path)]:
            if os.path.exists(candidate):
                video_path = candidate
                break

    calib_path = args.calib
    if not os.path.exists(calib_path):
        for candidate in [os.path.join("..", calib_path), os.path.join(UVO_ROOT, calib_path), os.path.join(os.path.dirname(UVO_ROOT), calib_path)]:
            if os.path.exists(candidate):
                calib_path = candidate
                break

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open video {video_path}")
        sys.exit(1)

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 704
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    intrinsics = load_calibration(calib_path, width=vid_w, height=vid_h)

    # Initialize UVO pipeline
    config = UVOConfig(
        frontend_mode=FrontendMode.HYBRID,
        depth_mode=DepthModelType.UNIDEPTH,
        enable_metric_depth=True,
        enable_loop_closure=True,
        ba_window_size=8
    )
    pipeline = UVOPipeline(config=config)

    # Initialize Costmap Engine
    costmap_engine = TraversabilityCostmap(CostmapConfig(
        min_x=0.5,
        max_x=8.5,
        min_y=-4.0,
        max_y=4.0,
        resolution=0.08,
        camera_height=0.45,
        camera_pitch_deg=10.0,
        safe_step_threshold=0.06,
        obstacle_step_threshold=0.16,
        ditch_depth_threshold=0.15,
        max_traversable_slope_deg=22.0,
        robot_radius=0.40,
        inflation_radius=0.75,
        subsample_step=2
    ))

    # Setup Geo-Anchor Manager
    geo_anchor = GeoAnchorManager()
    MJPEGStreamHandler.geo_anchor_mgr = geo_anchor

    # Setup Web streaming server if requested
    if args.web:
        server = HTTPServer(('0.0.0.0', args.port), MJPEGStreamHandler)
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        print(f"\n[WEB] Live Browser Dashboard active: http://localhost:{args.port}")
        print(f"[WEB] Open the link above in Chrome, Edge, or Firefox to watch live.\n")

    # Setup GUI Viewer if requested
    viewer = GUIViewer() if args.gui else None

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = None
    processed_count = 0
    raw_frame_idx = args.start_frame

    last_depth_colored = None
    last_bev_costmap = np.full((500, 500, 3), 40, dtype=np.uint8)
    last_costmap_res = None

    print("=" * 80)
    print("STARTING UVO 2.5D TRAVERSABILITY COSTMAP & SLAM VISUALIZER")
    print(f"Video: {video_path} | Max frames: {args.max_frames} | Step: {args.step}")
    print(f"Interactive GUI window: {'YES' if args.gui else 'NO'}")
    print(f"Web Browser Stream:     {'YES (http://localhost:' + str(args.port) + ')' if args.web else 'NO'}")
    print(f"Output video:           {args.save_video}")
    print("=" * 80)

    while cap.isOpened() and processed_count < args.max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if raw_frame_idx % args.step != 0:
            raw_frame_idx += 1
            continue

        timestamp = raw_frame_idx / video_fps
        t0 = time.perf_counter()

        # UVO pipeline step
        curr_pose, is_kf, tracking_res = pipeline.process_image(
            image=frame_bgr,
            timestamp=timestamp,
            intrinsics=intrinsics
        )
        t_pipeline = time.perf_counter() - t0
        instant_fps = 1.0 / max(1e-4, t_pipeline)

        # When a keyframe is formed, update metric depth and 2.5D costmap!
        if is_kf and len(pipeline.keyframes) > 0:
            kf = pipeline.keyframes[-1]
            if kf.depth_map is not None:
                last_depth_colored = colorize_depth(kf.depth_map)
                last_costmap_res = costmap_engine.process_depth_map(kf.depth_map, intrinsics)
                last_bev_costmap = last_costmap_res.render_bev_image(upscale_factor=5)

        # Compose unified dashboard
        dashboard = compose_dashboard(
            camera_img=frame_bgr,
            depth_colored=last_depth_colored,
            bev_costmap=last_bev_costmap,
            frame_idx=processed_count,
            fps=instant_fps,
            pose=curr_pose,
            tracking_res=tracking_res,
            is_keyframe=is_kf,
            costmap_res=last_costmap_res
        )

        # Update Web Stream & Telemetry
        if args.web:
            _, jpeg_data = cv2.imencode('.jpg', dashboard, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with MJPEGStreamHandler.lock:
                MJPEGStreamHandler.latest_jpeg = jpeg_data.tobytes()
                MJPEGStreamHandler.current_pose = curr_pose
                MJPEGStreamHandler.telemetry["x"] = float(curr_pose.t[0])
                MJPEGStreamHandler.telemetry["y"] = float(curr_pose.t[1])
                MJPEGStreamHandler.telemetry["z"] = float(curr_pose.t[2])
                MJPEGStreamHandler.telemetry["fps"] = float(instant_fps)
                MJPEGStreamHandler.telemetry["inliers"] = int(tracking_res.num_inliers) if tracking_res else 0
                MJPEGStreamHandler.telemetry["state"] = "KEYFRAME" if is_kf else "TRACKING"
                if geo_anchor is not None and geo_anchor.is_anchored:
                    lat, lon, alt = geo_anchor.local_to_wgs84(curr_pose.t)
                    MJPEGStreamHandler.telemetry["is_anchored"] = True
                    MJPEGStreamHandler.telemetry["lat"] = lat
                    MJPEGStreamHandler.telemetry["lon"] = lon
                    MJPEGStreamHandler.telemetry["alt"] = alt

        # Initialize video writer on first frame
        if writer is None and args.save_video:
            h_out, w_out = dashboard.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = min(15.0, video_fps / args.step)
            writer = cv2.VideoWriter(args.save_video, fourcc, out_fps, (w_out, h_out))
            print(f"[+] Recording dashboard video ({w_out}x{h_out}) to: {args.save_video}")

        if writer is not None:
            writer.write(dashboard)

        # Save snapshot on keyframes
        if args.save_snapshots and is_kf and processed_count in [1, 20, 50, 80, 120]:
            snap_path = f"scratch/dashboard_frame_{processed_count:04d}.png"
            cv2.imwrite(snap_path, dashboard)
            print(f"[Snapshot] Saved {snap_path}")

        # Desktop GUI viewer
        if viewer is not None:
            if not viewer.show(dashboard):
                print("Exit requested by user.")
                break

        processed_count += 1
        raw_frame_idx += 1

        if processed_count % 10 == 0 or is_kf:
            t = curr_pose.t
            kf_tag = "[KF]" if is_kf else "    "
            print(f"[{processed_count:03d}/{args.max_frames}] {kf_tag} Pos=[{t[0]:+5.2f}, {t[1]:+5.2f}, {t[2]:+5.2f}]m | {instant_fps:4.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[+] Video saved successfully to {args.save_video}")
    if viewer is not None:
        viewer.close()

    print("=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
