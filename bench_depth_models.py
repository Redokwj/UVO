import os
import sys
import time
import cv2
import numpy as np
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "bench_frames")
OUT_DIR = os.path.join(BASE_DIR, "docs", "RandD")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Metric3D v2 (ViT-Small)
print("\n[Init 1/3] Loading Metric3D v2 (ViT-Small)...")
from metric_scale.models.metric3d import Metric3DV2
metric3d_model = Metric3DV2(device="cpu")

# 2. Load Depth Anything v2 Metric (ViT-Small)
print("\n[Init 2/3] Loading Depth Anything v2 Metric (ViT-Small)...")
da_dir = os.path.join(BASE_DIR, "thirdparty", "Depth-Anything-V2", "metric_depth")
sys.path.insert(0, da_dir)
from depth_anything_v2.dpt import DepthAnythingV2
da_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
da_model = DepthAnythingV2(**da_cfg, max_depth=80)
da_ckpt = os.path.join(da_dir, "checkpoints", "depth_anything_v2_metric_vkitti_vits.pth")
da_model.load_state_dict(torch.load(da_ckpt, map_location="cpu", weights_only=False))
da_model.eval()

# 3. Load UniDepth V2 (ViT-Small/14)
print("\n[Init 3/3] Loading UniDepth V2 (ViT-Small/14)...")
unidepth_dir = os.path.join(BASE_DIR, "thirdparty", "UniDepth")
sys.path.insert(0, unidepth_dir)
from hubconf import UniDepth
unidepth_model = UniDepth(version="v2", backbone="vits14", pretrained=True).eval()

# Camera Intrinsics for our rover
# (fx, fy, cx, cy)
ROVER_K = (1041.36, 1041.86, 1019.55, 498.99)

def run_metric3d(img_bgr):
    t0 = time.perf_counter()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth = metric3d_model.predict_depth(img_rgb, intrinsics=ROVER_K)
    dt = time.perf_counter() - t0
    return depth, dt

def run_depth_anything(img_bgr):
    t0 = time.perf_counter()
    depth = da_model.infer_image(img_bgr)
    dt = time.perf_counter() - t0
    return depth, dt

def run_unidepth(img_bgr):
    t0 = time.perf_counter()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
    with torch.no_grad():
        preds = unidepth_model.infer(tensor)
    depth = preds["depth"].squeeze().cpu().numpy()
    dt = time.perf_counter() - t0
    return depth, dt

def colorize_depth(depth, min_d=1.0, max_d=50.0):
    d_norm = np.clip((depth - min_d) / (max_d - min_d), 0.0, 1.0)
    d_u8 = (255 * (1.0 - d_norm)).astype(np.uint8) # closer = brighter / hotter
    cmap = cv2.applyColorMap(d_u8, cv2.COLORMAP_MAGMA)
    return cmap

test_frames = [
    ("frame_00500.jpg", "Straight Field Road (dt=0.0s)"),
    ("frame_00580.jpg", "Sharp Rover Turn (dt=2.67s)"),
    ("frame_00800.jpg", "Distant Horizon (dt=10.0s)")
]

results = {
    "Metric3D v2": {"times": [], "near_dists": [], "medians": []},
    "Depth Anything v2": {"times": [], "near_dists": [], "medians": []},
    "UniDepth V2": {"times": [], "near_dists": [], "medians": []}
}

print("\n================== BENCHMARKING DEPTH MODELS ==================")

for fname, desc in test_frames:
    p = os.path.join(FRAMES_DIR, fname)
    im = cv2.imread(p)
    H, W = im.shape[:2]
    print(f"\n--- Testing Frame: {fname} ({desc}) ---")
    
    # 1. Metric3D
    d_m3d, t_m3d = run_metric3d(im)
    d_m3d_res = cv2.resize(d_m3d, (W, H), interpolation=cv2.INTER_LINEAR)
    c_m3d = colorize_depth(d_m3d_res)
    # Foreground patch (ground right in front of bumper: bottom 15%, center 30%)
    near_m3d = np.median(d_m3d_res[int(H*0.80):int(H*0.95), int(W*0.35):int(W*0.65)])
    med_m3d = np.median(d_m3d_res)
    results["Metric3D v2"]["times"].append(t_m3d)
    results["Metric3D v2"]["near_dists"].append(near_m3d)
    results["Metric3D v2"]["medians"].append(med_m3d)
    print(f"Metric3D v2:      Time={t_m3d*1000:.0f}ms | Bumper Ground={near_m3d:.2f}m | Median={med_m3d:.2f}m | Range=[{d_m3d_res.min():.2f}m .. {d_m3d_res.max():.2f}m]")
    
    # 2. Depth Anything v2
    d_da, t_da = run_depth_anything(im)
    d_da_res = cv2.resize(d_da, (W, H), interpolation=cv2.INTER_LINEAR)
    c_da = colorize_depth(d_da_res)
    near_da = np.median(d_da_res[int(H*0.80):int(H*0.95), int(W*0.35):int(W*0.65)])
    med_da = np.median(d_da_res)
    results["Depth Anything v2"]["times"].append(t_da)
    results["Depth Anything v2"]["near_dists"].append(near_da)
    results["Depth Anything v2"]["medians"].append(med_da)
    print(f"Depth Anything v2:Time={t_da*1000:.0f}ms | Bumper Ground={near_da:.2f}m | Median={med_da:.2f}m | Range=[{d_da_res.min():.2f}m .. {d_da_res.max():.2f}m]")
    
    # 3. UniDepth V2
    d_uni, t_uni = run_unidepth(im)
    d_uni_res = cv2.resize(d_uni, (W, H), interpolation=cv2.INTER_LINEAR)
    c_uni = colorize_depth(d_uni_res)
    near_uni = np.median(d_uni_res[int(H*0.80):int(H*0.95), int(W*0.35):int(W*0.65)])
    med_uni = np.median(d_uni_res)
    results["UniDepth V2"]["times"].append(t_uni)
    results["UniDepth V2"]["near_dists"].append(near_uni)
    results["UniDepth V2"]["medians"].append(med_uni)
    print(f"UniDepth V2:      Time={t_uni*1000:.0f}ms | Bumper Ground={near_uni:.2f}m | Median={med_uni:.2f}m | Range=[{d_uni_res.min():.2f}m .. {d_uni_res.max():.2f}m]")
    
    # Build 2x2 side-by-side comparison canvas:
    # [ Original RGB   ] [ Metric3D v2      ]
    # [ DepthAnything2 ] [ UniDepth V2       ]
    thumb_w, thumb_h = 640, 360
    im_thumb = cv2.resize(im, (thumb_w, thumb_h))
    c_m3d_thumb = cv2.resize(c_m3d, (thumb_w, thumb_h))
    c_da_thumb = cv2.resize(c_da, (thumb_w, thumb_h))
    c_uni_thumb = cv2.resize(c_uni, (thumb_w, thumb_h))
    
    # Put labels
    cv2.putText(im_thumb, f"RGB: {fname}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(c_m3d_thumb, f"Metric3D v2 ({near_m3d:.2f}m)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(c_da_thumb, f"DepthAnything2 ({near_da:.2f}m)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(c_uni_thumb, f"UniDepth V2 ({near_uni:.2f}m)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    row1 = np.hstack([im_thumb, c_m3d_thumb])
    row2 = np.hstack([c_da_thumb, c_uni_thumb])
    canvas = np.vstack([row1, row2])
    
    out_img = os.path.join(OUT_DIR, f"depth_comparison_{fname}")
    cv2.imwrite(out_img, canvas)
    print(f"--> Saved visual comparison to: {out_img}")

print("\n================== SUMMARY ==================")
for m_name in results:
    avg_t = np.mean(results[m_name]["times"]) * 1000
    avg_near = np.mean(results[m_name]["near_dists"])
    avg_med = np.mean(results[m_name]["medians"])
    print(f"{m_name:18s} | Latency: {avg_t:6.0f}ms | Avg Bumper Ground: {avg_near:5.2f}m | Avg Scene Median: {avg_med:5.2f}m")
