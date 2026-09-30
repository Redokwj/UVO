import os
import sys
import time
import cv2
import numpy as np
import torch
import types

# Stub pytorch_lightning before anything else for EfficientLoFTR
class DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass
class DummyModule(types.ModuleType):
    __path__ = []
    def __getattr__(self, name):
        return DummyClass

for m_name in ['pytorch_lightning', 'pytorch_lightning.callbacks', 'pytorch_lightning.callbacks.model_checkpoint']:
    sys.modules[m_name] = DummyModule(m_name)

# Add thirdparty paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIFTFEAT_DIR = os.path.join(BASE_DIR, "thirdparty", "LiftFeat")
ELOFTR_DIR = os.path.join(BASE_DIR, "thirdparty", "EfficientLoFTR")
FRAMES_DIR = os.path.join(BASE_DIR, "bench_frames")
OUT_DIR = os.path.join(BASE_DIR, "bench_results")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Initialize LiftFeat
print("[Init] Loading LiftFeat (ICRA 2025)...")
sys.path.insert(0, LIFTFEAT_DIR)
from models.liftfeat_wrapper import LiftFeat, MODEL_PATH as LIFT_WEIGHTS
liftfeat_model = LiftFeat(weight=LIFT_WEIGHTS, detect_threshold=0.05)
sys.path.pop(0)

# 2. Initialize EfficientLoFTR
print("[Init] Loading EfficientLoFTR (CVPR 2024)...")
sys.path.insert(0, ELOFTR_DIR)
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter
eloftr_cfg = deepcopy(full_default_cfg)
eloftr_model = LoFTR(config=eloftr_cfg)
eloftr_ckpt = torch.load(os.path.join(ELOFTR_DIR, "weights", "eloftr_outdoor.ckpt"), map_location="cpu", weights_only=False)
eloftr_model.load_state_dict(eloftr_ckpt["state_dict"])
eloftr_model = reparameter(eloftr_model).eval()
sys.path.pop(0)

def match_liftfeat(im1, im2):
    t0 = time.perf_counter()
    mkpts0, mkpts1 = liftfeat_model.match_liftfeat(im1, im2)
    dt = time.perf_counter() - t0
    return mkpts0, mkpts1, dt

def match_eloftr(im1, im2):
    t0 = time.perf_counter()
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    h, w = g1.shape
    new_w, new_h = (w // 32) * 32, (h // 32) * 32
    g1_res = cv2.resize(g1, (new_w, new_h))
    g2_res = cv2.resize(g2, (new_w, new_h))
    
    scale_x = w / new_w
    scale_y = h / new_h
    
    t1 = torch.from_numpy(g1_res)[None, None].float() / 255.0
    t2 = torch.from_numpy(g2_res)[None, None].float() / 255.0
    batch = {'image0': t1, 'image1': t2}
    with torch.no_grad():
        eloftr_model(batch)
    
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    
    # Scale back to original resolution
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y
    
    dt = time.perf_counter() - t0
    return mkpts0, mkpts1, dt

def eval_magsac(kpts0, kpts1):
    if len(kpts0) < 8:
        return 0, 0.0, None
    F, mask = cv2.findFundamentalMat(kpts0, kpts1, cv2.USAC_MAGSAC, 3.0, 0.999, 2000)
    if mask is None:
        return 0, 0.0, None
    inliers = int(mask.sum())
    ratio = (inliers / len(kpts0)) * 100.0
    return inliers, ratio, mask.ravel().astype(bool)

def draw_matches_side_by_side(im1, im2, kpts0, kpts1, mask, title, out_path):
    h, w = im1.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = im1
    canvas[:, w:] = im2
    
    # Pick a random subset of inliers to draw cleanly
    inlier_indices = np.where(mask)[0] if mask is not None else []
    if len(inlier_indices) > 150:
        np.random.seed(42)
        inlier_indices = np.random.choice(inlier_indices, 150, replace=False)
        
    for idx in inlier_indices:
        pt0 = (int(round(kpts0[idx][0])), int(round(kpts0[idx][1])))
        pt1 = (int(round(kpts1[idx][0])) + w, int(round(kpts1[idx][1])))
        cv2.line(canvas, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 3, (0, 0, 255), -1)
        cv2.circle(canvas, pt1, 3, (0, 0, 255), -1)
        
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(out_path, canvas)

scenarios = [
    ("Small Baseline (dt=0.07s)", "frame_00500.jpg", "frame_00502.jpg"),
    ("Medium Baseline (dt=0.50s)", "frame_00500.jpg", "frame_00515.jpg"),
    ("Wide Baseline / Turn (dt=2.67s)", "frame_00500.jpg", "frame_00580.jpg"),
    ("Long Baseline / Loop (dt=10.0s)", "frame_00500.jpg", "frame_00800.jpg")
]

results = {"LiftFeat": [], "EfficientLoFTR": []}

print("\n================== BENCHMARK EXECUTION ==================")
for sc_name, f1, f2 in scenarios:
    p1 = os.path.join(FRAMES_DIR, f1)
    p2 = os.path.join(FRAMES_DIR, f2)
    im1 = cv2.imread(p1)
    im2 = cv2.imread(p2)
    print(f"\n--- Scenario: {sc_name} ---")
    
    # 1. LiftFeat
    k0, k1, dt = match_liftfeat(im1, im2)
    inliers, ratio, mask = eval_magsac(k0, k1)
    print(f"LiftFeat (ICRA 2025): Matches={len(k0)}, Inliers={inliers}, Ratio={ratio:.1f}%, Time={dt*1000:.0f}ms")
    results["LiftFeat"].append({"matches": len(k0), "inliers": inliers, "ratio": ratio, "time": dt})
    draw_matches_side_by_side(im1, im2, k0, k1, mask, f"LiftFeat: {inliers}/{len(k0)} ({ratio:.1f}%) in {dt*1000:.0f}ms", 
                              os.path.join(OUT_DIR, f"liftfeat_{f1[:11]}_{f2[:11]}.jpg"))
    
    # 2. EfficientLoFTR
    k0, k1, dt = match_eloftr(im1, im2)
    inliers, ratio, mask = eval_magsac(k0, k1)
    print(f"EfficientLoFTR (CVPR 2024): Matches={len(k0)}, Inliers={inliers}, Ratio={ratio:.1f}%, Time={dt*1000:.0f}ms")
    results["EfficientLoFTR"].append({"matches": len(k0), "inliers": inliers, "ratio": ratio, "time": dt})
    draw_matches_side_by_side(im1, im2, k0, k1, mask, f"EfficientLoFTR: {inliers}/{len(k0)} ({ratio:.1f}%) in {dt*1000:.0f}ms", 
                              os.path.join(OUT_DIR, f"eloftr_{f1[:11]}_{f2[:11]}.jpg"))

print("\n================== SUMMARY ==================")
for m_name in ["LiftFeat", "EfficientLoFTR"]:
    avg_m = np.mean([r["matches"] for r in results[m_name]])
    avg_inl = np.mean([r["inliers"] for r in results[m_name]])
    avg_rat = np.mean([r["ratio"] for r in results[m_name]])
    avg_t = np.mean([r["time"] for r in results[m_name]])
    print(f"{m_name:16s} | Avg Matches: {avg_m:6.0f} | Avg Inliers: {avg_inl:6.0f} | Avg Ratio: {avg_rat:5.1f}% | Avg Latency: {avg_t*1000:5.0f}ms")
