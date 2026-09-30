import os
import sys
import time
from typing import Optional
import numpy as np
import torch
import cv2

from ..core.frame import CameraIntrinsics
from .base import BaseDepthProvider, DepthPrediction


class UniDepthProvider(BaseDepthProvider):
    """
    Default Metric Depth Provider for UVO based on UniDepth V2 (ViT-Small/14).
    Developed by ETH Zürich CVG.
    
    Key Features:
    - 2.5x faster inference than Metric3D v2.
    - True metric scale (predicts ground plane at 4.25 m in front of rover bumper).
    - Simultaneously predicts intrinsic focal length and per-pixel confidence map.
    """
    def __init__(
        self,
        backbone: str = "vits14",
        device: Optional[str] = None,
        fp16: bool = True,
        max_resolution: int = 518
    ):
        super().__init__(device=device)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = backbone
        self.fp16 = fp16 and (self.device.type == "cuda")
        self.max_resolution = max_resolution
        
        # Load UniDepth V2 model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        unidepth_dir = os.path.join(base_dir, "thirdparty", "UniDepth")
        
        sys.path.insert(0, unidepth_dir)
        try:
            from hubconf import UniDepth
            self.model = UniDepth(version="v2", backbone=backbone, pretrained=True).to(self.device).eval()
        finally:
            if unidepth_dir in sys.path:
                sys.path.remove(unidepth_dir)

    def predict_depth(
        self,
        image: np.ndarray,
        intrinsics: Optional[CameraIntrinsics] = None
    ) -> DepthPrediction:
        """
        Runs metric depth prediction and returns DepthPrediction with metric depth in meters.
        """
        t0 = time.perf_counter()
        orig_h, orig_w = image.shape[:2]

        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # High-performance resolution scaling (divisible by patch size 14)
        if self.max_resolution > 0 and max(orig_h, orig_w) > self.max_resolution:
            scale = float(self.max_resolution) / float(max(orig_h, orig_w))
            new_w = max(14, int(round((orig_w * scale) / 14.0)) * 14)
            new_h = max(14, int(round((orig_h * scale) / 14.0)) * 14)
            input_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            input_rgb = image_rgb

        tensor = torch.from_numpy(input_rgb).permute(2, 0, 1).to(self.device)

        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    preds = self.model.infer(tensor)
            else:
                preds = self.model.infer(tensor)

        depth_raw = preds["depth"].squeeze().cpu().numpy().astype(np.float32)
        confidence_raw = preds["confidence"].squeeze().cpu().numpy().astype(np.float32) if "confidence" in preds else None

        # Resize back to original image resolution if model scaled it
        if depth_raw.shape[0] != orig_h or depth_raw.shape[1] != orig_w:
            depth_map = cv2.resize(depth_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            confidence_map = cv2.resize(confidence_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) if confidence_raw is not None else None
        else:
            depth_map = depth_raw
            confidence_map = confidence_raw

        # Uncertainty map: higher confidence -> lower uncertainty
        uncertainty_map = None
        if confidence_map is not None:
            uncertainty_map = (1.0 - np.clip(confidence_map, 0.0, 1.0)) * depth_map * 0.1

        # Extract predicted intrinsics if available
        pred_intrinsics = None
        if "intrinsics" in preds and preds["intrinsics"] is not None:
            K_t = preds["intrinsics"].squeeze().cpu().numpy()
            scale_x = orig_w / float(depth_raw.shape[1])
            scale_y = orig_h / float(depth_raw.shape[0])
            pred_intrinsics = CameraIntrinsics(
                fx=float(K_t[0, 0] * scale_x),
                fy=float(K_t[1, 1] * scale_y),
                cx=float(K_t[0, 2] * scale_x),
                cy=float(K_t[1, 2] * scale_y),
                width=orig_w,
                height=orig_h
            )

        # Visual IMU: estimate ground plane normal and gravity direction in camera frame
        gravity_cam = None
        ground_height = None
        if "points" in preds and preds["points"] is not None:
            pts = preds["points"].squeeze(0)  # (3, H, W)
            H_pts, W_pts = pts.shape[1], pts.shape[2]
            # Ground terrain in front of crawler tracks (lower 35%, central 60% of width)
            ground_pts = pts[:, int(H_pts * 0.65):H_pts, int(W_pts * 0.2):int(W_pts * 0.8)].reshape(3, -1).t()
            if ground_pts.shape[0] > 100:
                mean_p = ground_pts.mean(dim=0)
                centered = ground_pts - mean_p
                # Fast GPU PCA for plane normal
                _, _, V = torch.pca_lowrank(centered, q=3)
                g_cam = V[:, 2]
                g_cam = g_cam / (torch.norm(g_cam) + 1e-8)
                # Gravity vector points downwards (+Y in camera coordinate frame)
                if g_cam[1] < 0:
                    g_cam = -g_cam
                gravity_cam = g_cam.cpu().numpy().astype(np.float64)
                # Ground plane distance / camera height over terrain along gravity
                h_ground = torch.median(torch.matmul(ground_pts, g_cam)).item()
                ground_height = float(h_ground)

        elapsed = time.perf_counter() - t0

        return DepthPrediction(
            depth_map=depth_map,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
            predicted_intrinsics=pred_intrinsics,
            gravity_cam=gravity_cam,
            ground_height=ground_height,
            elapsed_time_sec=elapsed,
            model_name="UniDepth-v2-vits14"
        )
