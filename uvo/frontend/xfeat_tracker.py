import time
from typing import Optional, Tuple
import numpy as np
import torch
import cv2

from ..core.geometry import SE3
from ..core.frame import Frame
from .base import BaseTracker, TrackingResult


class XFeatTracker(BaseTracker):
    """
    Real-time lightweight Front-End Visual Tracker using XFeat (CVPR 2024).
    Targeted for 60+ FPS high-rate frame-to-frame odometry tracking.
    """
    def __init__(
        self,
        top_k: int = 2048,
        min_cossim: float = 0.70,
        keyframe_parallax_threshold: float = 30.0,
        min_inliers: int = 40,
        fp16: bool = True
    ):
        super().__init__(
            keyframe_parallax_threshold=keyframe_parallax_threshold,
            min_inliers=min_inliers
        )
        self.top_k = top_k
        self.min_cossim = min_cossim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and (self.device.type == "cuda")
        
        # Load XFeat model from PyTorch Hub
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=top_k)
        self.model.eval()

    def extract_features(self, frame: Frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts XFeat keypoints, descriptors, and scores, caching them in Frame.
        """
        if "xfeat" in frame.feature_cache:
            c = frame.feature_cache["xfeat"]
            return c["keypoints"], c["descriptors"], c["scores"]

        img = frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    res = self.model.detectAndCompute(img, top_k=self.top_k)
            else:
                res = self.model.detectAndCompute(img, top_k=self.top_k)
            if res and len(res) > 0:
                data = res[0]
                kpts = data["keypoints"].cpu().numpy().astype(np.float32)
                desc = data["descriptors"].cpu().numpy().astype(np.float32)
                scores = data["scores"].cpu().numpy().astype(np.float32)
            else:
                kpts = np.empty((0, 2), dtype=np.float32)
                desc = np.empty((0, 64), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)

        frame.feature_cache["xfeat"] = {
            "keypoints": kpts,
            "descriptors": desc,
            "scores": scores
        }
        frame.keypoints = kpts
        frame.descriptors = desc
        frame.scores = scores
        return kpts, desc, scores

    def match_features(
        self,
        desc0: np.ndarray,
        desc1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs mutual nearest neighbor matching between cached descriptors.
        """
        if len(kpts0) == 0 or len(kpts1) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        d0 = torch.from_numpy(desc0).to(self.device)
        d1 = torch.from_numpy(desc1).to(self.device)

        with torch.no_grad():
            idx0, idx1 = self.model.match(d0, d1, min_cossim=self.min_cossim)

        if isinstance(idx0, torch.Tensor):
            idx0 = idx0.cpu().numpy()
        if isinstance(idx1, torch.Tensor):
            idx1 = idx1.cpu().numpy()

        mkpts0 = kpts0[idx0]
        mkpts1 = kpts1[idx1]
        return mkpts0, mkpts1

    def track(self, frame0: Frame, frame1: Frame) -> TrackingResult:
        """
        Tracks features between frame0 and frame1 using XFeat + MAGSAC++.
        """
        t0 = time.perf_counter()

        kpts0, desc0, _ = self.extract_features(frame0)
        kpts1, desc1, _ = self.extract_features(frame1)

        mkpts0, mkpts1 = self.match_features(desc0, desc1, kpts0, kpts1)
        num_matches = len(mkpts0)

        if num_matches < 8:
            return TrackingResult(
                matched_kpts0=mkpts0,
                matched_kpts1=mkpts1,
                inlier_mask=np.zeros(num_matches, dtype=bool),
                num_matches=num_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                mean_parallax_px=0.0,
                relative_pose=None,
                is_keyframe_candidate=False,
                elapsed_time_sec=time.perf_counter() - t0,
                tracker_name="XFeat"
            )

        rel_pose, inlier_mask, num_inliers = self.estimate_relative_pose_2d2d(
            mkpts0, mkpts1, frame1.intrinsics
        )

        inlier_ratio = (num_inliers / num_matches * 100.0) if num_matches > 0 else 0.0
        mean_parallax = self.compute_parallax(mkpts0, mkpts1, inlier_mask)

        is_keyframe = (mean_parallax >= self.keyframe_parallax_threshold) or (num_inliers < self.min_inliers * 1.5)
        elapsed = time.perf_counter() - t0

        return TrackingResult(
            matched_kpts0=mkpts0,
            matched_kpts1=mkpts1,
            inlier_mask=inlier_mask,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_parallax_px=mean_parallax,
            relative_pose=rel_pose,
            is_keyframe_candidate=is_keyframe,
            elapsed_time_sec=elapsed,
            tracker_name="XFeat"
        )
