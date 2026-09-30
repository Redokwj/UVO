import os
import sys
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import cv2

from ..core.geometry import SE3
from ..core.frame import Frame
from .base import BaseTracker, TrackingResult


class LiftFeatTracker(BaseTracker):
    """
    SOTA Feature Tracker based on LiftFeat (ICRA 2025).
    Superior robustness on wide baselines, acute turns, and low-texture offroad terrain.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        top_k: int = 2048,
        detect_threshold: float = 0.05,
        min_cossim: float = 0.50,
        keyframe_parallax_threshold: float = 28.0,
        min_inliers: int = 35,
        fp16: bool = True
    ):
        super().__init__(
            keyframe_parallax_threshold=keyframe_parallax_threshold,
            min_inliers=min_inliers
        )
        self.top_k = top_k
        self.detect_threshold = detect_threshold
        self.min_cossim = min_cossim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and (self.device.type == "cuda")
        
        # Load LiftFeat model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lift_dir = os.path.join(base_dir, "thirdparty", "LiftFeat")
        if weights_path is None:
            weights_path = os.path.join(lift_dir, "weights", "LiftFeat.pth")
            
        sys.path.insert(0, lift_dir)
        try:
            from models.liftfeat_wrapper import LiftFeat
            self.model = LiftFeat(weight=weights_path, top_k=top_k, detect_threshold=detect_threshold)
            self.model.eval()
        finally:
            if lift_dir in sys.path:
                sys.path.remove(lift_dir)

    def extract_features(self, frame: Frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts LiftFeat keypoints, descriptors, and scores for a frame, caching results in Frame.
        """
        if "liftfeat" in frame.feature_cache:
            c = frame.feature_cache["liftfeat"]
            return c["keypoints"], c["descriptors"], c["scores"]
            
        img = frame.image
        # LiftFeat expects BGR 3-channel image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    data = self.model.extract(img)
            else:
                data = self.model.extract(img)
            
        kpts = data["keypoints"].cpu().numpy().astype(np.float32)
        desc = data["descriptors"].cpu().numpy().astype(np.float32)
        scores = data["scores"].cpu().numpy().astype(np.float32)

        frame.feature_cache["liftfeat"] = {
            "keypoints": kpts,
            "descriptors": desc,
            "scores": scores
        }
        frame.keypoints = kpts
        frame.descriptors = desc
        frame.scores = scores
        return kpts, desc, scores

    def match_descriptors(
        self,
        desc0: np.ndarray,
        desc1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutual nearest neighbor matching with minimum cosine similarity gating.
        """
        d0 = torch.from_numpy(desc0).to(self.device)
        d1 = torch.from_numpy(desc1).to(self.device)

        # Normalize descriptors to unit hypersphere
        d0 = torch.nn.functional.normalize(d0, p=2, dim=1)
        d1 = torch.nn.functional.normalize(d1, p=2, dim=1)

        cossim = d0 @ d1.t()
        cossim_t = d1 @ d0.t()

        match0 = cossim.argmax(dim=1)
        match1 = cossim_t.argmax(dim=1)

        idx0 = torch.arange(len(match0), device=self.device)
        mutual = (match1[match0] == idx0)

        if self.min_cossim > 0:
            max_sim = cossim.max(dim=1).values
            valid = mutual & (max_sim >= self.min_cossim)
        else:
            valid = mutual

        matched_idx0 = idx0[valid].cpu().numpy()
        matched_idx1 = match0[valid].cpu().numpy()

        return kpts0[matched_idx0], kpts1[matched_idx1]

    def track(self, frame0: Frame, frame1: Frame) -> TrackingResult:
        """
        Tracks features between frame0 and frame1 using LiftFeat + MAGSAC++.
        """
        t0 = time.perf_counter()

        # 1. Feature extraction (cached if already present)
        kpts0, desc0, _ = self.extract_features(frame0)
        kpts1, desc1, _ = self.extract_features(frame1)

        if len(kpts0) == 0 or len(kpts1) == 0:
            return TrackingResult(
                matched_kpts0=np.empty((0, 2), dtype=np.float32),
                matched_kpts1=np.empty((0, 2), dtype=np.float32),
                inlier_mask=np.zeros(0, dtype=bool),
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                mean_parallax_px=0.0,
                relative_pose=None,
                is_keyframe_candidate=False,
                elapsed_time_sec=time.perf_counter() - t0,
                tracker_name="LiftFeat"
            )

        # 2. Descriptor matching
        mkpts0, mkpts1 = self.match_descriptors(
            desc0, desc1,
            kpts0, kpts1
        )

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
                tracker_name="LiftFeat"
            )

        # 3. Essential matrix & Relative pose recovery
        rel_pose, inlier_mask, num_inliers = self.estimate_relative_pose_2d2d(
            mkpts0, mkpts1, frame1.intrinsics
        )

        inlier_ratio = (num_inliers / num_matches * 100.0) if num_matches > 0 else 0.0
        mean_parallax = self.compute_parallax(mkpts0, mkpts1, inlier_mask)

        # 4. Keyframe candidacy check (large parallax or inlier depletion)
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
            tracker_name="LiftFeat"
        )
