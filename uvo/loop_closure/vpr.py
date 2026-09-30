import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from ..core.frame import Keyframe


class VPREngine:
    """
    Visual Place Recognition (VPR) Engine based on DINOv2 (ViT-Small/14).
    Produces 384-dimensional global descriptors invariant to lighting, seasonal changes,
    and textureless natural terrain.
    """
    def __init__(self, device: Optional[str] = None, img_size: int = 224, fp16: bool = True):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = (img_size // 14) * 14 # Ensure multiple of patch size 14
        self.fp16 = fp16 and (self.device.type == "cuda")
        
        # Load DINOv2 ViT-Small
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if self.fp16:
            self.model = self.model.half()
        self.model.to(self.device).eval()

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    @torch.no_grad()
    def extract_descriptor(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a L2-normalized 384-d global embedding for place recognition.
        """
        if image_bgr.ndim == 2:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half() / 255.0
        else:
            tensor = tensor.float() / 255.0
        tensor = (tensor - self.mean) / self.std

        # Extract CLS token from DINOv2
        feat = self.model(tensor)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        return feat_norm.squeeze(0).float().cpu().numpy().astype(np.float32)


class VPRDatabase:
    """
    In-memory spatial-temporal database for fast place recognition matching.
    """
    def __init__(self, vpr_engine: Optional[VPREngine] = None):
        self.vpr_engine = vpr_engine or VPREngine()
        self.keyframe_ids: List[int] = []
        self.timestamps: List[float] = []
        self.accumulated_distances: List[float] = []
        self.descriptors: List[np.ndarray] = []

    def add_keyframe(self, kf: Keyframe) -> np.ndarray:
        """
        Computes and records global descriptor for a keyframe.
        """
        if kf.vpr_descriptor is None:
            desc = self.vpr_engine.extract_descriptor(kf.frame.image)
            kf.vpr_descriptor = desc
        else:
            desc = kf.vpr_descriptor

        self.keyframe_ids.append(kf.frame_id)
        self.timestamps.append(kf.timestamp)
        self.accumulated_distances.append(getattr(kf, 'accumulated_distance_m', 0.0))
        self.descriptors.append(desc)
        return desc

    def query(
        self,
        query_desc: np.ndarray,
        current_kf_id: int,
        current_timestamp: float = 0.0,
        current_accum_dist_m: float = 0.0,
        min_gap_distance_m: float = 75.0,
        covisibility_exclude_ids: Optional[set] = None,
        min_gap_frames: int = 15,
        min_gap_time_sec: float = 0.0,
        top_k: int = 3,
        min_similarity: float = 0.72
    ) -> List[Tuple[int, float]]:
        """
        Searches database for candidate loop closures matching query descriptor.
        Applies speed-independent accumulated distance and covisibility graph filters.
        Returns: [(candidate_kf_id, cosine_similarity), ...]
        """
        if len(self.descriptors) == 0:
            return []

        # Stack descriptors matrix (N, 384)
        db_matrix = np.array(self.descriptors)
        # Cosine similarity is dot product of unit-norm vectors
        sims = db_matrix @ query_desc.flatten()

        candidates = []
        for idx, sim in enumerate(sims):
            kf_id = self.keyframe_ids[idx]
            kf_time = self.timestamps[idx]
            kf_dist = self.accumulated_distances[idx] if idx < len(self.accumulated_distances) else 0.0

            # 1. Covisibility exclusion (active local BA window / neighbors)
            if covisibility_exclude_ids and kf_id in covisibility_exclude_ids:
                continue

            # 2. Keyframe index exclusion
            if abs(kf_id - current_kf_id) < min_gap_frames:
                continue

            # 3. Speed-independent accumulated distance filter along trajectory arc
            if min_gap_distance_m > 0.0:
                if abs(current_accum_dist_m - kf_dist) < min_gap_distance_m:
                    continue

            # 4. Optional temporal exclusion
            if min_gap_time_sec > 0.0 and abs(current_timestamp - kf_time) < min_gap_time_sec:
                continue

            if sim >= min_similarity:
                candidates.append((kf_id, float(sim)))

        # Sort descending by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

