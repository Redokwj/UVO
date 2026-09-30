"""
Online 3DGS Dense SLAM Engine.

Incrementally constructs a photorealistic 3D Gaussian Splatting digital twin
parallel to UVO metric tracking. Enables novel viewpoint rendering (Virtual Drone Orbit)
and exports 3DGS scenes to standard PLY formats.
"""

from typing import Optional, Tuple, Dict, Any, List
import time
import numpy as np
import cv2
import torch
import torch.optim as optim

from uvo.core.frame import Keyframe, CameraIntrinsics
from uvo.core.geometry import SE3
from .gaussian_model import GaussianModel, GaussianConfig
from .renderer import GaussianRenderer, CameraView


class Online3DGSSLAM:
    """
    Online Dense 3D Gaussian Splatting SLAM Module.
    """
    def __init__(
        self,
        config: Optional[GaussianConfig] = None,
        lr_xyz: float = 1e-3,
        lr_features: float = 5e-3,
        lr_opacity: float = 1e-2,
        lr_scaling: float = 3e-3,
        device: str = "cuda"
    ):
        self.config = config or GaussianConfig()
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        
        self.model = GaussianModel(config=self.config, device=str(self.device))
        self.renderer = GaussianRenderer(device=str(self.device))
        
        self.lr_xyz = lr_xyz
        self.lr_features = lr_features
        self.lr_opacity = lr_opacity
        self.lr_scaling = lr_scaling
        
        self.optimizer: Optional[optim.Optimizer] = None
        self.keyframe_history: List[Keyframe] = []
        self.current_rover_pos: np.ndarray = np.zeros(3)

    def _init_optimizer(self):
        """Initializes PyTorch Adam optimizer for Gaussian parameters."""
        params = [
            {"params": [self.model._xyz], "lr": self.lr_xyz, "name": "xyz"},
            {"params": [self.model._features_dc], "lr": self.lr_features, "name": "features"},
            {"params": [self.model._opacity], "lr": self.lr_opacity, "name": "opacity"},
            {"params": [self.model._scaling], "lr": self.lr_scaling, "name": "scaling"}
        ]
        self.optimizer = optim.Adam(params)

    def process_keyframe(
        self,
        keyframe: Keyframe,
        num_opt_steps: int = 3,
        subsample_step: int = 4
    ) -> Dict[str, Any]:
        """
        Integrates a new metric keyframe from UVO:
        1. Backprojects UniDepth metric depth into 3D Gaussians in world space.
        2. Applies spatial voxel filtering to avoid redundancy.
        3. Runs fast online photometric refinement.
        """
        t0 = time.perf_counter()
        
        # Extract pose
        if hasattr(keyframe, 'pose_wc'):
            R_wc = keyframe.pose_wc.R
            t_wc = keyframe.pose_wc.t
            R_cw = keyframe.pose_cw.R
            t_cw = keyframe.pose_cw.t
        elif hasattr(keyframe, 'pose'):
            R_wc = keyframe.pose.R
            t_wc = keyframe.pose.t
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
        else:
            R_wc = np.eye(3, dtype=np.float64)
            t_wc = np.zeros(3, dtype=np.float64)
            R_cw = np.eye(3, dtype=np.float64)
            t_cw = np.zeros(3, dtype=np.float64)
        
        self.current_rover_pos = t_wc.copy()
        self.keyframe_history.append(keyframe)
        
        img_bgr = keyframe.frame.image
        depth_map = keyframe.depth_map
        intrinsics = keyframe.frame.intrinsics

        # 1. Incrementally add new 3D Gaussians from depth
        n_added = 0
        if depth_map is not None:
            n_added = self.model.add_from_depth(
                image_bgr=img_bgr,
                depth_map=depth_map,
                intrinsics=intrinsics,
                R_cw=R_cw,
                t_cw=t_cw,
                subsample_step=subsample_step
            )

        # 2. Online Photometric Optimization
        opt_loss = 0.0
        if self.model.num_gaussians > 0 and num_opt_steps > 0:
            self._init_optimizer()
            
            # Target image in PyTorch [0, 1]
            h_tgt, w_tgt = img_bgr.shape[:2]
            # Downsample target for fast optimization
            scale_opt = 0.5
            h_opt, w_opt = int(h_tgt * scale_opt), int(w_tgt * scale_opt)
            img_rgb_small = cv2.resize(img_bgr[:, :, ::-1], (w_opt, h_opt)).astype(np.float32) / 255.0
            tgt_tensor = torch.from_numpy(img_rgb_small).to(self.device)

            opt_view = CameraView(
                R_cw=R_cw.astype(np.float32),
                t_cw=t_cw.astype(np.float32),
                fx=float(intrinsics.fx * scale_opt),
                fy=float(intrinsics.fy * scale_opt),
                cx=float(intrinsics.cx * scale_opt),
                cy=float(intrinsics.cy * scale_opt),
                width=w_opt,
                height=h_opt
            )

            for _ in range(num_opt_steps):
                self.optimizer.zero_grad()
                rendered = self.renderer.render(self.model, opt_view)
                loss = torch.mean(torch.abs(rendered - tgt_tensor))
                loss.backward()
                self.optimizer.step()
                opt_loss = float(loss.item())

        elapsed = time.perf_counter() - t0

        return {
            "num_gaussians": self.model.num_gaussians,
            "added_this_kf": n_added,
            "opt_loss": opt_loss,
            "elapsed_ms": elapsed * 1000.0
        }

    def render_orbit_view(
        self,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 30.0,
        distance: float = 5.0,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """
        Renders a virtual "God's Eye" / Drone Orbit perspective centered around the rover.
        Returns BGR numpy image for OpenCV / display.
        """
        view = CameraView.create_orbit_view(
            target_xyz=self.current_rover_pos,
            distance=distance,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            fov_deg=65.0,
            width=width,
            height=height
        )
        with torch.no_grad():
            render_tensor = self.renderer.render(self.model, view)
            render_rgb = (render_tensor.cpu().numpy() * 255.0).astype(np.uint8)
            render_bgr = render_rgb[:, :, ::-1] # RGB to BGR
            
        return render_bgr

    def export_ply(self, filepath: str):
        """Exports the 3DGS scene to standard PLY format."""
        self.model.export_ply(filepath)
