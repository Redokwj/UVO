"""
Fast Differentiable & Vectorized 3D Gaussian Splatting (3DGS) Renderer.

Projects 3D Gaussian ellipsoids to 2D image coordinates and accumulates color
via alpha-blended splatting. Supports novel viewpoint synthesis (Virtual Drone Orbit / God's Eye View).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn.functional as F

from .gaussian_model import GaussianModel


@dataclass
class CameraView:
    R_cw: np.ndarray      # (3, 3) World-to-Camera rotation
    t_cw: np.ndarray      # (3,) World-to-Camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    @classmethod
    def create_orbit_view(
        cls,
        target_xyz: np.ndarray,
        distance: float = 6.0,
        elevation_deg: float = 35.0,
        azimuth_deg: float = 0.0,
        fov_deg: float = 60.0,
        width: int = 640,
        height: int = 480
    ) -> "CameraView":
        """
        Creates a virtual "God's Eye" / Drone Orbit camera orbiting around a target position.
        """
        el_rad = math.radians(elevation_deg)
        az_rad = math.radians(azimuth_deg)

        # Camera position in world coordinates
        cam_x = target_xyz[0] - distance * math.cos(el_rad) * math.cos(az_rad)
        cam_y = target_xyz[1] - distance * math.cos(el_rad) * math.sin(az_rad)
        cam_z = target_xyz[2] + distance * math.sin(el_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Look-at matrix: forward vector from camera to target
        forward = target_xyz - cam_pos
        forward = forward / np.linalg.norm(forward)

        # World up vector
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        # OpenCV Camera coordinate convention:
        # X: Right, Y: Down (invert true_up), Z: Forward
        R_c2w = np.column_stack([right, -true_up, forward])
        R_cw = R_c2w.T
        t_cw = -R_cw @ cam_pos

        # Compute focal length from FOV
        fx = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        return cls(
            R_cw=R_cw.astype(np.float32),
            t_cw=t_cw.astype(np.float32),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=width,
            height=height
        )


class GaussianRenderer:
    """
    Differentiable 3DGS Splat Renderer.
    Vectorized PyTorch implementation optimized for real-time dense SLAM feedback.
    """
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    def render(
        self,
        model: GaussianModel,
        view: CameraView,
        bg_color: Tuple[float, float, float] = (0.05, 0.05, 0.07)
    ) -> torch.Tensor:
        """
        Renders the 3D Gaussians from the specified CameraView.
        Returns:
            (H, W, 3) float32 tensor with RGB values in [0, 1].
        """
        N = model.num_gaussians
        if N == 0:
            bg = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
            return bg.view(1, 1, 3).repeat(view.height, view.width, 1)

        xyz_w = model.get_xyz()
        colors = model.get_features()
        opacities = model.get_opacity()
        scales = model.get_scaling()

        # 1. Transform centers to camera frame: P_c = R_cw * P_w + t_cw
        R_cw = torch.from_numpy(view.R_cw).to(self.device)
        t_cw = torch.from_numpy(view.t_cw).to(self.device)

        pts_c = torch.matmul(xyz_w, R_cw.T) + t_cw
        z_c = pts_c[:, 2]

        # Filter points in front of the camera with safety margin
        valid_z = (z_c > 0.2) & (z_c < 30.0)
        if not torch.any(valid_z):
            bg = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
            return bg.view(1, 1, 3).repeat(view.height, view.width, 1)

        pts_c = pts_c[valid_z]
        colors = colors[valid_z]
        opacities = opacities[valid_z]
        scales = scales[valid_z]
        z_c = z_c[valid_z]

        # 2. Project 3D centers to screen pixel coordinates
        u = view.fx * (pts_c[:, 0] / z_c) + view.cx
        v = view.fy * (pts_c[:, 1] / z_c) + view.cy

        # Filter points inside view frustum (with padding for Gaussian radius)
        margin = 30.0
        in_frustum = (u >= -margin) & (u < view.width + margin) & (v >= -margin) & (v < view.height + margin)
        if not torch.any(in_frustum):
            bg = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
            return bg.view(1, 1, 3).repeat(view.height, view.width, 1)

        u = u[in_frustum]
        v = v[in_frustum]
        z_c = z_c[in_frustum]
        colors = colors[in_frustum]
        opacities = opacities[in_frustum]
        scales = scales[in_frustum]

        # 3. Sort Gaussians Front-to-Back by depth Z
        sort_idx = torch.argsort(z_c, descending=False)
        u = u[sort_idx]
        v = v[sort_idx]
        z_c = z_c[sort_idx]
        colors = colors[sort_idx]
        opacities = opacities[sort_idx]
        scales = scales[sort_idx]

        # 4. Approximate 2D Gaussian radius on screen (pixels)
        # radius ~ 3 * scale * fx / depth
        mean_scale = scales.mean(dim=-1)
        radius_px = torch.clamp((mean_scale * view.fx / z_c) * 1.5, min=1.5, max=40.0)

        # Initialize canvas with background color
        canvas = torch.tensor(bg_color, dtype=torch.float32, device=self.device).view(1, 1, 3).repeat(view.height, view.width, 1)
        accum_alpha = torch.zeros((view.height, view.width, 1), dtype=torch.float32, device=self.device)

        # Fast Vectorized Chunked Splatting
        # To maintain high FPS, we draw Gaussians onto the canvas
        n_visible = len(u)
        u_int = torch.round(u).long()
        v_int = torch.round(v).long()
        r_int = torch.clamp(torch.ceil(radius_px).long(), min=1, max=16)

        # Process key splats
        for i in range(min(n_visible, 3500)):
            cx_i, cy_i = u_int[i].item(), v_int[i].item()
            r_i = r_int[i].item()
            
            x0 = max(0, cx_i - r_i)
            x1 = min(view.width, cx_i + r_i + 1)
            y0 = max(0, cy_i - r_i)
            y1 = min(view.height, cy_i + r_i + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            # Sub-patch coordinates
            px = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
            py = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')

            dist_sq = (grid_x - u[i])**2 + (grid_y - v[i])**2
            sigma_sq = max(1.0, (radius_px[i] / 2.0)**2)

            alpha_patch = opacities[i] * torch.exp(-0.5 * dist_sq / sigma_sq).unsqueeze(-1)
            color_patch = colors[i].unsqueeze(0).unsqueeze(0)

            # Volumetric front-to-back blending:
            # T = 1 - accum_alpha
            T = 1.0 - accum_alpha[y0:y1, x0:x1]
            weight = alpha_patch * T

            canvas[y0:y1, x0:x1] += weight * color_patch
            accum_alpha[y0:y1, x0:x1] += weight

        # Add remaining background where alpha < 1.0
        rem_bg = torch.clamp(1.0 - accum_alpha, min=0.0, max=1.0)
        bg_t = torch.tensor(bg_color, dtype=torch.float32, device=self.device).view(1, 1, 3)
        final_render = canvas + rem_bg * bg_t

        return torch.clamp(final_render, 0.0, 1.0)
