"""
3D Gaussian Splatting (3DGS) Representation for Online Dense SLAM.

Manages 3D Gaussians (positions, 3D covariance, rotations, opacities, and colors),
handles incremental initialization from metric depth, spatial voxel pruning,
and standard PLY export for WebGL/3D viewer inspection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GaussianConfig:
    init_opacity: float = 0.85
    min_opacity: float = 0.05
    max_scale: float = 0.35        # Max radius of Gaussian in meters
    voxel_filter_size: float = 0.06 # Spatial downsampling grid (6 cm)
    max_depth_m: float = 12.0      # Ignore points beyond 12 meters
    min_depth_m: float = 0.4       # Ignore points closer than 40 cm


class GaussianModel(nn.Module):
    """
    Parametric 3D Gaussian Scene Representation.
    Supports online dynamic growth as the robot explores new areas.
    """
    def __init__(self, config: Optional[GaussianConfig] = None, device: str = "cuda"):
        super().__init__()
        self.config = config or GaussianConfig()
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        
        # Trainable parameters (initialized empty)
        self._xyz = nn.Parameter(torch.empty(0, 3, device=self.device, dtype=torch.float32))
        self._features_dc = nn.Parameter(torch.empty(0, 3, device=self.device, dtype=torch.float32))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=self.device, dtype=torch.float32))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=self.device, dtype=torch.float32))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=self.device, dtype=torch.float32))

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_features(self) -> torch.Tensor:
        return torch.sigmoid(self._features_dc)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    def get_covariance(self) -> torch.Tensor:
        """
        Computes 3D covariance matrix Sigma = R * S * S^T * R^T for each Gaussian.
        Returns: (N, 3, 3) tensor.
        """
        R = self.quaternion_to_matrix(self.get_rotation())
        S = torch.diag_embed(self.get_scaling())
        M = torch.bmm(R, S)
        cov3D = torch.bmm(M, M.transpose(1, 2))
        return cov3D

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Converts normalized quaternions [w, x, y, z] to 3x3 rotation matrices.
        """
        w, x, y, z = quaternions.unbind(-1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack([
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ], -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def add_from_depth(
        self,
        image_bgr: np.ndarray,
        depth_map: np.ndarray,
        intrinsics,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        subsample_step: int = 4
    ) -> int:
        """
        Incrementally adds new 3D Gaussians from a calibrated RGB-D Keyframe.
        
        Args:
            image_bgr: (H, W, 3) uint8 image.
            depth_map: (H, W) float32 metric depth in meters.
            intrinsics: CameraIntrinsics or (fx, fy, cx, cy).
            R_cw, t_cw: World-to-Camera camera pose (P_c = R_cw * P_w + t_cw).
            subsample_step: Pixel stride to maintain real-time budget.
        Returns:
            Number of newly added Gaussians.
        """
        h, w = depth_map.shape[:2]
        if hasattr(intrinsics, 'fx'):
            fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
        else:
            fx, fy, cx, cy = intrinsics[:4]

        # Subsample grid
        v_grid, u_grid = np.mgrid[0:h:subsample_step, 0:w:subsample_step]
        u = u_grid.flatten()
        v = v_grid.flatten()
        d = depth_map[v, u]

        # Valid depth mask
        valid = (d >= self.config.min_depth_m) & (d <= self.config.max_depth_m)
        u, v, d = u[valid], v[valid], d[valid]
        if len(d) == 0:
            return 0

        # Unproject to camera frame
        x_c = (u - cx) * d / fx
        y_c = (v - cy) * d / fy
        z_c = d
        pts_c = np.column_stack([x_c, y_c, z_c]) # (N, 3)

        # Camera to World transform: P_w = R_wc * P_c + t_wc, where R_wc = R_cw.T, t_wc = -R_cw.T @ t_cw
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        pts_w = pts_c @ R_wc.T + t_wc

        # Sample RGB colors (convert BGR -> RGB in [0, 1])
        colors_rgb = image_bgr[v, u][:, ::-1].astype(np.float32) / 255.0

        # Spatial Voxel Grid Filtering: prevent adding points where Gaussians already exist
        if self.num_gaussians > 0:
            existing_pts = self._xyz.detach().cpu().numpy()
            voxel_size = self.config.voxel_filter_size
            
            # Simple voxel hash
            existing_voxels = set(map(tuple, np.floor(existing_pts / voxel_size).astype(np.int32)))
            new_voxels = np.floor(pts_w / voxel_size).astype(np.int32)
            
            keep_mask = np.array([tuple(vx) not in existing_voxels for vx in new_voxels], dtype=bool)
            pts_w = pts_w[keep_mask]
            colors_rgb = colors_rgb[keep_mask]
            d = d[keep_mask]

        n_new = len(pts_w)
        if n_new == 0:
            return 0

        # Calculate initial scale proportional to distance from camera (pixel footprint in 3D)
        # s = depth / focal_length * stride
        scale_val = (d / fx * subsample_step * 0.8).clip(0.01, self.config.max_scale)
        new_scaling = np.log(np.repeat(scale_val[:, None], 3, axis=1)).astype(np.float32)

        # Initial identity rotation (w=1, x=0, y=0, z=0)
        new_rotation = np.zeros((n_new, 4), dtype=np.float32)
        new_rotation[:, 0] = 1.0

        # Initial opacity logit: logit(p) = log(p / (1 - p))
        p_init = np.clip(self.config.init_opacity, 1e-4, 1.0 - 1e-4)
        new_opacity = np.full((n_new, 1), np.log(p_init / (1.0 - p_init)), dtype=np.float32)

        # Color logit: logit(color)
        c_clipped = np.clip(colors_rgb, 1e-4, 1.0 - 1e-4)
        new_features = np.log(c_clipped / (1.0 - c_clipped)).astype(np.float32)

        # Append to PyTorch parameters
        t_pts_w = torch.from_numpy(pts_w.astype(np.float32)).to(self.device)
        t_features = torch.from_numpy(new_features).to(self.device)
        t_scaling = torch.from_numpy(new_scaling).to(self.device)
        t_rotation = torch.from_numpy(new_rotation).to(self.device)
        t_opacity = torch.from_numpy(new_opacity).to(self.device)

        if self.num_gaussians == 0:
            self._xyz = nn.Parameter(t_pts_w)
            self._features_dc = nn.Parameter(t_features)
            self._scaling = nn.Parameter(t_scaling)
            self._rotation = nn.Parameter(t_rotation)
            self._opacity = nn.Parameter(t_opacity)
        else:
            self._xyz = nn.Parameter(torch.cat([self._xyz.data, t_pts_w], dim=0))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, t_features], dim=0))
            self._scaling = nn.Parameter(torch.cat([self._scaling.data, t_scaling], dim=0))
            self._rotation = nn.Parameter(torch.cat([self._rotation.data, t_rotation], dim=0))
            self._opacity = nn.Parameter(torch.cat([self._opacity.data, t_opacity], dim=0))

        return n_new

    def export_ply(self, filepath: str):
        """
        Exports 3D Gaussians to official standard 3DGS binary PLY format (Kerbl et al. / SuperSplat / WebGL).
        Properties:
            x, y, z, nx, ny, nz,
            f_dc_0, f_dc_1, f_dc_2 (Spherical Harmonics DC coefficients),
            opacity (logit scale),
            scale_0, scale_1, scale_2 (log scale),
            rot_0, rot_1, rot_2, rot_3 (quaternion w, x, y, z)
        """
        import os
        xyz = self.get_xyz().detach().cpu().numpy().astype(np.float32)
        n_pts = len(xyz)
        if n_pts == 0:
            print("[3DGS] Warning: No Gaussians to export.")
            return

        normals = np.zeros((n_pts, 3), dtype=np.float32)
        
        # In official 3DGS, RGB in [0, 1] is converted to SH DC: (RGB - 0.5) / 0.28209479177387814
        SH_C0 = 0.28209479177387814
        colors_rgb = self.get_features().detach().cpu().numpy().astype(np.float32)
        f_dc = ((colors_rgb - 0.5) / SH_C0).astype(np.float32)
        
        # Raw parameters (log scales and logits)
        opacity_raw = self._opacity.detach().cpu().numpy().astype(np.float32)
        scaling_raw = self._scaling.detach().cpu().numpy().astype(np.float32)
        rot_norm = self.get_rotation().detach().cpu().numpy().astype(np.float32)

        # Standard 3DGS binary PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_pts}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        # Pack into (N, 17) float32 array
        data_block = np.hstack([
            xyz,           # 0..2
            normals,       # 3..5
            f_dc,          # 6..8
            opacity_raw,   # 9
            scaling_raw,   # 10..12
            rot_norm       # 13..16
        ]).astype(np.float32)

        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(data_block.tobytes())

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[3DGS] Exported {n_pts} Gaussians to official 3DGS PLY: {filepath} ({file_size_mb:.2f} MB)")
