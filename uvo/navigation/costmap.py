"""
2.5D Traversability & Elevation Costmap from Foundation Metric Depth (UniDepth V2).

Transforms per-pixel metric 3D points (X, Y, Z) in camera frame into a 
bird's-eye-view (BEV) 2.5D elevation grid and occupancy/costmap (ROS OccupancyGrid compliant).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


@dataclass
class CostmapConfig:
    # Grid dimensions in robot frame (X: forward, Y: lateral/left)
    min_x: float = 0.5        # Min distance in front of rover bumper (m)
    max_x: float = 8.5        # Max forward detection distance (m)
    min_y: float = -4.0       # Left border (m)
    max_y: float = 4.0        # Right border (m)
    resolution: float = 0.08  # Cell size (8 cm per cell)
    
    # Camera mount geometry relative to rover base_link
    camera_height: float = 0.45      # Camera height above ground level (m)
    camera_pitch_deg: float = 12.0   # Downward tilt angle (degrees)
    
    # Traversability thresholds
    safe_step_threshold: float = 0.06       # Height difference < 6 cm -> Safe ground
    obstacle_step_threshold: float = 0.16   # Height difference > 16 cm -> Positive obstacle (lethal)
    ditch_depth_threshold: float = 0.15     # Drop below ground > 15 cm -> Negative obstacle (hole/trench)
    max_traversable_slope_deg: float = 22.0 # Slope > 22 deg -> Rollover hazard (critical slope)
    
    # Inflation layer parameters
    robot_radius: float = 0.40        # Robot footprint half-width/radius (m)
    inflation_radius: float = 0.75    # Buffer safety margin (m)
    
    # Subsampling step for depth map processing speed (3 = 9x faster, high terrain fidelity)
    subsample_step: int = 3


@dataclass
class CostmapResult:
    costmap: np.ndarray          # 2D uint8 grid: 0 (free) .. 100 (lethal obstacle), 255 (unknown)
    elevation_mean: np.ndarray   # 2D float32: mean elevation in meters (Z relative to ground)
    elevation_diff: np.ndarray   # 2D float32: max - min height within cell (m)
    slope_deg: np.ndarray        # 2D float32: surface slope in degrees
    hazard_labels: np.ndarray    # 2D uint8: 0=Safe, 1=Rough, 2=Obstacle(+), 3=Hole(-), 4=Slope, 255=Unknown
    
    # Grid spatial metadata
    resolution: float
    origin_x: float
    origin_y: float
    width_cells: int
    height_cells: int
    elapsed_time_ms: float

    def to_ros_occupancy_grid(self, frame_id: str = "base_link", timestamp=None) -> Any:
        """
        Converts the costmap to a ROS 2 nav_msgs/OccupancyGrid message if rclpy is installed.
        Returns None or dictionary if rclpy is not present.
        """
        try:
            from nav_msgs.msg import OccupancyGrid
            from std_msgs.msg import Header
            
            grid_msg = OccupancyGrid()
            grid_msg.header = Header()
            grid_msg.header.frame_id = frame_id
            if timestamp is not None:
                grid_msg.header.stamp = timestamp
                
            grid_msg.info.resolution = float(self.resolution)
            grid_msg.info.width = int(self.width_cells)
            grid_msg.info.height = int(self.height_cells)
            grid_msg.info.origin.position.x = float(self.origin_x)
            grid_msg.info.origin.position.y = float(self.origin_y)
            grid_msg.info.origin.position.z = 0.0
            
            # Convert 255 to -1 for standard ROS OccupancyGrid unknown
            ros_data = self.costmap.astype(np.int8)
            ros_data[self.costmap == 255] = -1
            grid_msg.data = ros_data.flatten().tolist()
            return grid_msg
        except ImportError:
            return {
                "frame_id": frame_id,
                "resolution": self.resolution,
                "width": self.width_cells,
                "height": self.height_cells,
                "origin_x": self.origin_x,
                "origin_y": self.origin_y,
                "costmap": self.costmap
            }

    def render_bev_image(self, upscale_factor: int = 5) -> np.ndarray:
        """
        Renders a colorized Bird's-Eye-View (BEV) visualization image of the costmap:
        - Gray (40,40,40): Unknown / Unobserved
        - Dark Green (40,160,40): Free / Smooth road
        - Yellow/Amber (0,200,255): Rough ground / Minor bumps
        - Bright Red (0,0,240): Positive Obstacle (tree, stone, wall)
        - Cyan/Blue (255,180,0): Negative Obstacle (hole, ditch, trench)
        - Magenta (200,0,200): Rollover slope hazard
        - Orange (0,120,230): Inflated safety zone around obstacles
        """
        h, w = self.costmap.shape
        bev_bgr = np.full((h, w, 3), 40, dtype=np.uint8) # Dark background
        
        # Free observed ground (cost 0..10)
        free_mask = (self.costmap <= 10) & (self.costmap != 255)
        bev_bgr[free_mask] = (40, 160, 40) # Green
        
        # Rough terrain (cost 11..70)
        rough_mask = (self.hazard_labels == 1) & (self.costmap != 255)
        bev_bgr[rough_mask] = (0, 200, 255) # Yellow/Amber
        
        # Inflated buffer zone
        inflated_mask = (self.costmap > 10) & (self.costmap < 100) & (~rough_mask) & (self.costmap != 255)
        bev_bgr[inflated_mask] = (0, 120, 230) # Orange
        
        # Hazards
        pos_obs = (self.hazard_labels == 2)
        bev_bgr[pos_obs] = (0, 0, 240) # Bright Red
        
        neg_obs = (self.hazard_labels == 3)
        bev_bgr[neg_obs] = (255, 180, 0) # Cyan/Blue
        
        slope_risk = (self.hazard_labels == 4)
        bev_bgr[slope_risk] = (200, 0, 200) # Magenta
        
        # Orient image: Forward (X) points UP, Lateral (Y: left->right) points RIGHT
        bev_oriented = cv2.flip(cv2.transpose(bev_bgr), 0)
        
        if upscale_factor > 1:
            out_h, out_w = bev_oriented.shape[0] * upscale_factor, bev_oriented.shape[1] * upscale_factor
            bev_canvas = cv2.resize(bev_oriented, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            
            # Draw distance range grid lines (every 2 meters forward)
            # Robot bumper is at origin_x (e.g. 0.5m)
            total_range_x = self.resolution * self.width_cells
            for dist_m in [2.0, 4.0, 6.0, 8.0]:
                if dist_m >= self.origin_x and dist_m < self.origin_x + total_range_x:
                    frac = (dist_m - self.origin_x) / total_range_x
                    y_px = int(out_h * (1.0 - frac))
                    cv2.line(bev_canvas, (0, y_px), (out_w, y_px), (70, 70, 70), 1, cv2.LINE_AA)
                    cv2.putText(bev_canvas, f"{dist_m:.0f}m", (10, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
                    
            # Centerline (Y = 0)
            total_range_y = self.resolution * self.height_cells
            frac_y = (0.0 - self.origin_y) / total_range_y
            x_center = int(out_w * frac_y)
            cv2.line(bev_canvas, (x_center, 0), (x_center, out_h), (80, 80, 80), 1, cv2.LINE_AA)
            
            # Draw Rover footprint at the bottom center
            rw_px = int(0.70 / self.resolution * upscale_factor) # Rover width ~ 0.70m
            rl_px = int(0.85 / self.resolution * upscale_factor) # Rover length ~ 0.85m
            pt1 = (x_center - rw_px // 2, out_h - 4)
            pt2 = (x_center + rw_px // 2, out_h - rl_px)
            cv2.rectangle(bev_canvas, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(bev_canvas, "ROVER", (x_center - 22, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            
            # HUD overlay
            hud_text = f"2.5D COSTMAP | {self.elapsed_time_ms:.1f}ms | RES {self.resolution*100:.0f}cm"
            cv2.putText(bev_canvas, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return bev_canvas
            
        return bev_oriented


class TraversabilityCostmap:
    """
    High-performance 2.5D Traversability Costmap Engine.
    Processes metric depth maps from UniDepth V2 and builds real-time obstacle & slope grids.
    """
    def __init__(self, config: Optional[CostmapConfig] = None):
        self.config = config or CostmapConfig()
        
        # Grid parameters
        self.nx = int(np.ceil((self.config.max_x - self.config.min_x) / self.config.resolution))
        self.ny = int(np.ceil((self.config.max_y - self.config.min_y) / self.config.resolution))
        self.total_cells = self.nx * self.ny
        
        # Precompute camera-to-robot coordinate rotation matrix
        # Camera frame: X right, Y down, Z forward
        # Robot frame:  X forward, Y left, Z up
        pitch_rad = np.deg2rad(self.config.camera_pitch_deg)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)
        
        # R_robot_cam:
        # [X_r]   [ 0,  -sin_p,   cos_p ] [X_c]
        # [Y_r] = [-1,       0,       0 ] [Y_c]
        # [Z_r]   [ 0,  -cos_p,  -sin_p ] [Z_c]
        self.R_robot_cam = np.array([
            [ 0.0, -sin_p,  cos_p],
            [-1.0,    0.0,    0.0],
            [ 0.0, -cos_p, -sin_p]
        ], dtype=np.float32)

    def process_depth_map(
        self,
        depth_map: np.ndarray,
        intrinsics,
        camera_height: Optional[float] = None,
        camera_pitch_deg: Optional[float] = None
    ) -> CostmapResult:
        """
        Converts 2D metric depth map to 2.5D Traversability Costmap.
        
        Args:
            depth_map: (H, W) float32 numpy array with metric depth in meters.
            intrinsics: CameraIntrinsics object or (fx, fy, cx, cy).
            camera_height: Optional override for camera height (m).
            camera_pitch_deg: Optional override for camera tilt (deg).
        """
        t0 = cv2.getTickCount()
        
        cam_h = camera_height if camera_height is not None else self.config.camera_height
        
        if camera_pitch_deg is not None and abs(camera_pitch_deg - self.config.camera_pitch_deg) > 1e-3:
            pitch_rad = np.deg2rad(camera_pitch_deg)
            cos_p = np.cos(pitch_rad)
            sin_p = np.sin(pitch_rad)
            R_rc = np.array([
                [ 0.0, -sin_p,  cos_p],
                [-1.0,    0.0,    0.0],
                [ 0.0, -cos_p, -sin_p]
            ], dtype=np.float32)
        else:
            R_rc = self.R_robot_cam

        step = self.config.subsample_step
        sub_depth = depth_map[::step, ::step]
        h_sub, w_sub = sub_depth.shape
        
        if hasattr(intrinsics, 'fx'):
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.cx, intrinsics.cy
        else:
            fx, fy, cx, cy = intrinsics[:4]
            
        # Adjust K for subsampling
        fx_s = fx / step
        fy_s = fy / step
        cx_s = cx / step
        cy_s = cy / step
        
        # Valid depth mask (e.g. 0.4m to 12.0m)
        valid_depth = (sub_depth > 0.3) & (sub_depth < 12.0)
        v_idx, u_idx = np.where(valid_depth)
        z_c = sub_depth[valid_depth]
        
        # 1. Unproject 2D pixels to 3D Camera Frame
        x_c = (u_idx - cx_s) * z_c / fx_s
        y_c = (v_idx - cy_s) * z_c / fy_s
        pts_c = np.column_stack([x_c, y_c, z_c]).astype(np.float32) # (N, 3)
        
        # 2. Transform into Robot base_link frame
        pts_r = pts_c @ R_rc.T
        pts_r[:, 2] += cam_h # Add camera height so ground is approximately Z=0
        
        # 3. Filter points within Costmap ROI
        roi_mask = (
            (pts_r[:, 0] >= self.config.min_x) & (pts_r[:, 0] < self.config.max_x) &
            (pts_r[:, 1] >= self.config.min_y) & (pts_r[:, 1] < self.config.max_y)
        )
        pts_roi = pts_r[roi_mask]
        
        # 4. Binning into 2D Grid Cells
        ix = ((pts_roi[:, 0] - self.config.min_x) / self.config.resolution).astype(np.int32)
        iy = ((pts_roi[:, 1] - self.config.min_y) / self.config.resolution).astype(np.int32)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        flat_idx = iy * self.nx + ix
        
        z_vals = pts_roi[:, 2]
        
        z_min_arr = np.full(self.total_cells, np.inf, dtype=np.float32)
        z_max_arr = np.full(self.total_cells, -np.inf, dtype=np.float32)
        z_sum_arr = np.zeros(self.total_cells, dtype=np.float32)
        counts_arr = np.bincount(flat_idx, minlength=self.total_cells)
        
        np.minimum.at(z_min_arr, flat_idx, z_vals)
        np.maximum.at(z_max_arr, flat_idx, z_vals)
        np.add.at(z_sum_arr, flat_idx, z_vals)
        
        counts_2d = counts_arr.reshape((self.ny, self.nx))
        observed_mask = counts_2d >= 2 # At least 2 points per cell for reliability
        
        z_min_2d = z_min_arr.reshape((self.ny, self.nx))
        z_max_2d = z_max_arr.reshape((self.ny, self.nx))
        z_mean_2d = np.zeros((self.ny, self.nx), dtype=np.float32)
        z_mean_2d[observed_mask] = z_sum_arr.reshape((self.ny, self.nx))[observed_mask] / counts_2d[observed_mask]
        
        delta_z = np.zeros((self.ny, self.nx), dtype=np.float32)
        delta_z[observed_mask] = z_max_2d[observed_mask] - z_min_2d[observed_mask]
        
        # 5. Slope calculation on mean elevation
        # We fill unobserved cells with near values for smooth gradient
        z_smooth = z_mean_2d.copy()
        if not np.all(observed_mask):
            z_smooth[~observed_mask] = 0.0
            
        res = self.config.resolution
        grad_x = cv2.Sobel(z_smooth, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * res)
        grad_y = cv2.Sobel(z_smooth, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * res)
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.rad2deg(slope_rad)
        
        # 6. Multi-Hazard Classification
        costmap = np.full((self.ny, self.nx), 255, dtype=np.uint8) # Default: 255 (unknown)
        hazard_labels = np.full((self.ny, self.nx), 255, dtype=np.uint8)
        
        # A. Safe ground: observed, low roughness, gentle slope
        safe_mask = observed_mask & (delta_z < self.config.safe_step_threshold) & (slope_deg < 15.0)
        costmap[safe_mask] = 0
        hazard_labels[safe_mask] = 0
        
        # B. Rough ground: 6cm to 16cm height delta
        rough_mask = observed_mask & (delta_z >= self.config.safe_step_threshold) & (delta_z < self.config.obstacle_step_threshold)
        costmap[rough_mask] = np.clip(15 + (delta_z[rough_mask] - self.config.safe_step_threshold) * 450, 15, 65).astype(np.uint8)
        hazard_labels[rough_mask] = 1
        
        # C. Positive Obstacles (rocks, trees, stumps, steps > 16cm)
        pos_obs_mask = observed_mask & (delta_z >= self.config.obstacle_step_threshold)
        costmap[pos_obs_mask] = 100
        hazard_labels[pos_obs_mask] = 2
        
        # D. Negative Obstacles (holes, ditches, trenches > 15cm drop below ground level)
        neg_obs_mask = observed_mask & (z_min_2d < -self.config.ditch_depth_threshold)
        costmap[neg_obs_mask] = 100
        hazard_labels[neg_obs_mask] = 3
        
        # E. Rollover Risk (slope > 22 deg)
        slope_mask = observed_mask & (slope_deg >= self.config.max_traversable_slope_deg)
        costmap[slope_mask] = 100
        hazard_labels[slope_mask] = 4
        
        # 7. Obstacle Inflation Layer (Distance Transform)
        lethal_mask = (costmap == 100).astype(np.uint8)
        if np.any(lethal_mask):
            # Compute Euclidean distance to nearest lethal obstacle in meters
            dist_map = cv2.distanceTransform(1 - lethal_mask, cv2.DIST_L2, 5) * res
            
            # Robot footprint inflation -> lethal
            r_robot = self.config.robot_radius
            r_inf = self.config.inflation_radius
            
            footprint_mask = (dist_map <= r_robot) & (costmap != 255)
            costmap[footprint_mask] = 100
            
            # Buffer decay zone
            decay_mask = (dist_map > r_robot) & (dist_map <= r_inf) & (costmap < 100) & (costmap != 255)
            if np.any(decay_mask):
                decay_costs = (1.0 - (dist_map[decay_mask] - r_robot) / (r_inf - r_robot)) * 95.0
                costmap[decay_mask] = np.maximum(costmap[decay_mask], decay_costs.astype(np.uint8))

        t1 = cv2.getTickCount()
        elapsed_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()
        
        return CostmapResult(
            costmap=costmap,
            elevation_mean=z_mean_2d,
            elevation_diff=delta_z,
            slope_deg=slope_deg,
            hazard_labels=hazard_labels,
            resolution=self.config.resolution,
            origin_x=self.config.min_x,
            origin_y=self.config.min_y,
            width_cells=self.nx,
            height_cells=self.ny,
            elapsed_time_ms=elapsed_ms
        )
