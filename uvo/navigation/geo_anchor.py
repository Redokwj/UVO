"""
Optical Landmark & Operator Geo-Anchor System (Human-in-the-Loop Georeferencing).

Enables true-world geographic alignment (WGS84 / UTM / ENU) in 100% GNSS-denied / EW-jammed environments.
Allows operators to anchor the local SLAM coordinate frame using visual landmarks:
1. Single-Point Position Anchor (known current location + optional heading).
2. Two-Point Landmark Alignment (Umeyama/Kabsch rigid alignment for automatic True North & metric scale verification).
3. Ray-Casting Landmark Projection (click on a landmark in the video frame, unproject via UniDepth metric depth).
4. Export to GeoJSON and KML (compatible with Google Earth, QGIS, ATAK, Delta, Kropyva).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import json
import numpy as np

from uvo.core.geometry import SE3, unproject_pixels
from uvo.core.sensors import GeoCoordinateTransformer


@dataclass
class GeoPoint:
    lat: float
    lon: float
    alt: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.lat, self.lon, self.alt)


class GeoAnchorManager:
    """
    Manages transformation between UVO's local metric SLAM coordinate system
    and real-world geographic coordinates (WGS84 / local ENU).
    
    Transformation model:
        P_enu = scale * (R_enu_local @ P_local) + t_enu_local
    """
    def __init__(
        self,
        ref_lat: float = 50.4501,
        ref_lon: float = 30.5234,
        ref_alt: float = 150.0
    ):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        
        self.transformer = GeoCoordinateTransformer(ref_lat, ref_lon, ref_alt)
        
        # Transformation parameters from local SLAM to ENU
        self.R_enu_local = np.eye(3, dtype=np.float64)
        self.t_enu_local = np.zeros(3, dtype=np.float64)
        self.scale = 1.0
        
        self.is_anchored = False
        self.anchor_history: List[Dict[str, Any]] = []

    def set_position_anchor(
        self,
        local_pos: Union[np.ndarray, list, tuple],
        geo_point: GeoPoint,
        heading_deg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Mode 1: Single-Point Position Anchor.
        Operator specifies that the robot is currently at geo_point.
        
        Args:
            local_pos: Current (X, Y, Z) in UVO local metric frame.
            geo_point: Real-world (Lat, Lon, Alt) provided by operator.
            heading_deg: Optional heading (azimuth in degrees, 0 = North, 90 = East).
        """
        local_p = np.asarray(local_pos, dtype=np.float64).flatten()
        enu_p = self.transformer.geodetic_to_enu(geo_point.lat, geo_point.lon, geo_point.alt)
        
        if heading_deg is not None:
            # Azimuth angle clockwise from North:
            # ENU frame: East = +X, North = +Y, Up = +Z.
            # Heading 0 deg = North (+Y)
            # Yaw angle in ENU: yaw = 90 - heading_deg
            yaw_enu_rad = math.radians(90.0 - heading_deg)
            cos_y = math.cos(yaw_enu_rad)
            sin_y = math.sin(yaw_enu_rad)
            
            # SLAM local frame: +Z forward (or +X forward depending on convention)
            # Assuming SLAM +X is forward:
            self.R_enu_local = np.array([
                [cos_y, -sin_y, 0.0],
                [sin_y,  cos_y, 0.0],
                [0.0,    0.0,   1.0]
            ], dtype=np.float64)
            
        self.t_enu_local = enu_p - self.scale * (self.R_enu_local @ local_p)
        self.is_anchored = True
        
        info = {
            "mode": "single_point",
            "geo_point": (geo_point.lat, geo_point.lon, geo_point.alt),
            "enu_anchor": enu_p.tolist(),
            "heading_deg": heading_deg
        }
        self.anchor_history.append(info)
        return info

    def align_two_landmarks(
        self,
        local_pos_1: Union[np.ndarray, list, tuple],
        geo_point_1: GeoPoint,
        local_pos_2: Union[np.ndarray, list, tuple],
        geo_point_2: GeoPoint,
        enforce_unit_scale: bool = True
    ) -> Dict[str, Any]:
        """
        Mode 2: Two-Point Landmark Alignment (Umeyama/Kabsch 2D closed-form).
        Automatically aligns UVO's orientation to True North and validates metric scale.
        
        Args:
            local_pos_1, local_pos_2: Positions in UVO local metric frame.
            geo_point_1, geo_point_2: Real-world GPS coordinates of the two landmarks.
            enforce_unit_scale: If True, uses scale=1.0 and verifies metric consistency.
        """
        p1_local = np.asarray(local_pos_1, dtype=np.float64).flatten()[:2]
        p2_local = np.asarray(local_pos_2, dtype=np.float64).flatten()[:2]
        
        enu1 = self.transformer.geodetic_to_enu(geo_point_1.lat, geo_point_1.lon, geo_point_1.alt)[:2]
        enu2 = self.transformer.geodetic_to_enu(geo_point_2.lat, geo_point_2.lon, geo_point_2.alt)[:2]
        
        d_local = float(np.linalg.norm(p2_local - p1_local))
        d_enu = float(np.linalg.norm(enu2 - enu1))
        
        if d_local < 1.0 or d_enu < 1.0:
            raise ValueError("Landmarks are too close (< 1 meter) for stable angular alignment.")
            
        ratio_scale = d_enu / d_local
        scale_drift_pct = abs(1.0 - ratio_scale) * 100.0
        
        # Calculate yaw rotation between local vector and ENU vector
        theta_local = math.atan2(p2_local[1] - p1_local[1], p2_local[0] - p1_local[0])
        theta_enu = math.atan2(enu2[1] - enu1[1], enu2[0] - enu1[0])
        delta_yaw = theta_enu - theta_local
        
        cos_y = math.cos(delta_yaw)
        sin_y = math.sin(delta_yaw)
        
        self.R_enu_local = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)
        
        self.scale = 1.0 if enforce_unit_scale else ratio_scale
        
        # Translation from centroid
        c_local = (p1_local + p2_local) / 2.0
        c_enu = (enu1 + enu2) / 2.0
        
        t_xy = c_enu - self.scale * (self.R_enu_local[:2, :2] @ c_local)
        self.t_enu_local = np.array([t_xy[0], t_xy[1], 0.0], dtype=np.float64)
        self.is_anchored = True
        
        # True North Azimuth of the local +X axis
        # local +X in ENU: (cos_y, sin_y). Azimuth = 90 - atan2(sin_y, cos_y)
        azimuth_deg = (90.0 - math.degrees(delta_yaw)) % 360.0
        
        info = {
            "mode": "two_landmarks",
            "distance_slam_m": d_local,
            "distance_real_m": d_enu,
            "metric_scale_ratio": ratio_scale,
            "scale_consistency_error_pct": scale_drift_pct,
            "true_north_azimuth_deg": azimuth_deg
        }
        self.anchor_history.append(info)
        return info

    def anchor_from_screen_landmark(
        self,
        camera_pose_wc: SE3,
        pixel_u: int,
        pixel_v: int,
        depth_map: np.ndarray,
        intrinsics,
        landmark_geo: GeoPoint,
        heading_deg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Mode 3: Ray-Casting Landmark Projection.
        Operator clicks on a recognized physical landmark (e.g. corner of hangar)
        on the camera feed at pixel (u, v) and enters its real GPS coordinates.
        Uses UniDepth metric depth to backproject the 3D ray and anchors the system.
        """
        h, w = depth_map.shape[:2]
        u_cl = np.clip(pixel_u, 0, w - 1)
        v_cl = np.clip(pixel_v, 0, h - 1)
        
        d = float(depth_map[v_cl, u_cl])
        if d <= 0.1 or d > 100.0 or math.isnan(d):
            # Window search for valid depth near click
            patch = depth_map[max(0, v_cl-2):min(h, v_cl+3), max(0, u_cl-2):min(w, u_cl+3)]
            valid = patch[patch > 0.1]
            if len(valid) == 0:
                raise ValueError(f"No valid metric depth at clicked pixel ({pixel_u}, {pixel_v}).")
            d = float(np.median(valid))
            
        fx = intrinsics.fx if hasattr(intrinsics, 'fx') else intrinsics[0]
        fy = intrinsics.fy if hasattr(intrinsics, 'fy') else intrinsics[1]
        cx = intrinsics.cx if hasattr(intrinsics, 'cx') else intrinsics[2]
        cy = intrinsics.cy if hasattr(intrinsics, 'cy') else intrinsics[3]
        
        # Unproject to camera frame
        x_c = (u_cl - cx) * d / fx
        y_c = (v_cl - cy) * d / fy
        z_c = d
        pt_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        
        # Transform landmark to local SLAM world frame: P_w = R_wc * P_c + t_wc
        pt_landmark_local = camera_pose_wc.R @ pt_cam + camera_pose_wc.t
        
        # Convert landmark real coordinates to ENU
        enu_landmark = self.transformer.geodetic_to_enu(landmark_geo.lat, landmark_geo.lon, landmark_geo.alt)
        
        if heading_deg is not None:
            yaw_enu_rad = math.radians(90.0 - heading_deg)
            cos_y = math.cos(yaw_enu_rad)
            sin_y = math.sin(yaw_enu_rad)
            self.R_enu_local = np.array([
                [cos_y, -sin_y, 0.0],
                [sin_y,  cos_y, 0.0],
                [0.0,    0.0,   1.0]
            ], dtype=np.float64)
            
        # The translation aligns the landmark in SLAM with the landmark in ENU
        self.t_enu_local = enu_landmark - self.scale * (self.R_enu_local @ pt_landmark_local)
        self.is_anchored = True
        
        # Calculate current rover GPS position
        rover_enu = self.scale * (self.R_enu_local @ camera_pose_wc.t) + self.t_enu_local
        rover_lat, rover_lon, rover_alt = self.transformer.enu_to_geodetic(rover_enu[0], rover_enu[1], rover_enu[2])
        
        info = {
            "mode": "raycast_screen_click",
            "pixel": (pixel_u, pixel_v),
            "landmark_distance_m": d,
            "landmark_geo": (landmark_geo.lat, landmark_geo.lon, landmark_geo.alt),
            "estimated_rover_lat": rover_lat,
            "estimated_rover_lon": rover_lon,
            "estimated_rover_alt": rover_alt
        }
        self.anchor_history.append(info)
        return info

    def local_to_wgs84(self, local_xyz: Union[np.ndarray, list, tuple]) -> Tuple[float, float, float]:
        """
        Converts a local SLAM metric coordinate (X, Y, Z) to WGS-84 (Lat, Lon, Alt).
        """
        p = np.asarray(local_xyz, dtype=np.float64).flatten()
        enu = self.scale * (self.R_enu_local @ p) + self.t_enu_local
        return self.transformer.enu_to_geodetic(enu[0], enu[1], enu[2])

    def wgs84_to_local(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        """
        Converts real-world WGS-84 (Lat, Lon, Alt) to UVO local SLAM metric coordinates (X, Y, Z).
        Enables Waypoint navigation in GNSS-denied zones!
        """
        enu = self.transformer.geodetic_to_enu(lat, lon, alt)
        p_local = (self.R_enu_local.T @ (enu - self.t_enu_local)) / self.scale
        return p_local

    def export_geojson(
        self,
        trajectory: List[Tuple[float, SE3]],
        filepath: str,
        name: str = "UVO Georeferenced Trajectory"
    ):
        """
        Exports the SLAM trajectory to standard GeoJSON format for viewing in
        QGIS, ATAK, Delta, Kropyva, or geojson.io.
        """
        coordinates = []
        for t, pose in trajectory:
            lat, lon, alt = self.local_to_wgs84(pose.t)
            coordinates.append([lon, lat, alt]) # GeoJSON order: [Lon, Lat, Alt]
            
        feature = {
            "type": "Feature",
            "properties": {
                "name": name,
                "anchor_ref_lat": self.ref_lat,
                "anchor_ref_lon": self.ref_lon,
                "scale": self.scale,
                "num_points": len(coordinates)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }
        geojson = {
            "type": "FeatureCollection",
            "features": [feature]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
            
        print(f"[GeoAnchor] Exported {len(coordinates)} points to GeoJSON: {filepath}")

    def export_kml(
        self,
        trajectory: List[Tuple[float, SE3]],
        filepath: str,
        name: str = "UVO Rover Path"
    ):
        """
        Exports trajectory to standard KML for direct 3D viewing in Google Earth.
        """
        coord_strings = []
        for t, pose in trajectory:
            lat, lon, alt = self.local_to_wgs84(pose.t)
            coord_strings.append(f"{lon:.8f},{lat:.8f},{alt:.2f}")
            
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="roverTrack">
      <LineStyle>
        <color>ff00ff00</color>
        <width>4</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>{name}</name>
      <styleUrl>#roverTrack</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>relativeToGround</altitudeMode>
        <coordinates>
          {" ".join(coord_strings)}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(kml_content)
            
        print(f"[GeoAnchor] Exported KML to: {filepath}")
