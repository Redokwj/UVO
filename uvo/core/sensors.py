import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

from .geometry import SE3


class NavSatStatus(IntEnum):
    """
    Standard GNSS Fix status compatible with sensor_msgs/NavSatStatus
    and MAVLink GPS status.
    """
    NO_FIX = -1
    FIX_2D = 0
    FIX_3D = 1
    RTK_FLOAT = 2
    RTK_FIXED = 3


@dataclass
class CameraExtrinsics:
    """
    Rigid calibration extrinsics between Camera and Vehicle Body / IMU frame.
    Rigid mount on rover chassis ensures static transform without gimbal lag.
    
    transform_body_cam: SE3 transformation from Camera optical frame to Body/IMU frame.
        P_body = transform_body_cam * P_cam
    """
    transform_body_cam: SE3 = field(default_factory=SE3.identity)
    mount_type: str = "rigid"
    description: str = "Rigid chassis mount"

    @property
    def transform_cam_body(self) -> SE3:
        """From Body frame to Camera optical frame."""
        return self.transform_body_cam.inv()

    @classmethod
    def from_rpy_and_xyz(
        cls,
        xyz: Union[list, tuple, np.ndarray],
        rpy_degrees: Union[list, tuple, np.ndarray],
        mount_type: str = "rigid",
        description: str = "Rigid chassis mount"
    ) -> "CameraExtrinsics":
        """
        Constructs extrinsics from translation [x, y, z] (m) and Euler angles [roll, pitch, yaw] (degrees).
        Rotation follows intrinsic Z-Y-X convention.
        """
        x, y, z = xyz
        roll_d, pitch_d, yaw_d = rpy_degrees
        r = math.radians(roll_d)
        p = math.radians(pitch_d)
        y_rad = math.radians(yaw_d)

        # Rotation matrices
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(r), -math.sin(r)],
            [0.0, math.sin(r), math.cos(r)]
        ], dtype=np.float64)

        Ry = np.array([
            [math.cos(p), 0.0, math.sin(p)],
            [0.0, 1.0, 0.0],
            [-math.sin(p), 0.0, math.cos(p)]
        ], dtype=np.float64)

        Rz = np.array([
            [math.cos(y_rad), -math.sin(y_rad), 0.0],
            [math.sin(y_rad), math.cos(y_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R = Rz @ Ry @ Rx
        t = np.array([x, y, z], dtype=np.float64)
        return cls(transform_body_cam=SE3(R, t), mount_type=mount_type, description=description)


@dataclass
class ImuMeasurement:
    """
    IMU data packet containing linear acceleration and angular velocity.
    Compatible with sensor_msgs/Imu (from MAVROS or standalone ROS topic/driver).
    """
    timestamp: float                       # Seconds (POSIX or monotonic ROS timestamp)
    angular_velocity: np.ndarray          # [gx, gy, gz] in rad/s
    linear_acceleration: np.ndarray       # [ax, ay, az] in m/s^2
    orientation: Optional[np.ndarray] = None  # [qx, qy, qz, qw] (optional onboard EKF filter quaternion)
    
    # 3x3 covariance matrices (or None if unstated)
    angular_velocity_cov: Optional[np.ndarray] = None
    linear_acceleration_cov: Optional[np.ndarray] = None
    orientation_cov: Optional[np.ndarray] = None

    def __post_init__(self):
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64).flatten()
        self.linear_acceleration = np.asarray(self.linear_acceleration, dtype=np.float64).flatten()
        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=np.float64).flatten()

    @classmethod
    def from_ros_msg(cls, msg: Any) -> "ImuMeasurement":
        """
        Parses ROS sensor_msgs/Imu message object or dict equivalent.
        Supports both ROS 1 / ROS 2 message objects and JSON/dict logs.
        """
        if isinstance(msg, dict):
            # Parse dict format
            stamp = msg.get("header", {}).get("stamp", 0.0)
            if isinstance(stamp, dict):
                timestamp = float(stamp.get("secs", 0)) + float(stamp.get("nsecs", 0)) * 1e-9
            else:
                timestamp = float(stamp)
                
            ang = msg.get("angular_velocity", {})
            acc = msg.get("linear_acceleration", {})
            ori = msg.get("orientation", {})
            
            gx = ang.get("x", 0.0) if isinstance(ang, dict) else ang[0]
            gy = ang.get("y", 0.0) if isinstance(ang, dict) else ang[1]
            gz = ang.get("z", 0.0) if isinstance(ang, dict) else ang[2]
            
            ax = acc.get("x", 0.0) if isinstance(acc, dict) else acc[0]
            ay = acc.get("y", 0.0) if isinstance(acc, dict) else acc[1]
            az = acc.get("z", 0.0) if isinstance(acc, dict) else acc[2]
            
            qx = ori.get("x", 0.0) if isinstance(ori, dict) else (ori[0] if ori else None)
            qy = ori.get("y", 0.0) if isinstance(ori, dict) else (ori[1] if ori else None)
            qz = ori.get("z", 0.0) if isinstance(ori, dict) else (ori[2] if ori else None)
            qw = ori.get("w", 1.0) if isinstance(ori, dict) else (ori[3] if ori else None)
            
            orientation = np.array([qx, qy, qz, qw], dtype=np.float64) if qx is not None else None
            return cls(
                timestamp=timestamp,
                angular_velocity=np.array([gx, gy, gz], dtype=np.float64),
                linear_acceleration=np.array([ax, ay, az], dtype=np.float64),
                orientation=orientation
            )
        else:
            # Native ROS message object (sensor_msgs/Imu)
            stamp = msg.header.stamp
            timestamp = stamp.to_sec() if hasattr(stamp, "to_sec") else (stamp.sec + stamp.nanosec * 1e-9)
            
            ang = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64)
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float64)
            
            # Check if orientation is valid (qw != 0)
            if hasattr(msg, "orientation") and (abs(msg.orientation.w) > 1e-6 or abs(msg.orientation.x) > 1e-6):
                ori = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float64)
            else:
                ori = None
                
            return cls(
                timestamp=timestamp,
                angular_velocity=ang,
                linear_acceleration=acc,
                orientation=ori
            )


class ImuBuffer:
    """
    Time-synchronized circular buffer for high-rate IMU measurements (e.g. 100-400 Hz).
    Allows retrieving time slices between consecutive visual camera frames.
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[ImuMeasurement] = []

    def add(self, meas: ImuMeasurement):
        self.buffer.append(meas)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_between(self, t0: float, t1: float) -> List[ImuMeasurement]:
        """Returns all IMU measurements between timestamp t0 and t1 inclusive."""
        return [m for m in self.buffer if t0 <= m.timestamp <= t1]

    def estimate_gravity_vector(self, window_sec: float = 1.0) -> np.ndarray:
        """
        Estimates the downward gravity vector in the IMU frame while the vehicle is stationary.
        Returns a unit 3D vector.
        """
        if not self.buffer:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
            
        t_end = self.buffer[-1].timestamp
        t_start = t_end - window_sec
        recent = [m.linear_acceleration for m in self.buffer if m.timestamp >= t_start]
        
        if not recent:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
            
        mean_acc = np.mean(recent, axis=0)
        norm = np.linalg.norm(mean_acc)
        if norm > 1e-4:
            return mean_acc / norm
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class NavSatMeasurement:
    """
    GNSS position measurement compatible with sensor_msgs/NavSatFix and MAVROS /mavros/global_position/raw/fix.
    """
    timestamp: float                       # Seconds
    latitude: float                        # Degrees (-90.0 to +90.0)
    longitude: float                       # Degrees (-180.0 to +180.0)
    altitude: float                        # Meters (Ellipsoidal or AMSL)
    status: NavSatStatus = NavSatStatus.FIX_3D
    position_covariance: Optional[np.ndarray] = None  # (3, 3) meters^2 [East, North, Up]
    satellites_visible: Optional[int] = None
    hdop: Optional[float] = None

    @classmethod
    def from_ros_msg(cls, msg: Any) -> "NavSatMeasurement":
        """
        Parses ROS sensor_msgs/NavSatFix message object or dict.
        """
        if isinstance(msg, dict):
            stamp = msg.get("header", {}).get("stamp", 0.0)
            if isinstance(stamp, dict):
                timestamp = float(stamp.get("secs", 0)) + float(stamp.get("nsecs", 0)) * 1e-9
            else:
                timestamp = float(stamp)
                
            lat = float(msg.get("latitude", 0.0))
            lon = float(msg.get("longitude", 0.0))
            alt = float(msg.get("altitude", 0.0))
            status_val = msg.get("status", {}).get("status", 0) if isinstance(msg.get("status"), dict) else 0
            
            cov_raw = msg.get("position_covariance", None)
            cov = np.array(cov_raw, dtype=np.float64).reshape((3, 3)) if cov_raw is not None else None
            
            return cls(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                status=NavSatStatus(status_val) if status_val in [-1, 0, 1, 2, 3] else NavSatStatus.FIX_3D,
                position_covariance=cov
            )
        else:
            stamp = msg.header.stamp
            timestamp = stamp.to_sec() if hasattr(stamp, "to_sec") else (stamp.sec + stamp.nanosec * 1e-9)
            
            cov = None
            if hasattr(msg, "position_covariance") and len(msg.position_covariance) == 9:
                cov = np.array(msg.position_covariance, dtype=np.float64).reshape((3, 3))
                
            status_val = msg.status.status if hasattr(msg, "status") else 1
            return cls(
                timestamp=timestamp,
                latitude=float(msg.latitude),
                longitude=float(msg.longitude),
                altitude=float(msg.altitude),
                status=NavSatStatus(status_val) if status_val in [-1, 0, 1, 2, 3] else NavSatStatus.FIX_3D,
                position_covariance=cov
            )


class GeoCoordinateTransformer:
    """
    High-precision WGS84 Geodetic to local Cartesian ENU (East-North-Up) metric coordinate transformer.
    Computes local tangent plane metric displacements without external GIS library dependencies.
    """
    # WGS84 Ellipsoid constants
    WGS84_A = 6378137.0          # Semi-major axis in meters
    WGS84_E2 = 0.00669437999014  # First eccentricity squared

    def __init__(self, ref_lat: float, ref_lon: float, ref_alt: float):
        """
        Anchor local metric origin (0, 0, 0) at the reference geodetic coordinates.
        """
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        
        self.lat_rad = math.radians(ref_lat)
        self.lon_rad = math.radians(ref_lon)
        
        # Prime vertical radius of curvature
        sin_lat = math.sin(self.lat_rad)
        cos_lat = math.cos(self.lat_rad)
        sin_lon = math.sin(self.lon_rad)
        cos_lon = math.cos(self.lon_rad)
        
        N = self.WGS84_A / math.sqrt(1.0 - self.WGS84_E2 * sin_lat * sin_lat)
        
        # Reference ECEF coordinates
        self.ref_x = (N + ref_alt) * cos_lat * cos_lon
        self.ref_y = (N + ref_alt) * cos_lat * sin_lon
        self.ref_z = (N * (1.0 - self.WGS84_E2) + ref_alt) * sin_lat
        
        # Rotation matrix from ECEF to local ENU
        self.R_ecef_enu = np.array([
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ], dtype=np.float64)

    def wgs84_to_ecef(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        sin_lat = math.sin(lat_r)
        cos_lat = math.cos(lat_r)
        sin_lon = math.sin(lon_r)
        cos_lon = math.cos(lon_r)
        
        N = self.WGS84_A / math.sqrt(1.0 - self.WGS84_E2 * sin_lat * sin_lat)
        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1.0 - self.WGS84_E2) + alt) * sin_lat
        return x, y, z

    def geodetic_to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Converts WGS84 (latitude, longitude, altitude) to local metric [East, North, Up] in meters.
        """
        x, y, z = self.wgs84_to_ecef(lat, lon, alt)
        dx = x - self.ref_x
        dy = y - self.ref_y
        dz = z - self.ref_z
        
        enu = self.R_ecef_enu @ np.array([dx, dy, dz], dtype=np.float64)
        return enu

    def enu_to_geodetic(self, east: float, north: float, up: float) -> Tuple[float, float, float]:
        """
        Converts local metric [East, North, Up] in meters back to WGS84 (latitude, longitude, altitude).
        Iterative Bowring algorithm.
        """
        enu = np.array([east, north, up], dtype=np.float64)
        d_ecef = self.R_ecef_enu.T @ enu
        x = self.ref_x + d_ecef[0]
        y = self.ref_y + d_ecef[1]
        z = self.ref_z + d_ecef[2]
        
        # ECEF to Geodetic
        p = math.sqrt(x * x + y * y)
        b = self.WGS84_A * math.sqrt(1.0 - self.WGS84_E2)
        e2_prime = (self.WGS84_A * self.WGS84_A - b * b) / (b * b)
        
        theta = math.atan2(z * self.WGS84_A, p * b)
        lat_r = math.atan2(
            z + e2_prime * b * (math.sin(theta) ** 3),
            p - self.WGS84_E2 * self.WGS84_A * (math.cos(theta) ** 3)
        )
        lon_r = math.atan2(y, x)
        
        sin_lat = math.sin(lat_r)
        N = self.WGS84_A / math.sqrt(1.0 - self.WGS84_E2 * sin_lat * sin_lat)
        alt = (p / math.cos(lat_r)) - N
        
        return math.degrees(lat_r), math.degrees(lon_r), alt
