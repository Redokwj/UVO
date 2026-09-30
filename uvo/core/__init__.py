"""
UVO Core module containing basic data structures, SE(3) Lie geometry,
camera models, frames, sensor definitions, and trajectory managers.
"""

from .geometry import (
    SE3,
    project_points,
    unproject_pixels,
    skew_symmetric,
    so3_exp,
    so3_log
)

from .frame import (
    CameraIntrinsics,
    Frame,
    Keyframe
)

from .sensors import (
    NavSatStatus,
    CameraExtrinsics,
    ImuMeasurement,
    ImuBuffer,
    NavSatMeasurement,
    GeoCoordinateTransformer
)

from .trajectory import (
    TrajectoryPoint,
    Trajectory
)

__all__ = [
    "SE3",
    "project_points",
    "unproject_pixels",
    "skew_symmetric",
    "so3_exp",
    "so3_log",
    "CameraIntrinsics",
    "Frame",
    "Keyframe",
    "NavSatStatus",
    "CameraExtrinsics",
    "ImuMeasurement",
    "ImuBuffer",
    "NavSatMeasurement",
    "GeoCoordinateTransformer",
    "TrajectoryPoint",
    "Trajectory"
]
