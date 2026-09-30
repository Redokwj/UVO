#!/usr/bin/env python3
"""
UVO ROS Wrapper (ROS 1)
Subscribes to:
  - /camera/image_raw (sensor_msgs/Image)
  - /mavros/imu/data (sensor_msgs/Imu)
Synchronizes frames and feeds them to the UVO pipeline.
Uses IMU orientation (fused EKF from Pixhawk) to extract the gravity vector and constrain roll/pitch.
"""

import sys
import os
import time
import numpy as np
from typing import Optional

try:
    import rospy
    import tf.transformations as tf_trans
    from sensor_msgs.msg import Image, Imu
    from cv_bridge import CvBridge
    import message_filters
except ImportError:
    print("ROS 1 (rospy, cv_bridge, sensor_msgs) is required to run this script.")
    sys.exit(1)

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

from uvo.pipeline import UVOPipeline, UVOConfig, FrontendMode, DepthModelType
from uvo.core.camera import CameraIntrinsics


class UVORosNode:
    def __init__(self):
        rospy.init_node("uvo_odometry_node", anonymous=True)
        
        self.bridge = CvBridge()
        
        # Load configuration from ROS parameters
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.imu_topic = rospy.get_param("~imu_topic", "/mavros/imu/data")
        
        fx = rospy.get_param("~fx", 778.58)
        fy = rospy.get_param("~fy", 1127.79)
        cx = rospy.get_param("~cx", 343.86)
        cy = rospy.get_param("~cy", 311.17)
        self.intrinsics = CameraIntrinsics.from_tuple((fx, fy, cx, cy), width=704, height=576)
        
        # Initialize UVO Pipeline
        config = UVOConfig(
            frontend_mode=FrontendMode.HYBRID,
            depth_mode=DepthModelType.UNIDEPTH,
            enable_metric_depth=True,
            enable_loop_closure=True,
            loop_min_gap_dist_m=80.0
        )
        self.pipeline = UVOPipeline(config=config)
        self.frame_count = 0
        
        # Setup Approximate Time Synchronizer for Image and IMU
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.imu_sub = message_filters.Subscriber(self.imu_topic, Imu)
        
        # Buffer up to 10 frames, sync within 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.imu_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        rospy.loginfo(f"[UVO Node] Initialized. Listening to {self.image_topic} and {self.imu_topic}")

    def get_gravity_in_camera_frame(self, imu_msg: Imu) -> np.ndarray:
        """
        Extracts gravity vector in the Camera frame from MAVROS IMU orientation.
        Assuming:
          - MAVROS provides IMU orientation in ENU frame (Z is up)
          - Camera is mounted rigidly to the rover.
        """
        # 1. Read IMU orientation (quaternion: x, y, z, w)
        q = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]
        
        # 2. Get rotation matrix from IMU to World (ENU)
        R_world_imu = tf_trans.quaternion_matrix(q)[:3, :3]
        
        # 3. Gravity in World (ENU) points strictly down: [0, 0, -1]
        g_world = np.array([0.0, 0.0, -1.0])
        
        # 4. Gravity in IMU frame: R_imu_world * g_world = R_world_imu.T * g_world
        g_imu = R_world_imu.T @ g_world
        
        # 5. Transform gravity from IMU frame to Camera frame
        # TODO: Replace R_cam_imu with actual calibration matrix for the specific rover setup!
        # Example standard setup (Camera Z forward, Y down, X right; IMU X forward, Y left, Z up):
        R_cam_imu = np.array([
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
            [ 1.0,  0.0,  0.0]
        ])
        
        g_cam = R_cam_imu @ g_imu
        return g_cam / np.linalg.norm(g_cam) # Normalize just in case

    def sync_callback(self, image_msg: Image, imu_msg: Imu):
        try:
            # Convert ROS Image to OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
            
        timestamp = image_msg.header.stamp.to_sec()
        
        # Get gravity vector constrained by IMU
        gravity_vector = self.get_gravity_in_camera_frame(imu_msg)
        
        t0 = time.perf_counter()
        
        # Run Odometry Pipeline
        pose_wc, is_kf, tracking_res = self.pipeline.process_image(
            image=cv_image,
            timestamp=timestamp,
            intrinsics=self.intrinsics,
            gravity_vector=gravity_vector
        )
        
        dt = time.perf_counter() - t0
        
        self.frame_count += 1
        fps = 1.0 / max(1e-4, dt)
        
        pos = pose_wc.t
        inliers = tracking_res.num_inliers if tracking_res else 0
        tag = "[KF]" if is_kf else "    "
        rospy.loginfo(f"[{self.frame_count:04d}] {tag} Pos: {pos[0]:+5.2f}, {pos[1]:+5.2f}, {pos[2]:+5.2f} | Inliers: {inliers:4d} | {dt*1000:4.1f}ms ({fps:4.1f} FPS)")
        
        # TODO: Publish `pose_wc` to a ROS TF or nav_msgs/Odometry topic for the flight controller!

if __name__ == "__main__":
    node = UVORosNode()
    rospy.spin()
