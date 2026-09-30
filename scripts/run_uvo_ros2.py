#!/usr/bin/env python3
"""
UVO ROS 2 Wrapper
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

from geometry_msgs.msg import PoseStamped, TwistStamped
import math

def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return [x, y, z, w]

class UVORos2Node(Node):
    def __init__(self):
        super().__init__('uvo_odometry_node')
        
        self.bridge = CvBridge()
        
        # 1. Declare and load parameters
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("fx", 778.58)
        self.declare_parameter("fy", 1127.79)
        self.declare_parameter("cx", 343.86)
        self.declare_parameter("cy", 311.17)
        
        image_topic = self.get_parameter("image_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        fx = self.get_parameter("fx").value
        fy = self.get_parameter("fy").value
        cx = self.get_parameter("cx").value
        cy = self.get_parameter("cy").value
        
        self.intrinsics = CameraIntrinsics.from_tuple((fx, fy, cx, cy), width=704, height=576)
        
        # 2. Initialize UVO Pipeline
        config = UVOConfig(
            frontend_mode=FrontendMode.HYBRID,
            depth_mode=DepthModelType.UNIDEPTH,
            enable_metric_depth=True,
            enable_loop_closure=True,
            loop_min_gap_dist_m=80.0
        )
        self.pipeline = UVOPipeline(config=config)
        self.frame_count = 0
        self.prev_pose = None
        self.prev_time = None
        
        # 3. Setup Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/uno/vision_pose/pose", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "/uno/vision_pose/velocity", 10)
        
        # 4. Setup Approximate Time Synchronizer for Image and IMU
        self.image_sub = message_filters.Subscriber(self, Image, image_topic)
        self.imu_sub = message_filters.Subscriber(self, Imu, imu_topic)
        
        # Buffer up to 10 frames, sync within 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info(f"[UVO Node] Initialized! Listening to {image_topic} and {imu_topic}")

    def get_gravity_in_camera_frame(self, imu_msg: Imu) -> np.ndarray:
        """
        Extracts gravity vector in the Camera frame from MAVROS IMU orientation.
        Assuming MAVROS provides IMU orientation in ENU frame (Z is up).
        """
        # 1. Read IMU orientation (quaternion: x, y, z, w)
        q = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]
        
        # 2. Get rotation matrix from IMU to World (ENU)
        R_world_imu = quaternion_to_rotation_matrix(q)
        
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
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        # Get timestamp in seconds (ROS 2 Time object)
        timestamp = image_msg.header.stamp.sec + (image_msg.header.stamp.nanosec * 1e-9)
        
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
        
        dt_compute = time.perf_counter() - t0
        self.frame_count += 1
        
        # Compute Velocity if we have previous pose
        linear_vel = np.zeros(3)
        if self.prev_pose is not None and self.prev_time is not None:
            dt_time = timestamp - self.prev_time
            if dt_time > 0:
                # Linear velocity in WORLD frame
                v_world = (pose_wc.t - self.prev_pose.t) / dt_time
                # Transform to BODY frame (standard for Pixhawk)
                linear_vel = pose_wc.R.T @ v_world
                
        self.prev_pose = pose_wc
        self.prev_time = timestamp

        # ----------------------------------------------------
        # Publish PoseStamped
        # ----------------------------------------------------
        pose_msg = PoseStamped()
        pose_msg.header = image_msg.header
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = float(pose_wc.t[0])
        pose_msg.pose.position.y = float(pose_wc.t[1])
        pose_msg.pose.position.z = float(pose_wc.t[2])
        
        q_pose = rotation_matrix_to_quaternion(pose_wc.R)
        pose_msg.pose.orientation.x = float(q_pose[0])
        pose_msg.pose.orientation.y = float(q_pose[1])
        pose_msg.pose.orientation.z = float(q_pose[2])
        pose_msg.pose.orientation.w = float(q_pose[3])
        
        self.pose_pub.publish(pose_msg)

        # ----------------------------------------------------
        # Publish TwistStamped (Velocity)
        # ----------------------------------------------------
        vel_msg = TwistStamped()
        vel_msg.header = image_msg.header
        vel_msg.header.frame_id = "base_link"  # Velocity is in body frame
        
        vel_msg.twist.linear.x = float(linear_vel[0])
        vel_msg.twist.linear.y = float(linear_vel[1])
        vel_msg.twist.linear.z = float(linear_vel[2])
        # Angular velocity can be added here if needed using so3_log on R.T * R_prev
        
        self.vel_pub.publish(vel_msg)

        # Logging
        fps = 1.0 / max(1e-4, dt_compute)
        pos = pose_wc.t
        inliers = tracking_res.num_inliers if tracking_res else 0
        tag = "[KF]" if is_kf else "    "
        self.get_logger().info(f"[{self.frame_count:04d}] {tag} Pos: {pos[0]:+5.2f}, {pos[1]:+5.2f}, {pos[2]:+5.2f} | Vel: {linear_vel[0]:+5.2f} m/s | Inliers: {inliers:4d} | {dt_compute*1000:4.1f}ms ({fps:4.1f} FPS)")



def main(args=None):
    rclpy.init(args=args)
    node = UVORos2Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down UVO Node.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
