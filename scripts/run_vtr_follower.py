#!/usr/bin/env python3
"""
Visual Teach & Repeat (VT&R) Follower Node for Tracked Rovers (ROS 2)
Implements a non-linear Lyapunov-inspired controller for Skid-Steer / Differential Drive.
"""

import sys
import os
import time
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
except ImportError:
    print("ROS 2 is required to run this VT&R script.")
    sys.exit(1)

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)

from uvo.navigation.localizer import VTRLocalizer
from uvo.navigation.map_serializer import load_vtr_map
from uvo.core.frame import CameraIntrinsics

class VTRFollowerNode(Node):
    def __init__(self):
        super().__init__('vtr_follower_node')
        
        self.bridge = CvBridge()
        
        # 1. Controller Parameters
        # Proportional gain for lateral (cross-track) error
        self.k_y = self.declare_parameter("k_y", 1.5).value
        # Proportional gain for heading error
        self.k_theta = self.declare_parameter("k_theta", 2.0).value
        # Base forward velocity
        self.v_ref = self.declare_parameter("v_ref", 0.5).value # m/s
        
        # 2. Setup ROS Interface
        image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        map_path = self.declare_parameter("map_path", "teach_map.pkl").value
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Intrinsics
        fx = self.declare_parameter("fx", 778.58).value
        fy = self.declare_parameter("fy", 1127.79).value
        cx = self.declare_parameter("cx", 343.86).value
        cy = self.declare_parameter("cy", 311.17).value
        self.intrinsics = CameraIntrinsics.from_tuple((fx, fy, cx, cy), width=704, height=576)
        
        # 3. Load Teach Map
        self.get_logger().info(f"Loading VT&R Map from {map_path} ...")
        try:
            teach_keyframes = load_vtr_map(map_path)
            self.localizer = VTRLocalizer(teach_keyframes, self.intrinsics)
            self.get_logger().info(f"VT&R Map loaded successfully! Found {len(teach_keyframes)} keyframes.")
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")
            self.localizer = None

    def extract_yaw_from_matrix(self, R: np.ndarray) -> float:
        """
        Extracts yaw angle (rotation around Y-axis) for OpenCV camera frame (X right, Y down, Z forward).
        """
        # Yaw is approximately the angle in the X-Z plane
        return math.atan2(-R[0, 2], R[0, 0])

    def image_callback(self, msg: Image):
        if self.localizer is None:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        t0 = time.perf_counter()
        
        # 1. Localize against the Teach Map
        loc_res = self.localizer.localize(cv_image, timestamp)
        if loc_res is None:
            self.get_logger().warn("Lost track! Stopping rover.")
            self.stop_rover()
            return
            
        teach_kf, T_curr_teach, debug = loc_res
        
        # 2. Extract Errors for Skid-Steer Controller
        # T_curr_teach transforms points from Teach to Current frame.
        # Translation vector t gives the position of the Teach frame origin in Current frame coordinates.
        # In OpenCV: X is right, Y is down, Z is forward.
        
        t = T_curr_teach.t
        R = T_curr_teach.R
        
        # Cross-track error (Lateral offset). 
        # If t[0] is positive, the Teach path is to our Right.
        e_y = float(t[0])
        
        # Heading error (Yaw offset).
        # We want our camera to align with the Teach camera.
        e_theta = self.extract_yaw_from_matrix(R)
        
        # 3. Non-Linear Skid-Steer Tracking Control Law
        # Target heading adjusts based on lateral error (Steering into the path)
        theta_target = math.atan(self.k_y * e_y)
        
        # Calculate Angular Velocity command
        omega = self.k_theta * (theta_target - e_theta)
        
        # Calculate Linear Velocity (slow down if heading error is large)
        # Allows "turning in place" if error is too big.
        v = self.v_ref * math.cos(e_theta)
        if abs(e_theta) > 0.5: # ~30 degrees
            v = 0.0 # Stop and turn in place
            
        # 4. Publish Command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)
        
        dt = time.perf_counter() - t0
        self.get_logger().info(f"VT&R: Lat Err: {e_y*100:+.1f}cm | Yaw Err: {math.degrees(e_theta):+.1f}deg | v: {v:.2f} m/s | w: {omega:.2f} rad/s | {dt*1000:.1f}ms")

    def stop_rover(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = VTRFollowerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_rover()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
