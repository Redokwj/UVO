import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional

from uvo.core.frame import Frame, Keyframe
from uvo.core.geometry import SE3
from uvo.frontend.xfeat_tracker import XFeatTracker
from uvo.loop_closure.vpr import VPRDatabase


class VTRLocalizer:
    """
    Visual Teach & Repeat (VT&R) Localizer.
    Loads a previously recorded map (Teach pass) and provides relative localization
    for the Follower robot (Repeat pass).
    """
    def __init__(self, teach_keyframes: List[Keyframe], intrinsics):
        self.teach_keyframes = teach_keyframes
        self.intrinsics = intrinsics
        
        # We need a matcher to match the current camera view against the teach keyframes
        self.matcher = XFeatTracker(top_k=2000)
        
        # Initialize VPR Database with the Teach Map for fast retrieval
        self.vpr = VPRDatabase()
        
        for kf in self.teach_keyframes:
            self.vpr.add_keyframe(kf)

    def localize(self, current_image: np.ndarray, current_timestamp: float) -> Optional[Tuple[Keyframe, SE3, dict]]:
        """
        Localize the current image against the Teach map.
        Returns:
            - closest_teach_kf: The keyframe from the teach map we matched against.
            - T_curr_teach: SE3 transform from Teach Keyframe to Current Robot pose.
            - debug_info: Dictionary with inlier count, tracking status, etc.
        """
        # 1. Query VPR to find the closest Keyframe in the recorded map
        # We search top-1 match
        query_desc = self.vpr.vpr_engine.extract_descriptor(current_image)
        matches = self.vpr.query(query_desc, current_kf_id=999999, min_gap_distance_m=0.0, min_gap_frames=0, top_k=3)
        if not matches:
            return None
            
        best_match_id = matches[0][0]
        # Find the actual keyframe object
        teach_kf = next((kf for kf in self.teach_keyframes if kf.frame_id == best_match_id), None)
        if teach_kf is None:
            return None

        # 2. Extract features from current image and match against Teach Keyframe
        curr_frame = Frame(frame_id=-1, timestamp=current_timestamp, image=current_image, intrinsics=self.intrinsics)
        self.matcher.extract_features(curr_frame)
        
        if teach_kf.frame.descriptors is None:
            self.matcher.extract_features(teach_kf.frame)
            
        pts_curr, pts_teach = self.matcher.match_features(
            curr_frame.descriptors, teach_kf.frame.descriptors,
            curr_frame.keypoints, teach_kf.frame.keypoints
        )
        if len(pts_curr) < 20:
            return None

        # 3. Retrieve 3D coordinates for the matched points from the Teach Keyframe
        # The Teach Keyframe has a depth map (from UniDepth) that gives physical 3D scale.
        obj_pts_3d = []
        img_pts_2d = []
        
        for i in range(len(pts_teach)):
            u, v = int(pts_teach[i][0]), int(pts_teach[i][1])
            # Ensure within image bounds
            if 0 <= v < teach_kf.depth_map.shape[0] and 0 <= u < teach_kf.depth_map.shape[1]:
                z = teach_kf.depth_map[v, u]
                if z > 0.1 and z < 30.0: # valid depth range
                    # Backproject Teach pixel to 3D point in Teach Camera frame
                    x = (u - self.intrinsics.cx) * z / self.intrinsics.fx
                    y = (v - self.intrinsics.cy) * z / self.intrinsics.fy
                    
                    obj_pts_3d.append([x, y, z])
                    img_pts_2d.append(pts_curr[i])
                    
        if len(obj_pts_3d) < 15:
            return None

        obj_pts_3d = np.array(obj_pts_3d, dtype=np.float32)
        img_pts_2d = np.array(img_pts_2d, dtype=np.float32)

        # 4. Solve PnP to find relative pose
        # This gives us T_curr_teach: Position of the CURRENT camera in the TEACH camera's coordinate frame!
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts_3d, 
            img_pts_2d, 
            self.intrinsics.to_matrix(), 
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 10:
            return None

        R_curr_teach, _ = cv2.Rodrigues(rvec)
        T_curr_teach = SE3(R_curr_teach, tvec.ravel())

        debug_info = {
            "inliers": len(inliers),
            "total_matches": len(obj_pts_3d)
        }

        return teach_kf, T_curr_teach, debug_info
