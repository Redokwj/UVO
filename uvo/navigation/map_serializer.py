import pickle
import cv2
import numpy as np
from typing import List

from uvo.core.frame import Frame, Keyframe, CameraIntrinsics
from uvo.core.geometry import SE3

def save_vtr_map(keyframes: List[Keyframe], filepath: str):
    """
    Serializes a list of Keyframes into a compact file for Teach & Repeat.
    Compresses RGB images and casts Depth maps to float16 to save space.
    """
    compressed_kfs = []
    for kf in keyframes:
        # 1. Compress RGB Image to JPEG
        img_encoded = None
        if kf.frame.image is not None:
            _, img_encoded = cv2.imencode('.jpg', kf.frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
        # 2. Compress Depth Map to float16 (saves 50% space, enough precision for VT&R)
        depth_map = None
        if kf.depth_map is not None:
            depth_map = kf.depth_map.astype(np.float16)

        data = {
            'frame_id': kf.frame_id,
            'timestamp': kf.timestamp,
            'image_jpg': img_encoded,
            'depth_map': depth_map,
            'pose_cw_R': kf.pose_cw.R,
            'pose_cw_t': kf.pose_cw.t,
            'intrinsics': (kf.frame.intrinsics.fx, kf.frame.intrinsics.fy, kf.frame.intrinsics.cx, kf.frame.intrinsics.cy),
            'vpr_descriptor': kf.vpr_descriptor,
            'accumulated_distance_m': kf.accumulated_distance_m
        }
        compressed_kfs.append(data)
        
    with open(filepath, 'wb') as f:
        pickle.dump(compressed_kfs, f)
    print(f"[VTR Map] Saved {len(compressed_kfs)} keyframes to {filepath}")

def load_vtr_map(filepath: str) -> List[Keyframe]:
    """
    Deserializes a Teach & Repeat map from file back into full Keyframe objects.
    """
    with open(filepath, 'rb') as f:
        compressed_kfs = pickle.load(f)
        
    keyframes = []
    for data in compressed_kfs:
        # 1. Decompress RGB Image
        image = None
        if data['image_jpg'] is not None:
            image = cv2.imdecode(data['image_jpg'], cv2.IMREAD_COLOR)
            
        # 2. Restore Depth Map to float32
        depth_map = None
        if data['depth_map'] is not None:
            depth_map = data['depth_map'].astype(np.float32)
            
        # 3. Restore Intrinsics and Pose
        intrinsics = CameraIntrinsics.from_tuple(data['intrinsics'])
        pose_cw = SE3(data['pose_cw_R'], data['pose_cw_t'])
        
        # 4. Construct Frame and Keyframe
        frame = Frame(
            frame_id=data['frame_id'],
            timestamp=data['timestamp'],
            image=image,
            intrinsics=intrinsics,
            pose_cw=pose_cw,
            is_keyframe=True
        )
        
        kf = Keyframe(frame=frame)
        kf.depth_map = depth_map
        kf.vpr_descriptor = data['vpr_descriptor']
        kf.accumulated_distance_m = data['accumulated_distance_m']
        
        keyframes.append(kf)
        
    print(f"[VTR Map] Loaded {len(keyframes)} keyframes from {filepath}")
    return keyframes
