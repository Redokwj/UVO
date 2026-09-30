import os
import sys
import torch
import cv2
import numpy as np

# Add XoFTR and MINIMA to path
THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "thirdparty")
XOFTR_DIR = os.path.join(THIRD_PARTY_DIR, "XoFTR")
MINIMA_DIR = os.path.join(THIRD_PARTY_DIR, "MINIMA")

class CrossModalTracker:
    """
    Wrapper for Cross-Modal matchers like XoFTR (TIR-RGB) and MINIMA.
    """
    def __init__(self, method="xoftr", weights_path=None, match_threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.matcher = None
        
        if method == "xoftr":
            if XOFTR_DIR not in sys.path:
                sys.path.insert(0, XOFTR_DIR)
                
            from src.xoftr.xoftr import XoFTR
            from src.config.default import get_cfg_defaults
            from src.utils.data_io import lower_config
            config = get_cfg_defaults(inference=True)
            config = lower_config(config)
            config["xoftr"]["match_coarse"]["thr"] = match_threshold
            
            self.matcher = XoFTR(config=config["xoftr"]).to(self.device).eval()
            if weights_path and os.path.exists(weights_path):
                print(f"[CrossModal] Loading XoFTR weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.matcher.load_state_dict(state_dict, strict=False)
            else:
                print("[CrossModal] WARNING: No weights provided for XoFTR, using random init.")
        else:
            raise NotImplementedError(f"Method {method} not implemented yet.")

    @torch.no_grad()
    def match_images(self, img0: np.ndarray, img1: np.ndarray):
        """
        Directly match two images.
        img0: TIR or RGB image (np.ndarray)
        img1: RGB image (np.ndarray)
        Returns: (pts0, pts1) matching coordinates.
        """
        # Convert to grayscale if they are color
        if len(img0.shape) == 3:
            img0_g = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        else:
            img0_g = img0
            
        if len(img1.shape) == 3:
            img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_g = img1
            
        # Resize to multiple of 8 if needed (XoFTR needs divisible by 8)
        # We will just pass them through for now, assuming caller handles size
        h0, w0 = img0_g.shape
        h1, w1 = img1_g.shape
        
        # Convert to tensor [1, 1, H, W]
        t0 = torch.from_numpy(img0_g).float()[None, None].to(self.device) / 255.0
        t1 = torch.from_numpy(img1_g).float()[None, None].to(self.device) / 255.0
        
        data = {'image0': t0, 'image1': t1}
        
        if self.method == "xoftr":
            self.matcher(data)
            # Extracted matches are in data['mkpts0_f'] and data['mkpts1_f']
            mkpts0 = data.get('mkpts0_f', torch.empty(0, 2)).cpu().numpy()
            mkpts1 = data.get('mkpts1_f', torch.empty(0, 2)).cpu().numpy()
            return mkpts0, mkpts1
