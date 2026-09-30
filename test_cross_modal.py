import cv2
import numpy as np
import time
import os
import sys

# Path to cross_modal.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'uvo', 'frontend'))

from uvo.frontend.cross_modal import CrossModalTracker

def main():
    weights = "thirdparty/MINIMA/weights/minima_xoftr.ckpt"
    tracker = CrossModalTracker(method="xoftr", weights_path=weights)

    img0 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    img1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    t0 = time.time()
    pts0, pts1 = tracker.match_images(img0, img1)
    t1 = time.time()
    print(f"Matched {len(pts0)} points in {(t1-t0)*1000:.2f} ms")

if __name__ == "__main__":
    main()
