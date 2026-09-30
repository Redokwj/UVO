"""
Road Plane Geometric Prior (Module 2 stub).
Preserved strictly as a deactivated stub per architecture guidelines.
Zero active heuristics are executed to prevent runtime instability or bias on off-road terrain.
"""

from typing import Optional, Tuple
import numpy as np


class RoadPlanePriorStub:
    """
    Deactivated Road Plane Prior placeholder.
    UVO relies on direct foundation metric depth (UniDepth V2) rather than synthetic planar assumptions.
    """
    def __init__(self, is_enabled: bool = False):
        self.is_enabled = is_enabled

    def estimate_ground_plane(self, points_3d: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Always returns None as this heuristic is deactivated.
        """
        return None
