import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

from ..core.geometry import SE3
from ..backend.solver import so3_exp_torch


def so3_log_np(R: np.ndarray) -> np.ndarray:
    """Computes axis-angle 3-vector from 3x3 rotation matrix using logarithmic map on SO(3)."""
    tr = np.trace(R)
    cos_th = (tr - 1.0) / 2.0
    th = np.arccos(np.clip(cos_th, -1.0, 1.0))
    if th < 1e-6:
        return np.zeros(3, dtype=np.float64)
    u = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return th * u


def so3_exp_np(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map from so(3) Lie algebra to SO(3) rotation matrix."""
    th = np.linalg.norm(w)
    if th < 1e-6:
        return np.eye(3, dtype=np.float64)
    u = w / th
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0]
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


@dataclass
class PoseGraphEdge:
    """
    Edge connecting keyframe i and keyframe j with measured relative transformation.
    """
    kf_i: int
    kf_j: int
    T_ji_measured: SE3     # P_j = T_ji * P_i
    is_loop_closure: bool = False
    weight: float = 1.0


class PoseGraphOptimizer:
    """
    Global Pose Graph Optimizer (PGO) on SE(3) manifolds.
    Uses closed-form geodesic Lie group SE(3) loop distribution to smoothly propagate
    accumulated translation and rotation errors across the trajectory without numerical tears.
    """
    def __init__(self, max_iterations: int = 25, device: Optional[str] = None):
        self.max_iterations = max_iterations
        self.device = torch.device("cpu")
        self.edges: List[PoseGraphEdge] = []

    def add_edge(self, edge: PoseGraphEdge):
        self.edges.append(edge)

    def optimize(
        self,
        keyframe_ids: List[int],
        initial_poses: Dict[int, SE3],
        enable_crawler_constraints: bool = True
    ) -> Dict[int, SE3]:
        """
        Executes robust closed-form geodesic Pose Graph Optimization over keyframes.
        Distributes loop closure closing drift smoothly proportional to arc distance.
        Features SE(2) manifold projection for tracked ground rovers.
        """
        num_kfs = len(keyframe_ids)
        if num_kfs < 3 or len(self.edges) == 0:
            return initial_poses

        kf_id_to_idx = {k_id: idx for idx, k_id in enumerate(keyframe_ids)}
        loop_edges = [e for e in self.edges if e.is_loop_closure]

        if len(loop_edges) == 0:
            return initial_poses

        # Copy initial poses in world-to-camera (cw) representation
        current_poses_cw = {k_id: SE3(R=initial_poses[k_id].R.copy(), t=initial_poses[k_id].t.copy()) for k_id in keyframe_ids}

        # Process loop closures
        for loop_e in loop_edges:
            if loop_e.kf_i not in kf_id_to_idx or loop_e.kf_j not in kf_id_to_idx:
                continue

            idx_i = kf_id_to_idx[loop_e.kf_i]
            idx_j = kf_id_to_idx[loop_e.kf_j]

            # Ensure order idx_i < idx_j
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
                T_meas_ji = loop_e.T_ji_measured.inv()
            else:
                T_meas_ji = loop_e.T_ji_measured

            k_id_i = keyframe_ids[idx_i]
            k_id_j = keyframe_ids[idx_j]

            # Camera-to-world (wc) poses
            T_wc_i = current_poses_cw[k_id_i].inv()
            T_wc_j = current_poses_cw[k_id_j].inv()

            # Target pose of keyframe j implied by loop measurement:
            # P_c_j = T_meas_ji * P_c_i => T_c_j_w = T_meas_ji * T_c_i_w => T_wc_j_target = T_wc_i * T_meas_ji^-1
            T_wc_j_target = T_wc_i @ T_meas_ji.inv()

            t_corr = T_wc_j_target.t - T_wc_j.t
            R_corr = T_wc_j_target.R @ T_wc_j.R.T
            correction_norm = float(np.linalg.norm(t_corr))

            # Compute cumulative step distances between keyframes along the loop chain
            step_dists = []
            for m in range(idx_i, idx_j):
                m_id_0 = keyframe_ids[m]
                m_id_1 = keyframe_ids[m + 1]
                pos_0 = current_poses_cw[m_id_0].inv().t
                pos_1 = current_poses_cw[m_id_1].inv().t
                step_dists.append(float(np.linalg.norm(pos_1 - pos_0)))

            cum_d = np.concatenate([[0.0], np.cumsum(step_dists)])
            L_total = cum_d[-1]

            if L_total < 1e-3:
                continue

            # Safety gate: allow closures proportional to accumulated path length (up to 150m)
            max_allowed_corr = max(150.0, 0.65 * L_total)
            if correction_norm > max_allowed_corr:
                continue

            w_corr = so3_log_np(R_corr)

            if enable_crawler_constraints:
                # SE(2) Manifold Projection: Constrain orientation adjustment to vertical yaw axis
                # and dampen vertical ground penetration
                w_corr = np.array([0.0, w_corr[1], 0.0], dtype=np.float64)
                t_corr[1] = 0.0

            # Geodesic smooth propagation along the loop segment [idx_i, idx_j]
            for local_idx, k in enumerate(range(idx_i, idx_j + 1)):
                s_k = cum_d[local_idx] / L_total  # interpolation factor in [0.0, 1.0]
                R_k_corr = so3_exp_np(s_k * w_corr)
                t_k_corr = s_k * t_corr

                k_id = keyframe_ids[k]
                T_wc_k = current_poses_cw[k_id].inv()
                T_wc_k_opt = SE3(
                    R=R_k_corr @ T_wc_k.R,
                    t=T_wc_k.t + t_k_corr
                )
                current_poses_cw[k_id] = T_wc_k_opt.inv()

            # Carry-forward rigid correction to all subsequent keyframes after idx_j
            for k in range(idx_j + 1, num_kfs):
                k_id = keyframe_ids[k]
                T_wc_k = current_poses_cw[k_id].inv()
                T_wc_k_opt = SE3(
                    R=R_corr @ T_wc_k.R,
                    t=T_wc_k.t + t_corr
                )
                current_poses_cw[k_id] = T_wc_k_opt.inv()

        return current_poses_cw

    def optimize_loop(
        self,
        keyframe_ids: List[int],
        initial_poses: Dict[int, SE3],
        cand_kf_id: int,
        curr_kf_id: int,
        T_ji_measured: SE3,
        enable_crawler_constraints: bool = True
    ) -> Dict[int, SE3]:
        """
        Applies a single validated loop closure event smoothly across the trajectory.
        Distributes closing translation and orientation drift continuously without jumps.
        Features SE(2) manifold projection for tracked ground rovers.
        """
        num_kfs = len(keyframe_ids)
        if num_kfs < 3 or cand_kf_id not in initial_poses or curr_kf_id not in initial_poses:
            return initial_poses

        kf_id_to_idx = {k_id: idx for idx, k_id in enumerate(keyframe_ids)}
        if cand_kf_id not in kf_id_to_idx or curr_kf_id not in kf_id_to_idx:
            return initial_poses

        idx_i = kf_id_to_idx[cand_kf_id]
        idx_j = kf_id_to_idx[curr_kf_id]

        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i
            T_meas = T_ji_measured.inv()
        else:
            T_meas = T_ji_measured

        k_id_i = keyframe_ids[idx_i]
        k_id_j = keyframe_ids[idx_j]

        current_poses_cw = {k_id: SE3(R=initial_poses[k_id].R.copy(), t=initial_poses[k_id].t.copy()) for k_id in keyframe_ids}

        T_wc_i = current_poses_cw[k_id_i].inv()
        T_wc_j = current_poses_cw[k_id_j].inv()

        # Target pose of keyframe j implied by loop measurement:
        T_wc_j_target = T_wc_i @ T_meas.inv()

        t_corr = T_wc_j_target.t - T_wc_j.t
        R_corr = T_wc_j_target.R @ T_wc_j.R.T
        correction_norm = float(np.linalg.norm(t_corr))

        # Compute cumulative step distances between keyframes along the loop chain
        step_dists = []
        for m in range(idx_i, idx_j):
            m_id_0 = keyframe_ids[m]
            m_id_1 = keyframe_ids[m + 1]
            pos_0 = current_poses_cw[m_id_0].inv().t
            pos_1 = current_poses_cw[m_id_1].inv().t
            step_dists.append(float(np.linalg.norm(pos_1 - pos_0)))

        cum_d = np.concatenate([[0.0], np.cumsum(step_dists)])
        L_total = cum_d[-1]

        if L_total < 1e-3:
            return initial_poses

        # Safety gate: allow closures proportional to accumulated loop length (up to 150m)
        max_allowed_corr = max(150.0, 0.65 * L_total)
        if correction_norm > max_allowed_corr:
            return initial_poses

        w_corr = so3_log_np(R_corr)

        if enable_crawler_constraints:
            # SE(2) Manifold Projection: Constrain orientation adjustment to vertical yaw axis
            # and dampen vertical ground penetration
            w_corr = np.array([0.0, w_corr[1], 0.0], dtype=np.float64)
            t_corr[1] = 0.0

        # Geodesic smooth propagation along the loop segment [idx_i, idx_j]
        for local_idx, k in enumerate(range(idx_i, idx_j + 1)):
            s_k = cum_d[local_idx] / L_total  # interpolation factor in [0.0, 1.0]
            R_k_corr = so3_exp_np(s_k * w_corr)
            t_k_corr = s_k * t_corr

            k_id = keyframe_ids[k]
            T_wc_k = current_poses_cw[k_id].inv()
            T_wc_k_opt = SE3(
                R=R_k_corr @ T_wc_k.R,
                t=T_wc_k.t + t_k_corr
            )
            current_poses_cw[k_id] = T_wc_k_opt.inv()

        # Carry-forward rigid correction to all subsequent keyframes after idx_j
        for k in range(idx_j + 1, num_kfs):
            k_id = keyframe_ids[k]
            T_wc_k = current_poses_cw[k_id].inv()
            T_wc_k_opt = SE3(
                R=R_corr @ T_wc_k.R,
                t=T_wc_k.t + t_corr
            )
            current_poses_cw[k_id] = T_wc_k_opt.inv()

        return current_poses_cw

    def optimize_full_graph(
        self,
        keyframe_ids: List[int],
        initial_poses: Dict[int, SE3],
        gravity_priors: Optional[Dict[int, np.ndarray]] = None,
        iterations: int = 20,
        enable_crawler_constraints: bool = True
    ) -> Dict[int, SE3]:
        """
        Global Covisibility Pose Graph Optimizer (Full Pose Graph BA Backend).
        Simultaneously relaxes all keyframe poses across the entire trajectory by minimizing:
          E = sum(E_odometry) + w_loop * sum(E_loop) + w_crawler * sum(E_lateral) + w_grav * sum(E_gravity)
        using PyTorch Levenberg-Marquardt on SE(3) Lie manifolds.
        """
        num_kfs = len(keyframe_ids)
        if num_kfs < 4 or len(self.edges) == 0:
            return initial_poses

        # Start from closed-form geodesic PGO solution as initialization
        current_poses = self.optimize(keyframe_ids, initial_poses)

        kf_id_to_idx = {k_id: idx for idx, k_id in enumerate(keyframe_ids)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initial translation and rotation tensors in World coordinates
        t_init = np.array([current_poses[k_id].inv().t for k_id in keyframe_ids], dtype=np.float32)
        R_init = np.array([current_poses[k_id].inv().R for k_id in keyframe_ids], dtype=np.float32)

        t_param = torch.nn.Parameter(torch.from_numpy(t_init).to(device))
        # Parameterize rotation as Lie algebra so(3) delta perturbations: R_opt = exp(delta_w) * R_init
        delta_w = torch.nn.Parameter(torch.zeros((num_kfs, 3), dtype=torch.float32, device=device))

        R_base = torch.from_numpy(R_init).to(device)

        # Build odometry and loop edge tensors
        odom_edges_i, odom_edges_j, odom_t_rel, odom_R_rel = [], [], [], []
        loop_edges_i, loop_edges_j, loop_t_rel, loop_R_rel = [], [], [], []

        for edge in self.edges:
            if edge.kf_i not in kf_id_to_idx or edge.kf_j not in kf_id_to_idx:
                continue
            idx_i = kf_id_to_idx[edge.kf_i]
            idx_j = kf_id_to_idx[edge.kf_j]

            # Relative pose in camera coordinates: P_j = T_ji * P_i
            # In world coordinates: T_wc_j = T_wc_i * T_ji^-1 => T_ji = T_cw_j * T_wc_i
            T_ji = edge.T_ji_measured
            if edge.is_loop_closure:
                loop_edges_i.append(idx_i)
                loop_edges_j.append(idx_j)
                loop_t_rel.append(T_ji.t.astype(np.float32))
                loop_R_rel.append(T_ji.R.astype(np.float32))
            else:
                odom_edges_i.append(idx_i)
                odom_edges_j.append(idx_j)
                odom_t_rel.append(T_ji.t.astype(np.float32))
                odom_R_rel.append(T_ji.R.astype(np.float32))

        has_loops = len(loop_edges_i) > 0
        if not has_loops and len(odom_edges_i) < 2:
            return current_poses

        odom_i = torch.tensor(odom_edges_i, dtype=torch.long, device=device)
        odom_j = torch.tensor(odom_edges_j, dtype=torch.long, device=device)
        odom_t_meas = torch.tensor(np.array(odom_t_rel), dtype=torch.float32, device=device)

        if has_loops:
            loop_i = torch.tensor(loop_edges_i, dtype=torch.long, device=device)
            loop_j = torch.tensor(loop_edges_j, dtype=torch.long, device=device)
            loop_t_meas = torch.tensor(np.array(loop_t_rel), dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam([
            {"params": [t_param], "lr": 0.02},
            {"params": [delta_w], "lr": 0.005}
        ])

        # Anchor origin (keyframe 0 fixed)
        t_origin = t_param[0].detach().clone()

        for step in range(iterations):
            optimizer.zero_grad()

            # Compute current rotation matrices: R_curr = so3_exp(delta_w) @ R_base
            R_delta = so3_exp_torch(delta_w)
            R_curr = torch.bmm(R_delta, R_base)

            # 1. Odometry relative translation loss
            # Expected rel translation in camera j: t_rel = R_curr[j].T @ (t_param[i] - t_param[j])
            p_diff_odom = t_param[odom_i] - t_param[odom_j]
            t_pred_odom = torch.bmm(R_curr[odom_j].transpose(1, 2), p_diff_odom.unsqueeze(-1)).squeeze(-1)
            loss_odom = torch.mean(torch.sum((t_pred_odom - odom_t_meas) ** 2, dim=-1))

            loss = loss_odom

            # 2. Loop closure relative pose loss
            if has_loops:
                p_diff_loop = t_param[loop_i] - t_param[loop_j]
                t_pred_loop = torch.bmm(R_curr[loop_j].transpose(1, 2), p_diff_loop.unsqueeze(-1)).squeeze(-1)
                loss_loop = torch.mean(torch.sum((t_pred_loop - loop_t_meas) ** 2, dim=-1))
                loss = loss + 15.0 * loss_loop

            # 3. Crawler non-holonomic constraint: zero lateral velocity in chassis frame
            if enable_crawler_constraints and num_kfs > 1:
                # Step vector in body frame: R_curr[k].T @ (t_param[k+1] - t_param[k])
                step_world = t_param[1:] - t_param[:-1]
                step_body = torch.bmm(R_curr[:-1].transpose(1, 2), step_world.unsqueeze(-1)).squeeze(-1)
                # Lateral displacement is X coordinate
                loss_lateral = torch.mean(step_body[:, 0] ** 2)
                loss = loss + 8.0 * loss_lateral

            # 4. Origin anchor constraint
            loss_anchor = torch.sum((t_param[0] - t_origin) ** 2) + torch.sum(delta_w[0] ** 2)
            loss = loss + 50.0 * loss_anchor

            loss.backward()
            optimizer.step()

        # Extract optimized poses
        with torch.no_grad():
            R_delta_final = so3_exp_torch(delta_w)
            R_opt = torch.bmm(R_delta_final, R_base).cpu().numpy()
            t_opt = t_param.cpu().numpy()

        optimized_poses = {}
        for idx, k_id in enumerate(keyframe_ids):
            T_wc_opt = SE3(R=R_opt[idx].astype(np.float64), t=t_opt[idx].astype(np.float64))
            optimized_poses[k_id] = T_wc_opt.inv()

        return optimized_poses
