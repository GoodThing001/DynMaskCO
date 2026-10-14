"""
JF2 Replanner — 把 FleetAssignmentHead 接进 strict-online joint assignment（Step 8-9）。

数据流（主控文档 §19）：
  FleetState → build_sound_candidate_set（sound hard mask）
             → build_jf2_features（veh/pair 特征 + base_score + candidate_mask）
             → [可选] base.encode → H → fleet_head → score = base_score + alpha*residual
             → greedy joint assignment（min score，容量重查）
             → per-vehicle greedy sequence

model/fleet_head=None：启发式路径（alpha=0，score=base_score）——**必须精确 == JF1-H**（Step 9 Gate-0）。
这是纯 NumPy 路径，可本地验证；model 路径需 JAX/GPU，服务器验证。

Author: JF2 Phase 1
Date: 2026-08-31
"""
import numpy as np

from strict_online_env import Replanner
from joint_fleet import JointAssignmentReplanner
from fleet_features import build_jf2_features


class JF2Replanner(Replanner):
    def __init__(self, dataset, capacity, tw_max=None, tw_speed=1.0,
                 model=None, fleet_head=None, alpha=None, decode_seed=42):
        self.capacity = capacity
        self.tw_max = tw_max if tw_max is not None else float(dataset['tw_end'].max())
        self.tw_speed = tw_speed
        self.model = model
        self.fleet_head = fleet_head
        self.alpha = alpha  # None = 用 head 里的 alpha；显式传入则覆盖（α=0 regression 用 0.0）
        self.decode_seed = decode_seed

        self.coords = dataset['coords'].astype(np.float32)
        self.demands = dataset['demands'].astype(np.float32)
        self.tw_start = dataset['tw_start'].astype(np.float32)
        self.tw_end = dataset['tw_end'].astype(np.float32)
        self.service_time = dataset.get('service_time', np.zeros_like(self.demands, np.float32))
        self.temp_class = dataset.get('temp_class', np.zeros_like(self.demands, np.int32))
        self.reveal_time = dataset.get('reveal_time', np.zeros_like(self.demands, np.float32))
        self.energy_mat = dataset.get('energy_mat', None)
        if self.energy_mat is not None:
            self.energy_mat = self.energy_mat.astype(np.float32)
        self.num_nodes = self.coords.shape[1]

        # 复用 JF1-H 的 _greedy_sequence（per-vehicle sequencing）
        self.inner = JointAssignmentReplanner('heuristic')

        # model 路径：lazy 建 encode_fn（避免纯启发式路径 import jax）
        self._encode_fn = None
        if model is not None:
            self._setup_model()

    def _setup_model(self):
        import jax
        import jax.numpy as jnp
        from cvrptw_utils import coord_normalize_visible
        model = self.model

        @jax.jit
        def encode_fn(raw_features, visible_mask=None, edge_feat=None):
            raw_features = raw_features.at[..., :2].set(
                coord_normalize_visible(raw_features[..., :2], visible_mask))
            return model.encode(raw_features, visible_mask=visible_mask, edge_feat=edge_feat)

        self._encode_fn = encode_fn

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        feat = build_jf2_features(env, inst_idx, clock, vehicles, served_mask,
                                  visible_ids, replan_ids)
        if not feat.active_vehicle_ids or not feat.customer_order:
            return
        score = self._score(env, inst_idx, feat, visible_ids, served_mask)  # [K, N]
        assignment = self._greedy_assign(env, inst_idx, feat, score)

        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = assignment.get(v.vehicle_id, [])
            suffix = self.inner._greedy_sequence(env, inst_idx, v, assigned)
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix

    def _score(self, env, inst_idx, feat, visible_ids, served_mask):
        """返回 [K, N] score。model=None 时 = base_score（启发式路径）。"""
        if self.model is None:
            return feat.base_score
        alpha = self.alpha if self.alpha is not None else 1.0  # 训练后 α=1.0（head 内部 self.alpha=0）
        if alpha == 0.0:
            return feat.base_score
        import jax.numpy as jnp
        H = self._encode(env, inst_idx, feat, visible_ids, served_mask)  # [1, N, D]
        anchor_ids = feat.anchor_ids[None, :]
        veh_feat = feat.veh_feat[None, :, :]
        pair_feat = feat.pair_feat[None, :, :, :]
        base_score = feat.base_score[None, :, :]
        cand_mask = feat.candidate_mask[None, :, :]
        _score, residual = self.fleet_head(H, anchor_ids, veh_feat, pair_feat,
                                           base_score, cand_mask)
        # head 返回的 score 用了 self.alpha（=0），这里用 α 重算 score = base + α·residual
        score = jnp.where(cand_mask, base_score + alpha * residual, -1e9)
        return np.array(score[0])

    def _encode(self, env, inst_idx, feat, visible_ids, served_mask):
        import jax.numpy as jnp
        vis_mask = np.zeros(self.num_nodes, dtype=bool)
        vis_mask[0] = True
        for vid in visible_ids:
            vis_mask[int(vid)] = True
        feat_arrays = [
            self.coords[inst_idx:inst_idx + 1],
            (self.demands[inst_idx:inst_idx + 1] / self.capacity)[..., None],
            (self.tw_start[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.tw_end[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.temp_class[inst_idx:inst_idx + 1] / 2.0)[..., None],
            (self.reveal_time[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
        ]
        raw = np.concatenate(feat_arrays, axis=-1).astype(np.float32)
        vis = vis_mask[None, ..., None]
        raw[..., 2:] = raw[..., 2:] * vis
        raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5
        edge_feat = self.energy_mat[inst_idx:inst_idx + 1] if self.energy_mat is not None else None
        H = np.array(self._encode_fn(
            jnp.array(raw), visible_mask=jnp.array(vis_mask[None]),
            edge_feat=jnp.array(edge_feat) if edge_feat is not None else None))
        return H

    def _greedy_assign(self, env, inst_idx, feat, score):
        """greedy joint assignment：按 customer_order 每客户选 min-score 可行车辆（容量重查）。"""
        vid_to_row = {vid: i for i, vid in enumerate(feat.active_vehicle_ids)}
        assignment = {vid: [] for vid in feat.active_vehicle_ids}
        assigned_load = {vid: 0.0 for vid in feat.active_vehicle_ids}
        for j in feat.customer_order:
            jidx = int(j)
            dj = float(env.demands[inst_idx, j])
            best_vid, best_s = None, -float('inf')
            for vid in feat.candidate.feasible_vehicles[j]:
                _node, _t, ld = feat.candidate.anchor_info[vid]
                if ld + assigned_load[vid] + dj > env.capacity + 1e-6:
                    continue
                s = float(score[vid_to_row[vid], jidx])
                if s > best_s:
                    best_s, best_vid = s, vid
            if best_vid is not None:
                assignment[best_vid].append(int(j))
                assigned_load[best_vid] += dj
        return assignment
