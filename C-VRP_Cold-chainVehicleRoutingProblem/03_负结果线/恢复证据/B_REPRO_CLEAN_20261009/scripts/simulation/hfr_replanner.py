"""
HFR-M0 Step 4 — Hierarchical Fleet–Route Replanner（inference，导师 §25-41）。

把训练好的 GroupingHead + RouteResidualAdapter（+ frozen base MaskCO）接进 strict-online
joint recourse。核心：把决策空间从 K 辆物理车降到「等价 route slot」，让 idle@depot symmetry
直接在 action space 消失（§27-28）。

推理闭环（§39 / §53.5）：
  FleetState → build_slots（anchored ready 车 + anonymous NEW_ROUTE idle pool）
             → HFR forward（H → G + A_joint；adjmat=None + timestep=0 与训练一致）
             → joint slot assignment（customer 顺序 = JF1-H pool 顺序；S = rank(GroupAffinity)
                + rank(RouteAffinity)，min-travel 只做 tie-break，§33-34）
             → ordered insertion（同一个 A_joint，§37-38；非 min-travel greedy）
             → no silent drop（unresolved → service-first fallback 到 JF1-H，§41）
             → write v.mutable_suffix（WAIT 语义复用 env，§40）

关键原则（§56）：
  - G/A 只 rank feasible candidates，绝不进 hard mask（§35）。
  - partition 与 route 用同一个 A_joint（§38）。
  - NEW_ROUTE 匿名（min idle id 只做 deterministic physical matching，不进 model target，§28）。
  - 不复刻第二套 StrictOnline FleetState（复用 env + joint_fleet 语义，§40）。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from strict_online_env import Replanner
from joint_fleet import JointAssignmentReplanner, get_global_pool


@dataclass
class FleetSlot:
    kind: str                     # 'anchored' / 'new_route'
    vehicle_id: Optional[int]
    anchor_node: int
    anchor_time: float
    anchor_load: float
    members: List[int] = field(default_factory=list)   # 有序（插入顺序即 route 顺序）


def rank_norm(values):
    """rank normalization（§34）：average rank for ties，映射到 [0,1]。"""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks / (n - 1)


class HFRReplanner(Replanner):
    def __init__(self, dataset, capacity, tw_max=None, tw_speed=1.0,
                 model=None, group_head=None, route_adapter=None, beta=1.0,
                 partition_mode='group', insert_mode='affinity'):
        self.capacity = capacity
        self.tw_max = tw_max if tw_max is not None else float(dataset['tw_end'].max())
        self.tw_speed = tw_speed
        self.model = model
        self.group_head = group_head
        self.route_adapter = route_adapter
        self.beta = beta
        # 2×2 消融开关：partition ∈ {group, travel}，insert ∈ {affinity, travel}。
        self.partition_mode = partition_mode
        self.insert_mode = insert_mode

        self.coords = dataset['coords'].astype(np.float32)
        self.demands = dataset['demands'].astype(np.float32)
        self.tw_start = dataset['tw_start'].astype(np.float32)
        self.tw_end = dataset['tw_end'].astype(np.float32)
        self.service_time = dataset.get('service_time', np.zeros_like(self.demands, np.float32))
        self.temp_class = dataset.get('temp_class', np.zeros_like(self.demands, np.int32))
        self.reveal_time = dataset.get('reveal_time', np.zeros_like(self.demands, np.float32))
        self.num_nodes = self.coords.shape[1]

        self.inner = JointAssignmentReplanner('heuristic')  # fallback + 复用 pool 顺序

        self._hf_fn = None
        if model is not None:
            self._setup_model()

    def _setup_model(self):
        import jax
        import jax.numpy as jnp
        from cvrptw_utils import coord_normalize_visible
        model, gh, ra, beta = self.model, self.group_head, self.route_adapter, self.beta

        @jax.jit
        def hf_forward(raw_features, visible_mask):
            raw_features = raw_features.at[..., :2].set(
                coord_normalize_visible(raw_features[..., :2], visible_mask))
            H = model.encode(raw_features, visible_mask=visible_mask)
            G = gh(H)
            A_base = model.decode(H, jnp.zeros((raw_features.shape[0],), dtype=jnp.float32), None)
            delta = ra(H, G)
            A_logit = beta * jax.nn.log_sigmoid(G)       # 纯 same-route 偏置（无 ΔA 无 A_base）
            A_group = delta + A_logit                    # ΔA + β·log_sigmoid(G)
            A_joint = A_base + A_group
            return G, A_joint, A_group, A_logit

        self._hf_fn = hf_forward

    def _compute_GA(self, inst_idx, visible_ids):
        """HFR forward → (G [N,N], A_joint [N,N])，与训练一致（edge_feat=None, adjmat=None）。"""
        import jax.numpy as jnp
        vis_mask = np.zeros(self.num_nodes, dtype=bool)
        vis_mask[0] = True
        for vid in visible_ids:
            vis_mask[int(vid)] = True
        feat = [
            self.coords[inst_idx:inst_idx + 1],
            (self.demands[inst_idx:inst_idx + 1] / self.capacity)[..., None],
            (self.tw_start[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.tw_end[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.temp_class[inst_idx:inst_idx + 1] / 2.0)[..., None],
            (self.reveal_time[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
        ]
        raw = np.concatenate(feat, axis=-1).astype(np.float32)
        vis = vis_mask[None, ..., None]
        raw[..., 2:] = raw[..., 2:] * vis
        raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5
        G, A_joint, A_group, A_logit = self._hf_fn(jnp.array(raw), jnp.array(vis_mask[None]))
        return np.array(G[0]), np.array(A_joint[0]), np.array(A_group[0]), np.array(A_logit[0])

    def _route_feasible(self, env, inst_idx, anchor_node, anchor_time, anchor_load, members):
        """模拟整条 route 的时间/容量/return 可行性（full check，no silent drop）。"""
        node = int(anchor_node)
        t = float(anchor_time)
        load = float(anchor_load)
        for c in members:
            c = int(c)
            arr = t + env.dist_mat[inst_idx, node, c] / env.tw_speed
            start = max(arr, env.tw_start[inst_idx, c])
            if start > env.tw_end[inst_idx, c] + 1e-6:
                return False
            if load + env.demands[inst_idx, c] > env.capacity + 1e-6:
                return False
            load += env.demands[inst_idx, c]
            t = start + env.service_time[inst_idx, c]
            node = c
        ret = t + env.dist_mat[inst_idx, node, 0] / env.tw_speed
        return ret <= env.tw_end[inst_idx, 0] + 1e-6

    def _insertion_gains(self, env, inst_idx, slot, j, A):
        """所有 feasible 插入位置 (p, gain)。insert_mode：
          'affinity' → gain = A[u_p,j]+A[j,u_{p+1}]-A[u_p,u_{p+1}]（§32，A_joint）
          'travel'   → gain = d[u_p,u_{p+1}] - (d[u_p,j]+d[j,u_{p+1}])（cheapest insertion）"""
        route = [slot.anchor_node] + list(slot.members) + [0]
        gains = []
        for p in range(len(slot.members) + 1):
            up = int(route[p]); un = int(route[p + 1])
            new_members = list(slot.members[:p]) + [int(j)] + list(slot.members[p:])
            if self._route_feasible(env, inst_idx, slot.anchor_node, slot.anchor_time,
                                    slot.anchor_load, new_members):
                if self.insert_mode == 'travel':
                    d_new = float(env.dist_mat[inst_idx, up, int(j)] + env.dist_mat[inst_idx, int(j), un])
                    d_old = float(env.dist_mat[inst_idx, up, un])
                    gain = d_old - d_new
                else:  # 'affinity'(A_joint) / 'group_only'(A_group) / 'group_logit'(A_logit)
                    gain = float(A[up, int(j)] + A[int(j), un] - A[up, un])
                gains.append((p, gain))
        return gains

    def _group_affinity(self, slot, j, G):
        if not slot.members:
            return 0.0
        return float(sum(G[int(i), int(j)] for i in slot.members) / len(slot.members))

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        pool, _locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids)

        # build_slots：anchored = ready@customer，idle_pool = idle@depot（anonymous NEW_ROUTE）
        slots = []
        idle_pool = []
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            if v.status == 'idle':
                idle_pool.append(int(v.vehicle_id))
            else:
                slots.append(FleetSlot('anchored', int(v.vehicle_id), int(v.current_node),
                                       float(v.ready_time), float(v.current_load)))

        if not slots and not idle_pool:
            return

        if pool:
            G, A, A_group, A_logit = self._compute_GA(inst_idx, visible_ids)
            if self.insert_mode == 'group_only':
                A_insert = A_group
            elif self.insert_mode == 'group_logit':
                A_insert = A_logit
            else:
                A_insert = A

            unresolved = []
            for j in pool:
                j = int(j)
                cands = []  # (slot_or_marker, best_p, best_gain, travel)
                for s in slots:
                    gains = self._insertion_gains(env, inst_idx, s, j, A_insert)
                    if not gains:
                        continue
                    best_p, best_gain = max(gains, key=lambda gp: gp[1])
                    travel = float(env.dist_mat[inst_idx, s.anchor_node, j]) / env.tw_speed
                    cands.append((s, best_p, best_gain, travel))
                if idle_pool and self._route_feasible(env, inst_idx, 0, float(clock), 0.0, [j]):
                    travel = float(env.dist_mat[inst_idx, 0, j]) / env.tw_speed
                    cands.append(('NEW_ROUTE', 0, 0.0, travel))

                if not cands:
                    unresolved.append(j)
                    continue

                if self.partition_mode == 'group':
                    gas = [0.0 if c[0] == 'NEW_ROUTE' else self._group_affinity(c[0], j, G) for c in cands]
                else:
                    gas = [-c[3] for c in cands]   # min-travel partition：closer = higher
                ras = [c[2] for c in cands]
                ga_r = rank_norm(gas); ra_r = rank_norm(ras)
                best_i = max(range(len(cands)), key=lambda i: (ga_r[i] + ra_r[i], -cands[i][3]))

                c = cands[best_i]
                if c[0] == 'NEW_ROUTE':
                    vid = idle_pool.pop(0)   # min idle id：只做 deterministic physical matching
                    slots.append(FleetSlot('new_route', vid, 0, float(clock), 0.0, members=[j]))
                else:
                    c[0].members.insert(c[1], j)

            # no silent drop（§41）：service-first fallback 到 JF1-H incumbent
            if unresolved:
                self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
                return

        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        by_id = {v.vehicle_id: v for v in vehicles}
        for s in slots:
            v = by_id[s.vehicle_id]
            suffix = list(s.members) + [0] if s.members else [0]
            if suffix == [0] and s.anchor_node != 0 and has_future:
                v.mutable_suffix = []   # WAIT：ready@customer 有未来 reveal，暂不 return
            else:
                v.mutable_suffix = suffix
        for vid in idle_pool:
            by_id[vid].mutable_suffix = [0]
