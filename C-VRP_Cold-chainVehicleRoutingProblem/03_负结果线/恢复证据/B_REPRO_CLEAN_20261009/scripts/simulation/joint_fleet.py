"""
JF0 / JF1 —— Joint Fleet Recourse 的 fleet-level 状态与 ownership 语义。

B0 已证明：17.6% gap 主要来自 fleet allocation（17.9%），sequencing 已饱和（−1.25%）。
本模块是 Joint Fleet 的基础（导师 §9-21）：

JF0（本文件）：
  - FleetAnchor：每辆 active 车的「规划锚点」（ready/idle/committed 不同语义）。
  - get_fleet_anchors / get_global_pool：fleet 状态 + 可释放客户池。
  - ownership 语义：executed prefix + committed leg = 冻结；mutable suffix = 可释放；
    future = 不可见。
  - get_vehicle_anchor_node：committed 车的 anchor 是「当前 committed 客户服务完成后的未来状态」。

JF1（joint assignment）：
  - JointAssignmentReplanner：在 fleet level 联合分配 visible 客户到车辆，每辆车内部仍用
    Resource Beam 排 suffix；三种 score：heuristic / shuffle / real。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict

from strict_online_env import Replanner, GreedyReplanner


@dataclass
class FleetAnchor:
    """一辆 active 车的规划锚点（参与 joint assignment 时的可规划状态，导师 §9）。"""
    vehicle_id: int
    status: str               # 'idle' / 'ready' / 'committed'
    current_node: int         # 当前节点（ready/idle）
    ready_time: float         # 可离开 current_node 的时刻
    current_load: float       # 当前载重
    committed_next: Optional[int] = None     # committed 车的下一个客户（冻结）
    committed_finish: Optional[float] = None  # committed 完成时刻
    locked_customers: Set[int] = field(default_factory=set)  # 冻结客户（committed_next）


def get_fleet_anchors(vehicles) -> List[FleetAnchor]:
    """返回所有 active 车辆的 planning anchor（closed 车排除）。导师 §9。"""
    anchors = []
    for v in vehicles:
        if v.status == 'closed':
            continue
        a = FleetAnchor(
            vehicle_id=v.vehicle_id,
            status=v.status,
            current_node=int(v.current_node),
            ready_time=float(v.ready_time),
            current_load=float(v.current_load),
        )
        if v.status == 'committed':
            a.committed_next = int(v.committed_next) if v.committed_next is not None else None
            a.committed_finish = float(v.committed_finish) if v.committed_finish is not None else None
            if a.committed_next not in (None, 0):
                a.locked_customers.add(a.committed_next)
        anchors.append(a)
    return anchors


def get_vehicle_anchor_node(anchor: FleetAnchor) -> int:
    """committed 车的规划锚点节点 = committed_next（服务完成后从该节点继续规划）。

    ready/idle 车 = current_node。导师 §9：committed 车 anchor 是「当前 committed customer
    服务完成后的未来可规划状态」。
    """
    if anchor.status == 'committed' and anchor.committed_next not in (None, 0):
        return anchor.committed_next
    return anchor.current_node


def get_global_pool(vehicles, served_mask, visible_ids, replan_ids=None) -> Tuple[List[int], Set[int]]:
    """可释放客户池 = visible ∩ unserved − committed −（非 replan 车的 mutable tail）。

    关键：只释放「本次要重规划的车」的 mutable tail；非 replan 车的 tail 冻结（保留旧 plan），
    否则同一客户会被两辆车服务（stale-tail duplicate，P0-CTRL-4 同款问题）。

    Returns:
        pool: 可重新分配的 visible 未服务客户（mutable ownership 已释放）
        locked: 冻结客户（committed_next + 非 replan 车的 tail）
    """
    locked = set()
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            locked.add(int(v.committed_next))
        # 非 replan 车的 mutable tail 冻结（保留旧 plan，防止 duplicate）
        if replan_ids is not None and v.vehicle_id not in replan_ids:
            for n in v.mutable_suffix:
                if n != 0:
                    locked.add(int(n))
    pool = [int(i) for i in visible_ids
            if not served_mask[int(i)] and int(i) not in locked]
    return pool, locked


def ownership_partition(vehicles, served_mask, visible_ids, reveal_time, clock):
    """把客户分成三类：frozen（已执行+committed）/ releasable（mutable）/ invisible（future）。

    Returns:
        frozen: set[int] 已执行 prefix + committed leg
        releasable: list[int] 可释放的 visible 未服务客户
        invisible: set[int] 未来未揭示客户
    """
    frozen = set(np.where(served_mask)[0])
    releasable = []
    invisible = set()
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            frozen.add(int(v.committed_next))
    for i in visible_ids:
        i = int(i)
        if i in frozen:
            continue
        releasable.append(i)
    for i in range(1, len(reveal_time)):
        if reveal_time[i] > clock + 1e-6 and not served_mask[i]:
            invisible.add(i)
    # visible_ids 本身已过滤 future（由 env 计算），invisible 是冗余保险
    return frozen, releasable, invisible


class JointAssignmentReplanner(Replanner):
    """JF1：fleet-level 联合分配 + 每辆车 greedy sequencing。

    释放 mutable ownership（只对 replan 车），在 fleet level 联合把 visible 未服务客户分配到
    「score 最高且还有容量」的车，而不是逐车顺序拍卖。

    score_mode（导师 §16）：
      - 'heuristic'：min travel(anchor_k, j)（纯启发式，无模型）
      - 'real'：MaskCO edge logits L[anchor_k, j]（learned preference 作 assignment score）
      - 'shuffle'：打乱 logits 的 customer 维度（保持 multiset，破坏 (anchor,j)→score 映射）

    real/shuffle 需 model + dataset（encode→decode 得 edge logits），heuristic 不需要。
    """

    def __init__(self, score_mode='heuristic', model=None, dataset=None, capacity=None,
                 tw_max=None, tw_speed=1.0, decode_seed=42, shuffle_seed=None):
        self.score_mode = score_mode
        self.decode_seed = decode_seed
        self.shuffle_seed = shuffle_seed
        if score_mode in ('real', 'shuffle'):
            assert model is not None and dataset is not None and capacity is not None and tw_max is not None
            import jax
            import jax.numpy as jnp
            self.model = model
            self.capacity = capacity
            self.tw_max = tw_max
            self.tw_speed = tw_speed
            self.coords = dataset['coords'].astype(np.float32)
            self.demands = dataset['demands'].astype(np.float32)
            self.tw_start = dataset['tw_start'].astype(np.float32)
            self.tw_end = dataset['tw_end'].astype(np.float32)
            self.service_time = dataset.get('service_time', np.zeros_like(self.demands, dtype=np.float32))
            self.temp_class = dataset.get('temp_class', np.zeros_like(self.demands, dtype=np.int32))
            self.reveal_time = dataset.get('reveal_time', np.zeros_like(self.demands, dtype=np.float32))
            self.energy_mat = dataset.get('energy_mat', None)
            if self.energy_mat is not None:
                self.energy_mat = self.energy_mat.astype(np.float32)
            from cvrptw_utils import coord_normalize_visible

            @jax.jit
            def _enc(raw_features, visible_mask=None, edge_feat=None):
                raw_features = raw_features.at[..., :2].set(
                    coord_normalize_visible(raw_features[..., :2], visible_mask))
                return self.model.encode(raw_features, visible_mask=visible_mask, edge_feat=edge_feat)
            self._enc = _enc

    def _compute_edge_logits(self, inst_idx, vis_mask):
        """encode → decode 得 edge logits L（N×N）。adjmat=None（无 masked adjacency）、timestep=0。"""
        import jax.numpy as jnp
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
        edge_feat = self.energy_mat[inst_idx:inst_idx + 1] if self.energy_mat is not None else None
        feats = np.array(self._enc(jnp.array(raw), visible_mask=jnp.array(vis_mask[None]),
                                   edge_feat=jnp.array(edge_feat) if edge_feat is not None else None))
        timestep = jnp.array([0.0], dtype=jnp.float32)
        logits = self.model.decode(feats, timestep, None)
        return np.array(logits[0])  # (N, N)

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        anchors = get_fleet_anchors(vehicles)
        pool, _locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids)
        active_ids = {a.vehicle_id for a in anchors
                      if a.status in ('idle', 'ready')
                      and (replan_ids is None or a.vehicle_id in replan_ids)}

        # real/shuffle：算 edge logits L（打乱 customer 维度 = shuffle）
        edge_logits = None
        if self.score_mode in ('real', 'shuffle'):
            vis_mask = np.zeros(self.coords.shape[1], dtype=bool)
            vis_mask[0] = True
            for vid in visible_ids:
                vis_mask[int(vid)] = True
            edge_logits = self._compute_edge_logits(inst_idx, vis_mask)
            if self.score_mode == 'shuffle':
                seed = self.shuffle_seed if self.shuffle_seed is not None else self.decode_seed
                rng = np.random.default_rng(seed)
                perm = rng.permutation(edge_logits.shape[1])
                edge_logits = edge_logits[:, perm]

        def _anchor_load(a):
            ld = a.current_load
            if a.status == 'committed' and a.committed_next not in (None, 0):
                ld += float(env.demands[inst_idx, a.committed_next])
            return ld

        # 联合分配：每个 pool 客户分给 score 最高且还有容量的 active 车
        assignment = {vid: [] for vid in active_ids}
        assigned_load = {vid: 0.0 for vid in active_ids}
        for j in pool:
            best_v, best_s = None, -float('inf')
            dj = float(env.demands[inst_idx, j])
            for a in anchors:
                if a.vehicle_id not in active_ids:
                    continue
                if _anchor_load(a) + assigned_load[a.vehicle_id] + dj > env.capacity + 1e-6:
                    continue
                anchor_node = get_vehicle_anchor_node(a)
                if self.score_mode == 'heuristic':
                    s = -float(env.dist_mat[inst_idx, anchor_node, j])  # 距离越小越好
                else:  # real / shuffle
                    s = float(edge_logits[anchor_node, j])  # logit 越大越好
                if s > best_s:
                    best_s, best_v = s, a.vehicle_id
            if best_v is not None:
                assignment[best_v].append(j)
                assigned_load[best_v] += dj

        # 每辆车 greedy 排自己的 assigned 客户（含 return-depot TW 检查）
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = assignment.get(v.vehicle_id, [])
            suffix = self._greedy_sequence(env, inst_idx, v, assigned)
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []  # WAIT
            else:
                v.mutable_suffix = suffix

    def _greedy_sequence(self, env, inst_idx, v, assigned):
        """对 assigned 客户做 greedy sequencing（含 return-depot TW 检查）。"""
        route = []
        current = int(v.current_node)
        cur_time = float(v.ready_time)
        load = float(v.current_load)
        unvisited = set(int(j) for j in assigned)
        while unvisited:
            best, best_d = None, float('inf')
            for j in unvisited:
                d = float(env.dist_mat[inst_idx, current, j])
                arr = cur_time + d / env.tw_speed
                arr = max(arr, env.tw_start[inst_idx, j])
                finish = arr + env.service_time[inst_idx, j]
                ret = finish + env.dist_mat[inst_idx, j, 0] / env.tw_speed
                if (arr <= env.tw_end[inst_idx, j] + 1e-6
                        and ret <= env.tw_end[inst_idx, 0] + 1e-6
                        and load + env.demands[inst_idx, j] <= env.capacity):
                    if d < best_d:
                        best_d, best = d, j
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            load += env.demands[inst_idx, best]
            cur_time = max(cur_time + env.dist_mat[inst_idx, current, best] / env.tw_speed,
                           env.tw_start[inst_idx, best]) + env.service_time[inst_idx, best]
            current = best
        route.append(0)
        return route

    def _greedy_sequence_report(self, env, inst_idx, v, assigned):
        """greedy sequencing，返回 (route, dropped)。dropped = 被 TW/容量/return 丢弃的客户。

        Step 4（no silent drop）：`_greedy_sequence` 静默丢弃，此版本显式报告被丢客户，
        供 `_repair_unresolved` 修复或 fallback。逻辑与 `_greedy_sequence` 完全一致。
        """
        route = []
        current = int(v.current_node)
        cur_time = float(v.ready_time)
        load = float(v.current_load)
        unvisited = set(int(j) for j in assigned)
        while unvisited:
            best, best_d = None, float('inf')
            for j in unvisited:
                d = float(env.dist_mat[inst_idx, current, j])
                arr = cur_time + d / env.tw_speed
                arr = max(arr, env.tw_start[inst_idx, j])
                finish = arr + env.service_time[inst_idx, j]
                ret = finish + env.dist_mat[inst_idx, j, 0] / env.tw_speed
                if (arr <= env.tw_end[inst_idx, j] + 1e-6
                        and ret <= env.tw_end[inst_idx, 0] + 1e-6
                        and load + env.demands[inst_idx, j] <= env.capacity):
                    if d < best_d:
                        best_d, best = d, j
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            load += env.demands[inst_idx, best]
            cur_time = max(cur_time + env.dist_mat[inst_idx, current, best] / env.tw_speed,
                           env.tw_start[inst_idx, best]) + env.service_time[inst_idx, best]
            current = best
        route.append(0)
        return route, list(unvisited)

    def _repair_unresolved(self, env, inst_idx, vehicles, anchors, anchor_info,
                           assignment, unresolved, active_ids, capacity):
        """把 sequencing 丢弃的客户尝试放到其它 sound-feasible 车辆（min-travel）。返回 still_unresolved。

        Step 4（no silent drop）：对每个 dropped 客户 j，尝试加入每辆 active 车的 assigned 集合并
        用 `_greedy_sequence_report` 重排；若 j 能被服务则选 min-travel 车重分配，否则留在
        still_unresolved（→ 调用方 service-first fallback 回退 incumbent）。
        """
        by_id = {v.vehicle_id: v for v in vehicles}
        still = []
        for j in unresolved:
            best_vid, best_travel = None, float('inf')
            for a in anchors:
                vid = a.vehicle_id
                if vid not in active_ids or vid not in by_id:
                    continue
                trial = list(assignment.get(vid, [])) + [int(j)]
                _route, dr = self._greedy_sequence_report(env, inst_idx, by_id[vid], trial)
                if int(j) in dr:
                    continue
                travel = float(env.dist_mat[inst_idx, anchor_info[vid][0], j]) / env.tw_speed
                if travel < best_travel:
                    best_travel, best_vid = travel, vid
            if best_vid is not None:
                assignment.setdefault(best_vid, []).append(int(j))
            else:
                still.append(j)
        return still


def compute_anchor_info(env, inst_idx, anchors):
    """每辆 active 车的 (anchor_node, anchor_time, anchor_load)。导师 §2。"""
    info = {}
    for a in anchors:
        node = get_vehicle_anchor_node(a)
        t = a.committed_finish if a.status == 'committed' else a.ready_time
        ld = float(a.current_load)
        if a.status == 'committed' and a.committed_next not in (None, 0):
            ld += float(env.demands[inst_idx, a.committed_next])
        info[a.vehicle_id] = (node, float(t), ld)
    return info


@dataclass(frozen=True)
class PairDecision:
    """pair (k,j) 的可行性判定 + reject reason + 数值（Step 3 false-negative audit 用）。

    reason: None=feasible；'capacity' / 'eta_tw' / 'return_depot'。
    """
    feasible: bool
    reason: Optional[str]
    anchor_node: int
    anchor_time: float
    anchor_load: float
    arrival: Optional[float] = None
    service_start: Optional[float] = None
    tw_slack: Optional[float] = None
    cap_slack: Optional[float] = None
    return_slack: Optional[float] = None


def pair_decision(env, inst_idx, anchor_info, vid, j, extra_load, capacity):
    """pair (k,j) 的 necessary feasibility（capacity + direct-arrival TW + return），返回结构化判定。

    三者都是 necessary condition（false-negative 应为 0）：
      - capacity：load + demand > Q ⟹ 任何 route 超载
      - eta_tw：直接 anchor→j 到达即晚于 tw_end ⟹ 任何 route（更晚）都迟到
      - return_depot：直接 j→depot 返回晚于 tw_end[0] ⟹ 任何 route（更晚返回）都超时
    但 return_depot 按主控文档默认只作 soft feature，需 audit 证明 zero false-negative 后才 hard mask。
    """
    node, t, ld0 = anchor_info[vid]
    dj = float(env.demands[inst_idx, j])
    tw_start_j = float(env.tw_start[inst_idx, j])
    tw_end_j = float(env.tw_end[inst_idx, j])
    tw_end_0 = float(env.tw_end[inst_idx, 0])
    cap_slack = capacity - (ld0 + extra_load + dj)
    if cap_slack < -1e-6:
        return PairDecision(False, 'capacity', node, t, ld0, cap_slack=cap_slack)
    travel = float(env.dist_mat[inst_idx, node, j]) / env.tw_speed
    arrival = t + travel
    start = max(arrival, tw_start_j)
    tw_slack = tw_end_j - start
    if tw_slack < -1e-6:
        return PairDecision(False, 'eta_tw', node, t, ld0, arrival=arrival,
                            service_start=start, tw_slack=tw_slack, cap_slack=cap_slack)
    ready = start + float(env.service_time[inst_idx, j])
    ret = ready + float(env.dist_mat[inst_idx, j, 0]) / env.tw_speed
    return_slack = tw_end_0 - ret
    if return_slack < -1e-6:
        return PairDecision(False, 'return_depot', node, t, ld0, arrival=arrival,
                            service_start=start, tw_slack=tw_slack, cap_slack=cap_slack,
                            return_slack=return_slack)
    return PairDecision(True, None, node, t, ld0, arrival=arrival, service_start=start,
                        tw_slack=tw_slack, cap_slack=cap_slack, return_slack=return_slack)


def pair_feasible(env, inst_idx, anchor_info, vid, j, extra_load, capacity):
    """pair (k,j) 的 necessary feasibility（capacity + ETA-TW + return）。返回 travel 或 None。

    导师 §16：hard feasibility mask 只做 necessary condition，不保证整条 route 可行
    （Level-2 由 per-vehicle greedy/Resource Beam 验证）。结构化判定见 pair_decision()。
    """
    d = pair_decision(env, inst_idx, anchor_info, vid, j, extra_load, capacity)
    if not d.feasible:
        return None
    return float(env.dist_mat[inst_idx, d.anchor_node, j]) / env.tw_speed


@dataclass
class CandidateSet:
    """jf2_sound_v1 冻结的候选集（与 learned score 完全解耦，主控文档 Step 5）。

    customer_order 沿用 JF1-H 的 pool 顺序（不 MRV，Part XVIII）；hard mask 只用 Step 3
    审计证明 sound 的 necessary condition（capacity + direct-arrival TW）；return_slack
    作 soft feature 不 mask（§3.3 E）。alpha=0 时（score=min-travel）已实证与 JF1-H exact 一致。
    """
    inst_idx: int
    clock: float
    customer_order: List[int]                   # 沿用 JF1-H pool 顺序
    active_vehicle_ids: List[int]               # idle/ready 且参与 replan（升序）
    anchor_info: Dict[int, Tuple[int, float, float]]   # vid -> (node, time, load)
    feasible_vehicles: Dict[int, List[int]]     # customer -> sound-feasible vehicles（升序）
    return_slack: Dict[Tuple[int, int], float]  # (vid, j) -> return_slack（soft feature）


def build_sound_candidate_set(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                              replan_ids=None):
    """jf2_sound_v1：从 FleetState 构造候选集，hard mask 只用 sound necessary condition。

    hard mask（Step 3 审计 false-negative=0）：
      - capacity：anchor_load + d_j > Q
      - direct-arrival TW：anchor→j 直接到达即晚于 tw_end[j]
    soft feature（不 mask）：return_slack（主控文档 §3.3 E 默认先作 feature）。
    """
    anchors = get_fleet_anchors(vehicles)
    pool, _locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids)
    active_ids = sorted(a.vehicle_id for a in anchors
                        if a.status in ('idle', 'ready')
                        and (replan_ids is None or a.vehicle_id in replan_ids))
    active_anchors = [a for a in anchors if a.vehicle_id in active_ids]
    anchor_info = compute_anchor_info(env, inst_idx, active_anchors)
    customer_order = [int(j) for j in pool]
    feasible_vehicles = {}
    return_slack = {}
    for j in customer_order:
        vids = []
        for vid in active_ids:
            d = pair_decision(env, inst_idx, anchor_info, vid, j, 0.0, env.capacity)
            if d.reason in ('capacity', 'eta_tw'):
                continue  # sound hard mask
            vids.append(vid)
            if d.return_slack is not None:
                return_slack[(vid, j)] = d.return_slack
        feasible_vehicles[j] = vids
    return CandidateSet(inst_idx=int(inst_idx), clock=float(clock),
                        customer_order=customer_order, active_vehicle_ids=active_ids,
                        anchor_info=anchor_info, feasible_vehicles=feasible_vehicles,
                        return_slack=return_slack)


class JointAssignmentBeamReplanner(JointAssignmentReplanner):
    """JF1.5：Joint Assignment Beam（MRV ordering + top-L vehicles + B beam，min-travel score）。

    candidate_mode：
      - 'legacy_jf1h'（默认）：忠实退化到冻结 JF1-H —— 走 JointAssignmentReplanner('heuristic')
        的旧路径（pool 顺序 + 容量-only + min-travel + greedy sequence），**不调用 pair_feasible、
        不 MRV、不加 fallback**。用于 JF2 Gate-0 的 faithful regression（B=1 必须复现 24.5019/100%）。
      - 'safe_pair'：JF1.5 的 assignment beam（pair_feasible hard filter + MRV）。已判空，仅作
        degenerate assignment-surrogate diagnostic 归档。

    与 JF1-H（greedy，B=1）的唯一区别是 search depth：每个 customer 保留 top-L 个可行车辆的
    分配选择，beam 维护 B 个 partial assignment，最终取累计 travel 最小的 assignment。
    """

    def __init__(self, beam_width=1, top_l=4, candidate_mode='legacy_jf1h'):
        if candidate_mode not in ('legacy_jf1h', 'safe_pair'):
            raise ValueError("candidate_mode 只支持 legacy_jf1h / safe_pair")
        # 继承 JF1-H 的字段（score_mode='heuristic' 不需要 model/dataset）。
        super().__init__(score_mode='heuristic')
        self.candidate_mode = candidate_mode
        self.beam_width = beam_width
        self.top_l = top_l

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        if self.candidate_mode == 'legacy_jf1h':
            # 忠实 JF1-H：直接走冻结的 heuristic 路径（pool 顺序 + 容量-only + min-travel），
            # 不经过 pair_feasible / MRV / beam。这样 B=1 与 JointAssignmentReplanner('heuristic')
            # 是同一代码路径，逐实例 exact 一致。
            super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
            return
        anchors = get_fleet_anchors(vehicles)
        pool, _locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids)
        active_ids = {a.vehicle_id for a in anchors
                      if a.status in ('idle', 'ready')
                      and (replan_ids is None or a.vehicle_id in replan_ids)}
        active_anchors = [a for a in anchors if a.vehicle_id in active_ids]
        if not active_anchors or not pool:
            return

        anchor_info = compute_anchor_info(env, inst_idx, active_anchors)

        # MRV ordering：feasible_vehicle_count 升序（初始状态），tw_end 升序 tiebreak
        def _feasible_count(j):
            return sum(1 for vid in active_ids
                       if pair_feasible(env, inst_idx, anchor_info, vid, j, 0.0,
                                        env.capacity) is not None)

        ordered = sorted(pool, key=lambda j: (_feasible_count(j),
                                              float(env.tw_end[inst_idx, j])))

        # beam search over assignment
        beams = [{'assigned': {vid: [] for vid in active_ids},
                  'load': {vid: 0.0 for vid in active_ids},
                  'cost': 0.0}]
        for j in ordered:
            new_beams = []
            for b in beams:
                cands = []
                for vid in active_ids:
                    travel = pair_feasible(env, inst_idx, anchor_info, vid, j,
                                           b['load'][vid], env.capacity)
                    if travel is not None:
                        cands.append((travel, vid))
                cands.sort(key=lambda x: x[0])  # min travel 优先
                if not cands:
                    # 无可行车辆：该 beam 保留但 cost 惩罚（漏服务）
                    nb = {'assigned': {k: list(v) for k, v in b['assigned'].items()},
                          'load': dict(b['load']), 'cost': b['cost'] + 1e9}
                    new_beams.append(nb)
                    continue
                for travel, vid in cands[:self.top_l]:
                    nb = {'assigned': {k: list(v) for k, v in b['assigned'].items()},
                          'load': dict(b['load']), 'cost': b['cost'] + travel}
                    nb['assigned'][vid].append(j)
                    nb['load'][vid] += float(env.demands[inst_idx, j])
                    new_beams.append(nb)
            beams = sorted(new_beams, key=lambda b: b['cost'])[:self.beam_width]

        best = beams[0]  # 已按 cost 升序
        assignment = best['assigned']

        # 每辆车 greedy sequence（复用 JointAssignmentReplanner._greedy_sequence）。
        # 注意：safe_pair 已判空归档；no-silent-drop 的 report/repair 原语（_greedy_sequence_report /
        # _repair_unresolved）供 JF2 skeleton 的 greedy assigner 使用，不在此 retro-fit（否则因
        # safe_pair 早期决策发散，fallback 无法恢复被丢客户且会引入新 drop）。
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = assignment.get(v.vehicle_id, [])
            suffix = self._greedy_sequence(env, inst_idx, v, assigned)
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix
