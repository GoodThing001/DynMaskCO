"""
Strict Online Decoder — 严格 non-anticipatory 事件驱动滚动时域解码（R1.5 重构）。

2026-08-28 重构：车队事件语义（no premature close + service-completion event）统一收进
`simulation/strict_online_env.py`（P0-SIM）。本文件只负责：

  - MaskCOReplanner：模型的 Replanner 实现（encode → incumbent → masked adjacency →
    decode → resource beam → suffix）。Fix B：incumbent_builder='edd'/'nn'。
  - StrictOnlineDecoder：StrictOnlineEnv 的薄封装（decode_instance → env.run → trace 评估）。
  - build_maskco_replanner：保留旧签名，返回 MaskCOReplanner（向后兼容 run_r1_eval）。

与旧实现的区别：
  - 不再有「cand_alloc > inc_alloc 否则回滚 EDD」的 gate（这是模型==贪心的根因 R1）；
    模型 beam 的 plan 被直接采用，模型 preference 直接生效。
  - 车队不再 route 空了就 closed（P0-FLEET）；事件包含 service-completion（P0-EVENT）。

Author: P0-Protocol Repair (R1.5)
Date: 2026-08-28
"""

import sys, os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# P0-M-DET：强制 FP32 matmul（禁用 TF32）。RTX 5090 默认对 float32 matmul 走 TF32
# （10-bit mantissa），使 decode 的 logits 跨 run 有 ~1e-3 级噪声，足以翻转 beam 的
# argmax → 决策分叉（R1.5-M smoke_a≠smoke_b 的根因，非 seeding bug）。FP32 消除主噪声源。
jax.config.update('jax_default_matmul_precision', 'float32')

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from cvrptw_utils import coord_normalize_visible
from authoritative_evaluator import evaluate_execution_trace
from strict_online_env import (StrictOnlineEnv, Replanner, VehicleState,
                               VehicleTrace, ServiceRecord, GreedyReplanner)
from joint_fleet import JointAssignmentReplanner, JointAssignmentBeamReplanner  # JF1-H / JF1.5


def derive_seed(base, *values):
    """稳定的整数 seed 混合（与 resource_beam.derive_seed 一致，P0-M3）。"""
    x = int(base) & 0xFFFFFFFF
    for v in values:
        v = int(v) & 0xFFFFFFFF
        x = (1664525 * (x ^ v) + 1013904223) & 0xFFFFFFFF
    return int(x)


# ============================================================================
# 贪心 incumbent（EDD / NN）—— 模型 reconstruction context
# ============================================================================

def _eddy_incumbent(inst_idx, visible_ids, current_node, current_time, current_load,
                    served_mask, dist_mat, tw_start, tw_end, service_time, speed,
                    demands, capacity):
    """EDD 贪心构造 visible feasible incumbent（从当前状态出发，TW+容量约束）。"""
    route = [int(current_node)]
    current, cur_time = int(current_node), float(current_time)
    load = float(current_load)
    unvisited = set(int(i) for i in visible_ids) - set(np.where(served_mask)[0])
    while unvisited:
        best, best_score = None, float('inf')
        for j in unvisited:
            d = dist_mat[inst_idx, current, j]
            arr = cur_time + d / speed
            arr = max(arr, tw_start[inst_idx, j])
            if arr <= tw_end[inst_idx, j] and load + demands[inst_idx, j] <= capacity:
                score = tw_end[inst_idx, j] + 0.01 * d
                if score < best_score:
                    best_score, best = score, j
        if best is None:
            break
        route.append(best)
        unvisited.remove(best)
        cur_time = max(cur_time + dist_mat[inst_idx, current, best] / speed,
                       tw_start[inst_idx, best]) + service_time[inst_idx, best]
        load += demands[inst_idx, best]
        current = best
    if route[-1] != 0:
        route.append(0)
    return np.array(route, dtype=np.int32)


def _nn_incumbent(inst_idx, visible_ids, current_node, current_time, current_load,
                  served_mask, dist_mat, tw_start, tw_end, service_time, speed,
                  demands, capacity):
    """Nearest-neighbor 贪心 incumbent（Fix B 诊断用，dist 为主）。"""
    route = [int(current_node)]
    current, cur_time = int(current_node), float(current_time)
    load = float(current_load)
    unvisited = set(int(i) for i in visible_ids) - set(np.where(served_mask)[0])
    while unvisited:
        best, best_dist = None, float('inf')
        for j in unvisited:
            d = dist_mat[inst_idx, current, j]
            arr = cur_time + d / speed
            arr = max(arr, tw_start[inst_idx, j])
            if arr <= tw_end[inst_idx, j] and load + demands[inst_idx, j] <= capacity:
                if d < best_dist:
                    best_dist, best = d, j
        if best is None:
            break
        route.append(best)
        unvisited.remove(best)
        cur_time = max(cur_time + dist_mat[inst_idx, current, best] / speed,
                       tw_start[inst_idx, best]) + service_time[inst_idx, best]
        load += demands[inst_idx, best]
        current = best
    if route[-1] != 0:
        route.append(0)
    return np.array(route, dtype=np.int32)


# ============================================================================
# MaskCOReplanner —— 模型的 Replanner（Fix B：incumbent_builder）
# ============================================================================

def _first_action(route, current_node):
    """route 的第一个非 current_node 节点（0=return depot，-1=None）。"""
    if route is None:
        return -1
    for n in route:
        n = int(n)
        if n == int(current_node):
            continue
        return n
    return 0


def _num_planned_customers(route, current_node):
    """route 里非 0、非 current_node 的客户数（判断模型是换序还是改服务数量）。"""
    if route is None:
        return 0
    return sum(1 for n in route if int(n) not in (0, int(current_node)))


def _apply_selection(selection_mode, candidate_exists, candidate_n_planned,
                     inc_n_planned, candidate_cost, inc_cost):
    """R1.6 selection：raw（无条件接受 candidate）vs guarded（service-first lexicographic guard）。

    guarded 规则 Φ(x)=(-N_served, N_late, N_viol, C_dist)，Resource Beam 已硬约束可行，退化为
    服务数优先 + 同服务数比 cost。返回 (candidate_accepted, selection_reason)。
    """
    if selection_mode == 'guarded':
        if not candidate_exists:
            return False, 'candidate_missing'
        if candidate_n_planned > inc_n_planned:
            return True, 'more_service'
        if candidate_n_planned < inc_n_planned:
            return False, 'less_service'
        if candidate_cost < inc_cost - 1e-9:
            return True, 'lower_cost_same_service'
        return False, 'not_better'
    # raw
    if candidate_exists:
        return True, 'raw_accept'
    return False, 'candidate_missing'


class MaskCOReplanner(Replanner):
    """模型 replanner：顺序拍卖，每辆车用 MaskCO encode→decode→beam 建 suffix。

    直接采用 beam 的 plan（无 incumbent/candidate gate，R1 根因已消除）。
    """

    def __init__(self, dataset, capacity, model, tw_max, tw_speed=1.0,
                 beam_width=16, incumbent_builder='edd', decode_seed=42,
                 logit_mode='real', enable_local_counterfactual=False,
                 selection_mode='raw', shuffle_seed=None,
                 save_proposal_trace=False):
        if selection_mode not in ('raw', 'guarded'):
            raise ValueError('selection_mode 只支持 raw / guarded')
        self.capacity = capacity
        self.model = model
        self.tw_max = tw_max
        self.tw_speed = tw_speed
        self.beam_width = beam_width
        self.incumbent_builder = incumbent_builder  # 'edd' | 'nn'
        self.decode_seed = decode_seed
        self.logit_mode = logit_mode  # 'real' | 'shuffle'
        self.enable_local_counterfactual = enable_local_counterfactual  # H2=True
        self.selection_mode = selection_mode  # 'raw' | 'guarded'（R1.6）
        # R1.7：shuffle_seed 独立于 decode_seed（adjacency mask 用 decode_seed 派生、固定）。
        # None = 从 decode_seed 派生（向后兼容 H3）；显式传入则只改 logits shuffle、不动 mask。
        self.shuffle_seed = shuffle_seed
        self.audit = []  # per-(event,vehicle) 审计记录
        # R1.7-2：proposal trace（完整 route + current_node），供 Proposal Structure Audit 用。
        self.save_proposal_trace = save_proposal_trace
        self.proposal_trace = []

        self.coords = dataset['coords'].astype(np.float32)
        self.demands = dataset['demands'].astype(np.float32)
        self.tw_start = dataset['tw_start'].astype(np.float32)
        self.tw_end = dataset['tw_end'].astype(np.float32)
        self.service_time = dataset.get('service_time',
            np.zeros_like(self.demands, dtype=np.float32))
        self.temp_class = dataset.get('temp_class',
            np.zeros_like(self.demands, dtype=np.int32))
        self.reveal_time = dataset.get('reveal_time',
            np.zeros_like(self.demands, dtype=np.float32))
        self.energy_mat = dataset.get('energy_mat', None)
        if self.energy_mat is not None:
            self.energy_mat = self.energy_mat.astype(np.float32)
        self.num_nodes = self.coords.shape[1]

        # P0-M3：dist_mat 一致性 —— 优先用 dataset 的 dist_mat（asymmetric/real network），
        # 与 StrictOnlineEnv / Resource Beam / evaluator 同源。
        if 'dist_mat' in dataset:
            self.dist_mat = dataset['dist_mat'].astype(np.float32)
        else:
            diff = self.coords[:, :, None, :] - self.coords[:, None, :, :]
            self.dist_mat = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        @jax.jit
        def encode_fn(raw_features, visible_mask=None, edge_feat=None):
            raw_features = raw_features.at[..., :2].set(
                coord_normalize_visible(raw_features[..., :2], visible_mask))
            return model.encode(raw_features, visible_mask=visible_mask, edge_feat=edge_feat)
        self.encode_fn = encode_fn

        import importlib.util
        _rb_path = os.path.join(os.path.dirname(__file__), 'resource_beam.py')
        _rb_spec = importlib.util.spec_from_file_location('resource_beam', _rb_path)
        _rb = importlib.util.module_from_spec(_rb_spec)
        _rb_spec.loader.exec_module(_rb)
        self._rb = _rb

    def _incumbent(self, inst_idx, visible_ids, current_node, current_time,
                   current_load, serve_plus):
        fn = _nn_incumbent if self.incumbent_builder == 'nn' else _eddy_incumbent
        return fn(inst_idx, visible_ids, current_node, current_time, current_load,
                  serve_plus, self.dist_mat, self.tw_start, self.tw_end,
                  self.service_time, self.tw_speed, self.demands, self.capacity)

    def reset_audit(self):
        self.audit = []
        self.proposal_trace = []

    def _route_cost(self, inst_idx, route):
        """route（np 或 list，含 current_node 起 + 末尾 0）的总距离。"""
        total = 0.0
        for k in range(len(route) - 1):
            a, b = int(route[k]), int(route[k + 1])
            total += float(self.dist_mat[inst_idx, a, b])
        return total

    def _build_suffix(self, inst_idx, current_node, current_time, current_load,
                      serve_plus, visible_ids, clock, vehicle_id, event_id, replan_reason):
        """从单辆车当前状态建 beam suffix。返回 (suffix, audit)。P0-M3 重构。"""
        num_nodes = self.num_nodes
        # 1. 可见掩码（visible + served + allocated）
        vis_mask = np.zeros(num_nodes, dtype=bool)
        vis_mask[0] = True
        for vid in visible_ids:
            if 0 <= vid < num_nodes:
                vis_mask[vid] = True
        for s in np.where(serve_plus)[0]:
            if 0 <= s < num_nodes:
                vis_mask[s] = True

        # 2. 7D features（单实例）
        feat_arrays = [
            self.coords[inst_idx:inst_idx + 1],
            (self.demands[inst_idx:inst_idx + 1] / self.capacity)[..., None],
            (self.tw_start[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.tw_end[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
            (self.temp_class[inst_idx:inst_idx + 1] / 2.0)[..., None],
            (self.reveal_time[inst_idx:inst_idx + 1] / self.tw_max)[..., None],
        ]
        raw_features = np.concatenate(feat_arrays, axis=-1).astype(np.float32)
        vis = vis_mask[None, ..., None]
        raw_features[..., 2:] = raw_features[..., 2:] * vis
        raw_features[..., :2] = raw_features[..., :2] * vis + (1.0 - vis) * 0.5

        edge_feat_b = self.energy_mat[inst_idx:inst_idx + 1] if self.energy_mat is not None else None
        features = np.array(self.encode_fn(
            jnp.array(raw_features), visible_mask=jnp.array(vis_mask[None]),
            edge_feat=jnp.array(edge_feat_b) if edge_feat_b is not None else None,
        ))

        # 3. incumbent（EDD/NN）→ masked adjacency → decode
        incumbent = self._incumbent(inst_idx, visible_ids, current_node,
                                    current_time, current_load, serve_plus)
        inc_first = _first_action(incumbent, current_node)
        inc_cost = self._route_cost(inst_idx, incumbent)
        inc_n_planned = _num_planned_customers(incumbent, current_node)

        B, N, D = features.shape
        adjmat = np.zeros((B, N, N), dtype=np.float32)
        for k in range(len(incumbent) - 1):
            a, b = int(incumbent[k]), int(incumbent[k + 1])
            if 0 <= a < N and 0 <= b < N:
                adjmat[0, a, b] = 1.0
        keep_rate = 0.7
        # P0-M3：decision_seed = derive_seed(decode_seed, inst, event, vehicle)
        decision_seed = derive_seed(self.decode_seed, inst_idx, event_id, vehicle_id)
        mask_seed = derive_seed(decision_seed, 1001)
        rng = np.random.default_rng(mask_seed)
        mask = (rng.random((1, N, N)) < keep_rate).astype(np.float32)
        adjmat = adjmat * mask
        timestep = jnp.array([keep_rate], dtype=jnp.float32)
        logits = self.model.decode(features, timestep, jnp.array(adjmat))
        edge_logits = np.array(logits[0])
        # R1.7：mask（adjacency corruption）由 decode_seed 固定；shuffle_seed 可独立变化。
        # 这样 H3-G 的 5 个 shuffle seed 只改 logits shuffle、不动 adjacency mask。
        if self.shuffle_seed is not None:
            shuffle_seed = derive_seed(self.shuffle_seed, decision_seed, 2001)
        else:
            shuffle_seed = derive_seed(decision_seed, 2001)

        # 4. Resource Beam（logits 不再被改；shuffle 只发生在 beam ranking 层）
        searcher = self._rb.ResourceBeamSearcher(
            self.coords[inst_idx], self.tw_start[inst_idx], self.tw_end[inst_idx],
            self.service_time[inst_idx], self.demands[inst_idx], self.capacity,
            speed=self.tw_speed, K=self.beam_width, visible_mask=vis_mask,
            depot_dispatch_time=float(clock), tw_margin=0.0,
            dist_mat=self.dist_mat[inst_idx],
        )

        if self.logit_mode == 'real':
            selected_candidate, selected_success, selected_meta = searcher.generate_from_state(
                edge_logits, current_node, current_time, current_load, serve_plus,
                single_route=True, logit_mode='real', shuffle_seed=shuffle_seed,
                return_meta=True)
            # H2 counterfactual：同 state 同时算 shuffled proposal（只 audit，不执行）
            if self.enable_local_counterfactual:
                shuffle_cf, shuffle_cf_success, shuffle_cf_meta = searcher.generate_from_state(
                    edge_logits, current_node, current_time, current_load, serve_plus,
                    single_route=True, logit_mode='shuffle', shuffle_seed=shuffle_seed,
                    return_meta=True)
            else:
                shuffle_cf = None
        else:  # shuffle（H3）
            selected_candidate, selected_success, selected_meta = searcher.generate_from_state(
                edge_logits, current_node, current_time, current_load, serve_plus,
                single_route=True, logit_mode='shuffle', shuffle_seed=shuffle_seed,
                return_meta=True)
            shuffle_cf = None

        # 5. incumbent / candidate / executed 三对象拆分 + selection（raw vs guarded，R1.6）
        candidate_exists = selected_candidate is not None
        candidate_n_planned = (_num_planned_customers(selected_candidate, current_node)
                               if candidate_exists else 0)
        candidate_cost = self._route_cost(inst_idx, selected_candidate) if candidate_exists else float('nan')

        # P0-M6（R1.6）：raw = 无条件接受 candidate（测 raw utility）；
        # guarded = service-first lexicographic incumbent-preserving guard（formal 方法）。
        candidate_accepted, selection_reason = _apply_selection(
            self.selection_mode, candidate_exists, candidate_n_planned,
            inc_n_planned, candidate_cost, inc_cost)

        executed_route = selected_candidate if candidate_accepted else incumbent
        fallback_to_incumbent = not candidate_accepted

        candidate_first = _first_action(selected_candidate, current_node) if candidate_exists else -1
        executed_first = _first_action(executed_route, current_node)
        executed_cost = self._route_cost(inst_idx, executed_route)

        audit = {
            'event_id': int(event_id),
            'clock': float(clock),
            'vehicle_id': int(vehicle_id),
            'replan_reason': replan_reason,
            'incumbent_builder': self.incumbent_builder,
            'logit_mode': self.logit_mode,
            'selection_mode': self.selection_mode,
            'decode_seed': int(self.decode_seed),
            'decision_seed': int(decision_seed),
            'initial_actionable_count': int(selected_meta['initial_actionable_count']) if selected_meta else 0,
            'candidate_exists': bool(candidate_exists),
            'candidate_accepted': bool(candidate_accepted),
            'selection_reason': selection_reason,
            'beam_complete': bool(selected_success),
            'fallback_to_incumbent': bool(fallback_to_incumbent),
            'inc_first': int(inc_first),
            'candidate_first': int(candidate_first),
            'executed_first': int(executed_first),
            'candidate_vs_incumbent_changed': bool(candidate_exists and candidate_first != inc_first),
            'inc_cost': float(inc_cost),
            'candidate_cost': float(candidate_cost),
            'executed_cost': float(executed_cost),
            'inc_n_planned': int(inc_n_planned),
            'candidate_n_planned': int(candidate_n_planned),
        }

        # H2 的 same-state real-vs-shuffle divergence（H3 无 counterfactual）
        if self.enable_local_counterfactual and shuffle_cf is not None:
            local_real_exists = selected_candidate is not None
            local_shuffle_exists = shuffle_cf is not None
            real_first = _first_action(selected_candidate, current_node) if local_real_exists else -1
            shuffle_first = _first_action(shuffle_cf, current_node) if local_shuffle_exists else -1
            audit.update({
                'local_real_exists': bool(local_real_exists),
                'local_shuffle_exists': bool(local_shuffle_exists),
                'local_real_first': int(real_first),
                'local_shuffle_first': int(shuffle_first),
                'local_first_diverged': bool(local_real_exists and local_shuffle_exists
                                             and real_first != shuffle_first),
            })
        else:
            audit.update({
                'local_real_exists': False, 'local_shuffle_exists': False,
                'local_real_first': -1, 'local_shuffle_first': -1,
                'local_first_diverged': False,
            })

        cand_suffix = [int(n) for n in executed_route if int(n) != int(current_node)]

        # R1.7-2：proposal trace（完整 route，供结构审计）。route 均以 current_node 开头、0 结尾。
        trace = None
        if self.save_proposal_trace:
            trace = {
                'event_id': int(event_id),
                'clock': float(clock),
                'vehicle_id': int(vehicle_id),
                'replan_reason': replan_reason,
                'current_node': int(current_node),
                'incumbent_route': [int(n) for n in incumbent],
                'candidate_route': ([int(n) for n in selected_candidate]
                                    if candidate_exists else None),
                'executed_route': [int(n) for n in executed_route],
                'inc_cost': float(inc_cost),
                'candidate_cost': float(candidate_cost),
                'executed_cost': float(executed_cost),
                'candidate_exists': bool(candidate_exists),
                'candidate_accepted': bool(candidate_accepted),
                'selection_reason': selection_reason,
                'initial_actionable_count': (
                    int(selected_meta['initial_actionable_count']) if selected_meta else 0),
            }
        return cand_suffix, audit, trace

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved = env.get_reserved_customers(vehicles)  # P0-B
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)  # P0-C
        event_id = getattr(env, 'event_id', -1)
        visible_pending = [int(i) for i in visible_ids
                           if not served_mask[i] and int(i) not in reserved]
        allocated = set()
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue  # P0-CTRL：非 needs_replan 车保留旧 plan
            serve_plus = served_mask.copy()
            for r in reserved:
                serve_plus[r] = True
            for a in allocated:
                serve_plus[a] = True
            suffix, audit, trace = self._build_suffix(
                inst_idx, v.current_node, v.ready_time, v.current_load,
                serve_plus, visible_ids, clock,
                vehicle_id=v.vehicle_id, event_id=event_id,
                replan_reason=v.replan_reason,
            )
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []  # WAIT
            else:
                v.mutable_suffix = suffix
            for o in suffix:
                if o != 0:
                    allocated.add(int(o))
            audit.update({
                'inst_idx': int(inst_idx),
                'visible_pending_count': len(visible_pending),
                'reserved_count': len(reserved),
            })
            self.audit.append(audit)
            if trace is not None:
                trace.update({
                    'instance_id': int(inst_idx),
                    'visible_pending_count': len(visible_pending),
                    'reserved_count': len(reserved),
                })
                self.proposal_trace.append(trace)


def build_maskco_replanner(dataset, capacity, model, tw_max, tw_speed=1.0,
                           beam_width=16, decode_seed=42, incumbent_builder='edd',
                           logit_mode='real', enable_local_counterfactual=False,
                           selection_mode='raw', shuffle_seed=None,
                           save_proposal_trace=False):
    """保留旧签名，返回 MaskCOReplanner（向后兼容 run_r1_eval.py）。"""
    return MaskCOReplanner(dataset, capacity, model, tw_max, tw_speed,
                           beam_width, incumbent_builder, decode_seed, logit_mode,
                           enable_local_counterfactual=enable_local_counterfactual,
                           selection_mode=selection_mode, shuffle_seed=shuffle_seed,
                           save_proposal_trace=save_proposal_trace)


def build_replanner(method, dataset, capacity, model, tw_max, tw_speed, beam_width,
                    decode_seed=42, shuffle_seed=None, save_proposal_trace=False,
                    candidate_mode='legacy_jf1h'):
    """H0=edd / H1=nn / H2=model(NN+real,raw) / H3=model_shuffle(NN+shuffle,raw) /
    H2-G=model_guarded(NN+real,guarded) / H3-G=model_guarded_shuffle(NN+shuffle,guarded)。

    H2/H2-G 开 enable_local_counterfactual（同 state 算 shuffled proposal 作 mechanism audit）；
    H3/H3-G 本身跑 shuffled trajectory，不需 counterfactual。
    shuffle_seed（R1.7）：独立于 decode_seed，只改 logits shuffle、不动 adjacency mask。
    save_proposal_trace（R1.7-2）：保存完整 route trace 供 Proposal Structure Audit。
    candidate_mode（JF2）：joint_beam 的 candidate 语义——'legacy_jf1h'（忠实 JF1-H 退化）
    或 'safe_pair'（JF1.5 assignment beam）。
    """
    if method == 'edd':
        return GreedyReplanner('edd')
    if method == 'nn':
        return GreedyReplanner('nn')
    if method == 'joint_heuristic':
        return JointAssignmentReplanner('heuristic')  # JF1-H（fleet-level 联合分配）
    if method == 'joint_beam':
        return JointAssignmentBeamReplanner(beam_width=beam_width, top_l=4,
                                            candidate_mode=candidate_mode)  # JF1.5 / JF2 Gate-0
    if method == 'joint_real':
        return JointAssignmentReplanner('real', model=model, dataset=dataset, capacity=capacity,
                                        tw_max=tw_max, tw_speed=tw_speed, decode_seed=decode_seed)
    if method == 'joint_shuffle':
        return JointAssignmentReplanner('shuffle', model=model, dataset=dataset, capacity=capacity,
                                        tw_max=tw_max, tw_speed=tw_speed, decode_seed=decode_seed,
                                        shuffle_seed=shuffle_seed)
    if method == 'model':
        return build_maskco_replanner(
            dataset, capacity, model, tw_max, tw_speed, beam_width,
            decode_seed=decode_seed, incumbent_builder='nn', logit_mode='real',
            enable_local_counterfactual=True, selection_mode='raw',
            save_proposal_trace=save_proposal_trace)
    if method == 'model_guarded':
        return build_maskco_replanner(
            dataset, capacity, model, tw_max, tw_speed, beam_width,
            decode_seed=decode_seed, incumbent_builder='nn', logit_mode='real',
            enable_local_counterfactual=True, selection_mode='guarded',
            save_proposal_trace=save_proposal_trace)
    if method == 'model_guarded_shuffle':  # H3-G
        return build_maskco_replanner(
            dataset, capacity, model, tw_max, tw_speed, beam_width,
            decode_seed=decode_seed, incumbent_builder='nn', logit_mode='shuffle',
            enable_local_counterfactual=False, selection_mode='guarded',
            shuffle_seed=shuffle_seed, save_proposal_trace=save_proposal_trace)
    if method == 'model_shuffle':
        return build_maskco_replanner(
            dataset, capacity, model, tw_max, tw_speed, beam_width,
            decode_seed=decode_seed, incumbent_builder='nn', logit_mode='shuffle',
            enable_local_counterfactual=False, save_proposal_trace=save_proposal_trace)
    raise ValueError(f"unknown method: {method}")


# ============================================================================
# StrictOnlineDecoder —— StrictOnlineEnv 薄封装
# ============================================================================

class StrictOnlineDecoder:
    """strict-online 事件驱动解码：委托 StrictOnlineEnv，用 execution trace 评估。"""

    def __init__(self, dataset, capacity, model=None, tw_max=None, tw_speed=1.0,
                 beam_width=16, replan_fn=None, num_vehicles=25, incumbent_builder='edd'):
        self.dataset = dataset
        self.capacity = capacity
        self.tw_max = tw_max if tw_max is not None else float(dataset['tw_end'].max())
        self.tw_speed = tw_speed
        self.beam_width = beam_width
        self.num_vehicles = num_vehicles

        if replan_fn is None:
            if model is not None:
                replan_fn = build_maskco_replanner(
                    dataset, capacity, model, self.tw_max, tw_speed,
                    beam_width, incumbent_builder=incumbent_builder)
            else:
                replan_fn = GreedyReplanner(incumbent_builder=incumbent_builder)

        self.replanner = replan_fn
        self.env = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles, replan_fn)
        self.num_instances = self.env.num_instances

    def decode_instance(self, inst_idx, horizon=None):
        traces, served_mask = self.env.run(inst_idx)
        metrics = evaluate_execution_trace(
            traces,
            self.env.coords[inst_idx], self.env.tw_start[inst_idx],
            self.env.tw_end[inst_idx], self.env.service_time[inst_idx],
            self.env.demands[inst_idx], self.capacity, speed=self.tw_speed,
            dist_mat=self.env.dist_mat[inst_idx],
        )
        # 兼容旧返回：(final_route, metrics)
        final_route = [0]
        for tr in traces:
            if tr.services:
                for sr in tr.services:
                    final_route.append(int(sr.node))
                final_route.append(0)
        return np.array(final_route, dtype=np.int32), metrics


# ============================================================================
# 单元测试（用贪心 replanner 验证事件驱动逻辑）
# ============================================================================

if __name__ == '__main__':
    print("Testing StrictOnlineDecoder (event-driven rolling horizon, R1.5)...")

    N = 5
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
    tw_end = np.array([[30., 30., 30., 30., 30.]], dtype=np.float32)
    service_time = np.zeros((1, 5), dtype=np.float32)
    temp_class = np.zeros((1, 5), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 5., 10.]], dtype=np.float32)

    dataset = {
        'coords': coords, 'demands': demands,
        'tw_start': tw_start, 'tw_end': tw_end,
        'service_time': service_time, 'temp_class': temp_class,
        'reveal_time': reveal_time,
    }

    decoder = StrictOnlineDecoder(dataset, capacity=10, model=None, tw_max=30.0,
                                  replan_fn=GreedyReplanner('edd'))
    route, metrics = decoder.decode_instance(0)
    print(f"  route: {route.tolist()}")
    print(f"  complete: {metrics['complete']}, distance: {metrics['distance_cost']:.2f}")
    assert metrics['complete'], "贪心 decode 应服务所有客户"
    assert metrics['tw_feasible'] and metrics['capacity_feasible']

    print("\n✅ StrictOnlineDecoder (R1.5) 测试通过！")
    print("说明：事件驱动 + no premature close + service-completion event + "
          "execution-trace 评估。")
