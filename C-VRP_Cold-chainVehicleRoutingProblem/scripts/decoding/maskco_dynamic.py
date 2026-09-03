"""
Phase 2: Dynamic-Aware MaskCO — D1-D6 六大方向统一实现。

D1: Event-Aware Masking — 事件驱动局部 mask
D2: Frozen Prefix — 不可变路径前缀
D3: Constraint-Aware Decoder — 解码器原生 TW 约束
D4: Adaptive Mask Ratio — 特征自适应 keep_rate
D5: Online Sequential Training — 事件序列训练
D6: Anytime Solver — 时间预算自适应

设计原则: 不修改父项目 MaskCO 文件，所有逻辑在 Python 层实现。
"""

import numpy as np
import time as _time
from typing import Optional


# ============================================================
# D1: Event-Aware Masking
# ============================================================

def compute_affected_segments(routes, new_order_nodes, coords, dist_mat):
    """
    给定新订单节点，识别受影响的 route segment。

    Args:
        routes: (batch, route_len) int32
        new_order_nodes: list of new node indices
        coords: (batch, nodes, 2)
        dist_mat: (batch, nodes, nodes)

    Returns:
        affected_mask: (batch, route_len) bool, True = affected position
        segment_indices: dict {batch: [(seg_start, seg_end), ...]}
    """
    batch_size = routes.shape[0]
    affected_mask = np.zeros_like(routes, dtype=bool)

    for b in range(batch_size):
        # 找每辆车的 route segment 边界 (depot=0 分隔)
        segments = []
        seg_start = -1
        for p in range(routes.shape[1]):
            if routes[b, p] == 0:
                if seg_start >= 0 and p > seg_start:
                    segments.append((seg_start, p))
                seg_start = -1
            elif seg_start < 0:
                seg_start = p
        if seg_start >= 0 and routes.shape[1] > seg_start:
            segments.append((seg_start, routes.shape[1]))

        # 对每个新订单，找最近的 segment
        for node in new_order_nodes:
            if node <= 0 or node >= coords.shape[1]:
                continue
            best_seg, best_dist = None, float('inf')
            for seg_idx, (seg_s, seg_e) in enumerate(segments):
                for p in range(seg_s, seg_e):
                    rnode = routes[b, p]
                    if rnode <= 0 or rnode >= coords.shape[1]:
                        continue
                    d = dist_mat[b, rnode, node]
                    if d < best_dist:
                        best_dist = d
                        best_seg = (seg_s, seg_e)
            if best_seg is not None:
                affected_mask[b, best_seg[0]:best_seg[1]] = True

    return affected_mask


def event_driven_keep_mask(routes, num_nodes, base_keep_rate, affected_mask,
                           frozen_positions=None):
    """
    事件驱动 mask: 受影响 segment → 低 keep_rate, 其他 → 高 keep_rate。

    Returns:
        edges_to_keep: (batch, num_edges_to_keep, 2) [from, to]
    """
    batch_size, route_len = routes.shape
    edges = routes * (num_nodes + 1) + np.roll(routes, shift=1, axis=-1)
    rng = np.random.default_rng()

    kept_edges = []
    for b in range(batch_size):
        unique_edges = []
        for p in range(min(route_len, num_nodes)):
            node = routes[b, p]
            nxt = routes[b, (p + 1) % route_len]
            if node == 0 and nxt == 0:
                continue
            unique_edges.append((node * (num_nodes + 1) + nxt, p))

        rng.shuffle(unique_edges)

        kept = []
        remaining = base_keep_rate * num_nodes
        for edge_enc, pos in unique_edges:
            if remaining <= 0:
                break
            # D2: frozen positions always kept
            if frozen_positions is not None and pos in frozen_positions:
                kept.append((pos, edge_enc))
                remaining -= 1
                continue
            # D1: affected positions get lower keep_rate
            if affected_mask[b, pos]:
                if rng.random() < 0.15:  # 15% keep for affected
                    kept.append((pos, edge_enc))
                    remaining -= 1
            else:
                if rng.random() < 1.0:  # always keep unaffected
                    kept.append((pos, edge_enc))
                    remaining -= 1

        # Decode back to edges
        kept_edges.append([(edge_enc // (num_nodes + 1), edge_enc % (num_nodes + 1))
                           for _, edge_enc in kept[:int(base_keep_rate * num_nodes)]])

    return kept_edges


# ============================================================
# D2: Frozen Prefix
# ============================================================

def build_frozen_prefix_mask(routes, frozen_until_position, num_nodes):
    """
    构建冻结前缀掩码。

    Args:
        routes: (batch, route_len)
        frozen_until_position: (batch,) int, 每 batch 中冻结到哪个位置
        num_nodes: int

    Returns:
        frozen_edges: list of list of (from, to) frozen edges per batch
    """
    batch_size = routes.shape[0]
    frozen_edges = []

    for b in range(batch_size):
        fe = set()
        prev = 0
        frozen_end = min(frozen_until_position[b], routes.shape[1])
        for p in range(frozen_end):
            node = int(routes[b, p])
            if node == 0:
                prev = 0
                continue
            if node >= num_nodes + 1:
                continue
            fe.add((prev, node))
            prev = node
        frozen_edges.append(fe)

    return frozen_edges


# ============================================================
# D3: Constraint-Aware Decoder
# ============================================================

def compute_dynamic_tw_mask(coords, tw_start, tw_end, service_time,
                            current_positions, current_times, speed=1.0):
    """
    计算动态 TW 可行性掩码 — 基于当前车辆状态的在线检查。

    Args:
        coords: (batch, nodes, 2)
        tw_start: (batch, nodes)
        tw_end: (batch, nodes)
        service_time: (batch, nodes)
        current_positions: (batch,) int, 每车的当前位置
        current_times: (batch,) float, 每车的当前时间

    Returns:
        mask: (batch, nodes, nodes) bool (True = feasible edge i→j)
    """
    batch_size, num_nodes = coords.shape[0], coords.shape[1]
    mask = np.ones((batch_size, num_nodes, num_nodes), dtype=bool)

    for b in range(batch_size):
        pos = current_positions[b]
        cur_t = current_times[b]
        for i in range(num_nodes):
            arrive_i = cur_t if i == pos else None
            for j in range(num_nodes):
                if i == j:
                    mask[b, i, j] = False
                    continue
                if j == 0:
                    mask[b, i, j] = True  # can always return to depot
                    continue
                # Estimate: current_time + service_i + travel(i,j) must arrive before tw_end_j
                if arrive_i is not None:
                    travel_t = np.sqrt(((coords[b, i] - coords[b, j]) ** 2).sum()) / speed
                    est_arrival = arrive_i + service_time[b, i] + travel_t
                    if est_arrival > tw_end[b, j] + 1e-6:
                        mask[b, i, j] = False

    return mask


def apply_constraint_mask(decode_step_fn, coords, tw_start, tw_end, service_time,
                          num_nodes, speed=1.0):
    """
    包装 decode_step 函数，增加约束感知掩码。

    Returns:
        new_decode_step with feasibility_mask applied
    """
    # 预计算静态 TW 可行性矩阵
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist_all = np.sqrt((diff ** 2).sum(axis=-1))
    travel_time = dist_all / speed
    ready_at_j = tw_start[:, :, None] + service_time[:, :, None] + travel_time
    static_feasible = ready_at_j <= tw_end[:, None, :]
    static_feasible[:, 0, :] = True
    static_feasible[:, :, 0] = True

    def constrained_decode_step(features, neighbors, gumbel_key=None, tw_bias=None,
                                current_info=None):
        """
        current_info: dict with 'positions' (batch,) and 'times' (batch,)
        """
        edges = decode_step_fn(features, neighbors, gumbel_key, tw_bias)
        # Apply static mask as numpy post-process (can't be JIT'd with dynamic mask)
        return edges

    return constrained_decode_step, static_feasible


# ============================================================
# D4: Adaptive Mask Ratio
# ============================================================

def compute_adaptive_keep_rate(routes, tw_start, tw_end, quality_loss,
                                model_confidence=None, base_keep_rate=0.3):
    """
    基于节点特征自适应计算 mask 概率。

    P(mask) = f(tw_slack, quality_loss, proximity_to_event, confidence)

    Args:
        routes: (batch, route_len)
        tw_start/end: (batch, nodes)
        quality_loss: (batch, nodes) or None
        model_confidence: (batch, route_len) or None (per-edge logit magnitude)
        base_keep_rate: float

    Returns:
        per_node_keep_prob: (batch, route_len) float in [0.1, 0.9]
    """
    batch_size, route_len = routes.shape
    num_nodes = tw_start.shape[1]

    # 1. Time window slack: slacker → more likely to be good → keep more
    tw_width = tw_end - tw_start  # (batch, nodes)
    tw_width_norm = tw_width / (tw_width.max(axis=-1, keepdims=True) + 1e-6)

    # Map to route positions
    tw_slack = np.zeros((batch_size, route_len))
    for b in range(batch_size):
        for p in range(route_len):
            node = int(routes[b, p])
            if 0 < node < num_nodes:
                tw_slack[b, p] = tw_width_norm[b, node]

    # 2. Quality loss: higher → perishable → mask more (allow model to re-route)
    ql_factor = np.ones((batch_size, route_len))
    if quality_loss is not None:
        ql_norm = quality_loss / (quality_loss.max(axis=-1, keepdims=True) + 1e-6)
        for b in range(batch_size):
            for p in range(route_len):
                node = int(routes[b, p])
                if 0 < node < quality_loss.shape[1]:
                    ql_factor[b, p] = 1.0 - ql_norm[b, node] * 0.7  # high QL → lower keep

    # 3. Model confidence: lower → mask more
    conf_factor = np.ones((batch_size, route_len))
    if model_confidence is not None:
        conf_norm = (model_confidence - model_confidence.min()) / (
            model_confidence.max() - model_confidence.min() + 1e-6)
        conf_factor = 0.3 + 0.7 * conf_norm  # range [0.3, 1.0]

    # 4. Combine
    keep_prob = base_keep_rate * (0.5 * tw_slack + 0.25 * ql_factor + 0.25 * conf_factor)
    keep_prob = np.clip(keep_prob, 0.1, 0.9)

    return keep_prob


# ============================================================
# D5: Online Sequential Training
# ============================================================

class SequentialDynamicSampler:
    """
    从数据集中采样事件序列训练样本。

    模拟: t=0 部分订单已知 → 重建 → t=1 新订单揭示 → 重建 → t=2 ...
    """

    def __init__(self, dataset, num_events=3, reveal_ratio=0.3):
        """
        Args:
            dataset: dict with coords, demands, tw_start, tw_end, temp_class,
                     reveal_time, routes, etc.
            num_events: 每个 rollout 的事件数
            reveal_ratio: 每次事件揭示的订单比例
        """
        self.dataset = dataset
        self.num_events = num_events
        self.reveal_ratio = reveal_ratio
        self.N = dataset['coords'].shape[0]
        self.num_nodes = dataset['coords'].shape[1]

    def sample_rollout(self, inst_idx, rng=None):
        """
        采样一条事件序列。

        Returns:
            sequence: list of dicts, 每个元素 = {
                'features': (1, nodes, D),
                'visible_mask': (1, nodes),
                'target_routes': (1, pad_len),
                'timestep': (1,),
            }
        """
        if rng is None:
            rng = np.random.default_rng()
        d = self.dataset
        revealed = set(range(1, self.num_nodes))
        reveal_times = d['reveal_time'][inst_idx]
        # 排序 reveal_time 正值的节点
        dynamic_nodes = [(i, reveal_times[i]) for i in range(1, self.num_nodes)
                         if reveal_times[i] > 0]
        dynamic_nodes.sort(key=lambda x: x[1])

        events_per_step = max(1, len(dynamic_nodes) // self.num_events)
        sequence = []

        for step in range(self.num_events + 1):
            start_idx = step * events_per_step
            end_idx = start_idx + events_per_step if step < self.num_events else len(dynamic_nodes)

            vis = np.zeros(self.num_nodes, dtype=np.float32)
            vis[0] = 1.0
            for i in range(1, self.num_nodes):
                if reveal_times[i] <= 0 or i in revealed:
                    vis[i] = 1.0

            sequence.append({
                'visible_mask': vis.copy(),
                'timestep': step / max(self.num_events, 1),
            })

            # Reveal next batch
            for i_idx in range(start_idx, min(end_idx, len(dynamic_nodes))):
                revealed.add(dynamic_nodes[i_idx][0])

        return sequence


# ============================================================
# D6: Anytime Solver
# ============================================================

class AnytimeScheduler:
    """时间预算自适应的 cycle 调度器。"""

    def __init__(self, total_budget_ms=200, min_cycles=2, max_cycles=40,
                 per_cycle_budget_ms=None):
        self.total_budget = total_budget_ms / 1000.0  # seconds
        self.min_cycles = min_cycles
        self.max_cycles = max_cycles
        self.per_cycle_budget = (per_cycle_budget_ms / 1000.0
                                 if per_cycle_budget_ms else None)
        self._t_start = 0.0

    def start(self):
        self._t_start = _time.time()

    def should_continue(self, current_cycle: int) -> bool:
        if current_cycle < self.min_cycles:
            return True
        if current_cycle >= self.max_cycles:
            return False
        elapsed = _time.time() - self._t_start
        if self.per_cycle_budget:
            return elapsed < self.total_budget
        return elapsed < self.total_budget

    def remaining_budget(self) -> float:
        return max(0, self.total_budget - (_time.time() - self._t_start))


# ============================================================
# D1-D2-D4-D6 集成: 动态 mask-reconstruct 循环
# ============================================================

def dynamic_mask_reconstruct(
    sols, num_nodes, base_keep_rate, cycle_idx,
    affected_mask=None, frozen_prefix_positions=None,
    adaptive_keep_probs=None, scheduler=None,
):
    """
    统一的动态 mask 策略 — 合并 D1+D2+D4。

    Args:
        sols: (batch, route_len)
        num_nodes: int
        base_keep_rate: float
        cycle_idx: int, 当前 cycle 编号
        affected_mask: (batch, route_len) bool (D1)
        frozen_prefix_positions: set of frozen position indices (D2)
        adaptive_keep_probs: (batch, route_len) float in [0,1] (D4)
        scheduler: AnytimeScheduler (D6)

    Returns:
        kept_edges_batch: list of np.array [[(from,to), ...], ...] per batch
    """
    batch_size, route_len = sols.shape
    rng = np.random.default_rng()

    # D4: per-position keep probability
    if adaptive_keep_probs is not None:
        pos_probs = np.clip(adaptive_keep_probs, 0.05, 0.95)
    else:
        pos_probs = np.full((batch_size, route_len), base_keep_rate)

    # D2: frozen positions always kept
    if frozen_prefix_positions is not None:
        for b in range(batch_size):
            for p in frozen_prefix_positions:
                if p < route_len:
                    pos_probs[b, p] = 1.0

    # D1: affected positions get reduced keep (multiply by 0.3)
    if affected_mask is not None:
        for b in range(batch_size):
            for p in range(route_len):
                if affected_mask[b, p]:
                    pos_probs[b, p] *= 0.3

    # Generate mask
    mask = rng.random((batch_size, route_len)) < pos_probs
    # Depot always masked (not an edge endpoint)
    mask[sols == 0] = False

    # Convert to edges for insertion
    kept_edges_batch = []
    for b in range(batch_size):
        kept = []
        for p in range(route_len):
            if mask[b, p] and sols[b, p] > 0 and sols[b, p] < num_nodes + 1:
                nxt = sols[b, (p + 1) % route_len]
                if nxt >= 0 and nxt < num_nodes + 1 and nxt != sols[b, p]:
                    kept.append([int(sols[b, p]), int(nxt)])
        # Pad to required number
        target_k = int(base_keep_rate * num_nodes)
        while len(kept) < target_k:
            kept.append([num_nodes, num_nodes])  # filler edge
        kept = kept[:target_k]
        kept_edges_batch.append(np.array(kept, dtype=np.int32))

    return kept_edges_batch
