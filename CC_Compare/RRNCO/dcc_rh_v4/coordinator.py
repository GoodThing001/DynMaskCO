"""确定性多车协调器（R0.5 Stage A.1，纯 NumPy/stdlib）。

接收每辆 replan 车的一份**预计算模型排序**（orderings: vid -> tuple），贪心分配；
跨车竞争按固定顺序裁决：

    是否开启新车辆 → 模型归一化排名 → 增量距离 → vehicle_id → customer_id

- 「开启新车辆」代价：已有 suffix / 非 depot anchor / load>0 的已启动车辆 = 0；
  空载 depot 闲置车 = 1。即优先续用已启动车辆，仅当现有车辆无法安全追加时才
  启用新车（外层负责 fleet packing，故正式名称为 RRNCO-Ordering-RH-D）。
- 增量距离只用于同排名 tie-break，绝不替代模型排序。
- 已分配客户跨车唯一；暂时不可行 → deferred（非 fallback）；排序非法 → EDD fallback。
"""
from dataclasses import dataclass, field

from preference import validate_ordering
from pickup_certificate import (VehicleState, certify_append, append_state,
                                wait_or_close)


@dataclass
class CoordinationResult:
    suffixes: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    deferred: tuple = ()
    assigned: tuple = ()
    fallback_reason: str = None


def _node_idx(view, node_id):
    return view.node_ids.index(int(node_id))


def _open_cost(state):
    """是否已启动该车：已启动=0（优先续用），空载 depot 闲置=1（新开）。"""
    started = bool(state.suffix) or int(state.pos_node_id) != 0 \
        or float(state.load) > 1e-9
    return 0 if started else 1


def _coord_edd(view, replan, pool, reason='EDD fallback（模型偏好不可用）'):
    """EDD 安全 fallback：按 EDD 顺序 + 独立认证贪心分配（不依赖模型排序）。"""
    idx = {n: i for i, n in enumerate(view.node_ids)}
    states = {int(v.vehicle_id): VehicleState(int(v.anchor_node_id),
                                              float(v.ready_time), float(v.load), ())
              for v in replan}
    assigned = set()
    decisions = []
    edd_order = sorted(pool, key=lambda c: float(view.tw_end[idx[c]]))
    for c in edd_order:
        for v in sorted(replan, key=lambda v: int(v.vehicle_id)):
            vid = int(v.vehicle_id)
            ok, _ = certify_append(view, states[vid], c)
            if ok:
                states[vid] = append_state(view, states[vid], c)
                assigned.add(c)
                decisions.append({'vehicle_id': vid, 'customer_id': c,
                                  'rank': None, 'incremental_distance': None,
                                  'fallback': True})
                break
    suffixes = {}
    for v in replan:
        vid = int(v.vehicle_id)
        suffixes[vid] = (states[vid].suffix + (0,)) if states[vid].suffix \
            else wait_or_close(view, v)
    return CoordinationResult(
        suffixes=suffixes, decisions=decisions,
        deferred=tuple(sorted(pool - assigned)),
        assigned=tuple(sorted(assigned)),
        fallback_reason=reason,
    )


def coordinate(view, orderings):
    """协调所有 replan 车（orderings 已由 adapter 预计算并校验）。

    orderings: {vehicle_id -> tuple[int, ...]}（客户真实 id，最偏好在前）。
    任一排序非法/缺失 → EDD fallback。
    """
    replan = sorted((v for v in view.vehicles
                     if int(v.vehicle_id) in set(int(x) for x in view.replan_ids)),
                    key=lambda v: int(v.vehicle_id))
    pool = set(int(c) for c in view.pool_customer_ids)
    n_pool = len(pool)

    # ---- 防御：排序必须完整覆盖 pool ----
    for v in replan:
        problems = validate_ordering(orderings.get(int(v.vehicle_id), ()),
                                     view.pool_customer_ids)
        if problems:
            return _coord_edd(view, replan, pool,
                              reason='; '.join(problems))

    states = {int(v.vehicle_id): VehicleState(int(v.anchor_node_id),
                                              float(v.ready_time), float(v.load), ())
              for v in replan}
    assigned = set()
    decisions = []
    idx = {n: i for i, n in enumerate(view.node_ids)}
    while True:
        candidates = []   # (vid, customer, rank_idx, incremental_dist, open_cost)
        for vid in sorted(orderings):
            for c in orderings[vid]:
                if c in assigned:
                    continue
                ok, _ = certify_append(view, states[vid], c)
                if ok:
                    rank = orderings[vid].index(c)
                    dist = float(view.dist_mat[idx[states[vid].pos_node_id]][idx[c]])
                    candidates.append((vid, c, rank, dist, _open_cost(states[vid])))
                    break
        if not candidates:
            break
        best = min(candidates,
                   key=lambda t: (t[4], t[2] / max(1, n_pool), t[3], t[0], t[1]))
        vid, c, rank, dist, oc = best
        states[vid] = append_state(view, states[vid], c)
        assigned.add(c)
        decisions.append({'vehicle_id': vid, 'customer_id': c, 'rank': rank,
                          'rank_normalized': rank / max(1, n_pool),
                          'incremental_distance': dist,
                          'open_new_vehicle': oc, 'fallback': False})

    suffixes = {}
    for v in replan:
        vid = int(v.vehicle_id)
        suffixes[vid] = (states[vid].suffix + (0,)) if states[vid].suffix \
            else wait_or_close(view, v)

    return CoordinationResult(
        suffixes=suffixes, decisions=decisions,
        deferred=tuple(sorted(pool - assigned)),
        assigned=tuple(sorted(assigned)),
        fallback_reason=None,
    )
