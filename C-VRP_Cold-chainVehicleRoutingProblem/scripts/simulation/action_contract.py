"""
P0-A：full-fleet coupled Action Contract v1（02 手册 §P0-A / 01 主控 Layer II-III）。

在 strict-online 决策点（RecourseSnapshotV2 恢复的 fleet 状态）上，对**当前 customer j**
枚举可部署的 coupled action `a = (slot, insertion_position)`，并安装完整 full-fleet plan。

核心语义：
  - slot 集合 = anchored slots ∪ {NEW_ROUTE}（permutation-aware）：
      * ready@customer / committed 车 → anchored slot，稳定身份 = anchor_node（current_node
        或 committed_next），**不绑定 physical vehicle id**；
      * idle@depot 车 → 合并为匿名 NEW_ROUTE（存在 iff idle 车池非空）；
      * physical matching 在 action 选定后 deterministic（min vehicle_id）。
  - action = (customer, slot, position)；predecessor/successor 由 position 决定。
  - 候选必须通过完整 route replay certificate（customer TW + capacity + depot return）。
  - 安装 = 从原 plan 移除 j + 插入目标 slot/position；**其他车的 route 完全不变**；
    ownership exactly once（full-fleet 层面）。
  - incumbent（no-op，j 留在原位）永远在候选集。

边界（02 §P0-A4）：certificate 只保证当前 action 对当前已知 plan 约束可行，不保证未知
未来订单最终 complete。

author: P0-A
date: 2026-09-02
"""
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from strict_online_env import VehicleState


# ---------------------------------------------------------------------------
# 基础结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSlot:
    """slot 身份（stable，不绑定 physical vehicle id）。

    anchor 编码：
      - > 0  → ready@customer（current_node）或 committed（committed_next）车的 anchor_node；
      - < 0  → idle@depot 但已有非空 suffix 的 depot-route 车，vid = -anchor - 1（负编码）；
      - new_route → anchor = -1（匿名，physical matching 后置）。
    """
    kind: str        # 'anchored' | 'new_route'
    anchor: int


@dataclass(frozen=True)
class FleetAction:
    """对当前 customer 的 coupled action。"""
    customer: int
    slot: ActionSlot
    position: int          # 插入位置（0..len(suffix)）
    predecessor: int       # 插入后 j 的前驱 node（anchor_node 或 suffix 元素，0=depot）
    successor: int         # 插入后 j 的后继 node（suffix 元素或 0=return depot）
    incumbent: bool = False

    def action_id(self) -> str:
        return (f"c{self.customer}:{self.slot.kind}:{self.slot.anchor}"
                f":p{self.position}:{self.predecessor}>{self.successor}")


@dataclass
class VehiclePlan:
    """一辆 active 车的规划状态 + 有序 suffix（客户，不含尾部 0）。"""
    vehicle_id: int
    anchor_node: int
    anchor_time: float
    anchor_load: float
    suffix: tuple


FleetPlan = Dict[int, VehiclePlan]


def _dist(env, inst_idx, i, j):
    return float(env.dist_mat[inst_idx, int(i), int(j)])


def _travel(env, inst_idx, i, j):
    return _dist(env, inst_idx, i, j) / env.tw_speed


# ---------------------------------------------------------------------------
# plan / slot 提取
# ---------------------------------------------------------------------------

def build_vehicle_plans(env, inst_idx, vehicles) -> FleetPlan:
    """从 fleet 状态提取每辆 active 车的 (anchor_node, anchor_time, anchor_load, suffix)。"""
    plans = {}
    for v in vehicles:
        if v.status == 'closed':
            continue
        if v.status == 'committed' and v.committed_next not in (None, 0):
            node = int(v.committed_next)
            t = float(v.committed_finish)
            ld = float(v.current_load) + float(env.demands[inst_idx, node])
        else:
            node = int(v.current_node)
            t = float(v.ready_time)
            ld = float(v.current_load)
        plans[v.vehicle_id] = VehiclePlan(
            vehicle_id=v.vehicle_id, anchor_node=node, anchor_time=t, anchor_load=ld,
            suffix=tuple(int(x) for x in v.mutable_suffix if int(x) != 0))
    return plans


def build_slots(env, inst_idx, vehicles):
    """返回 (anchored_slots, new_route_available, idle_ids)。

    anchored_slots：list of (ActionSlot, vehicle_id, VehiclePlan)。
      - ready@customer / committed（anchor_node != 0）→ 身份 = anchor_node；
      - idle@depot 且已有非空 suffix → depot-route，身份 = -(vid+1)（负编码）；
    new_route_available：是否存在「空」idle@depot 车（同质，匿名 NEW_ROUTE）。
    idle_ids：空 idle 车 id（physical matching 用，min vid 优先）。
    """
    plans = build_vehicle_plans(env, inst_idx, vehicles)
    anchored = []
    idle_ids = []
    for vid, p in sorted(plans.items()):
        if p.anchor_node == 0 and not p.suffix:
            idle_ids.append(vid)
        elif p.anchor_node == 0:
            anchored.append((ActionSlot('anchored', -(vid + 1)), vid, p))
        else:
            anchored.append((ActionSlot('anchored', p.anchor_node), vid, p))
    return anchored, bool(idle_ids), idle_ids


# ---------------------------------------------------------------------------
# certificate（完整 route replay）
# ---------------------------------------------------------------------------

def certify_route(env, inst_idx, anchor_node, anchor_time, anchor_load, suffix):
    """对一条从 anchor 状态出发的 suffix 做完整 TW/capacity/return replay。

    返回 dict：feasible / reason / dist / tw_slack / cap_slack / return_slack。
    TW 语义与 authoritative_evaluator 一致：先判 arrive > tw_end，再等 tw_start。
    """
    cur = int(anchor_node)
    t = float(anchor_time)
    load = float(anchor_load)
    dist = 0.0
    tw_slack = float('inf')
    cap_slack = env.capacity - load
    reason = None
    for j in suffix:
        arrive = t + _travel(env, inst_idx, cur, j)
        if arrive > float(env.tw_end[inst_idx, j]) + 1e-6:
            return {'feasible': False, 'reason': 'tw', 'dist': dist,
                    'tw_slack': tw_slack, 'cap_slack': cap_slack, 'return_slack': None}
        start = max(arrive, float(env.tw_start[inst_idx, j]))
        load += float(env.demands[inst_idx, j])
        if load > env.capacity + 1e-6:
            return {'feasible': False, 'reason': 'capacity', 'dist': dist,
                    'tw_slack': tw_slack, 'cap_slack': cap_slack, 'return_slack': None}
        tw_slack = min(tw_slack, float(env.tw_end[inst_idx, j]) - arrive)
        cap_slack = min(cap_slack, env.capacity - load)
        dist += _dist(env, inst_idx, cur, j)
        t = start + float(env.service_time[inst_idx, j])
        cur = int(j)
    return_time = t + _travel(env, inst_idx, cur, 0)
    dist += _dist(env, inst_idx, cur, 0)
    return_slack = float(env.tw_end[inst_idx, 0]) - return_time
    if return_slack < -1e-6:
        return {'feasible': False, 'reason': 'return', 'dist': dist,
                'tw_slack': tw_slack, 'cap_slack': cap_slack, 'return_slack': return_slack}
    return {'feasible': True, 'reason': None, 'dist': dist,
            'tw_slack': tw_slack, 'cap_slack': cap_slack, 'return_slack': return_slack}


# ---------------------------------------------------------------------------
# full-fleet 安装与 ownership 校验
# ---------------------------------------------------------------------------

def find_customer_slot(plans, customer):
    """返回 customer 当前所在的 vehicle_id 与 suffix 中的 index；pool 未分配则 (None, None)。"""
    for vid, p in plans.items():
        if customer in p.suffix:
            return vid, p.suffix.index(customer)
    return None, None


def apply_action(plans, action, src_vid=None):
    """安装 action：从 src 车移除 customer（若已分配），插入目标 slot/position。

    返回新的 FleetPlan（deep copy 语义，原 plans 不变）。其他车 route 完全不变。
    """
    new = {vid: VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
           for vid, p in plans.items()}
    if src_vid is None:
        src_vid, _ = find_customer_slot(new, action.customer)
    # 1. 从原车移除 customer
    if src_vid is not None:
        p = new[src_vid]
        lst = list(p.suffix)
        if action.customer in lst:
            lst.remove(action.customer)
        new[src_vid] = VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                   tuple(lst))
    # 2. 插入目标 slot
    if action.slot.kind == 'new_route':
        vid = _match_new_route(new)
        p = new[vid]
        lst = list(p.suffix)
        lst.insert(action.position, action.customer)
        new[vid] = VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                               tuple(lst))
    else:
        vid = _match_anchored(new, action.slot.anchor)
        p = new[vid]
        lst = list(p.suffix)
        lst.insert(action.position, action.customer)
        new[vid] = VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                               tuple(lst))
    return new


def _match_new_route(plans):
    """NEW_ROUTE 的 deterministic physical matching：取「空」idle@depot 车的 min vehicle_id。"""
    idle = [vid for vid, p in plans.items() if p.anchor_node == 0 and not p.suffix]
    if not idle:
        raise ValueError("NEW_ROUTE 无可用空 idle 车")
    return min(idle)


def _match_anchored(plans, anchor):
    """anchored slot 的 physical matching。anchor>0 找 anchor_node；anchor<0 找 vid=-anchor-1。"""
    if anchor < 0:
        vid = -anchor - 1
        if vid in plans:
            return vid
    else:
        for vid, p in plans.items():
            if p.anchor_node == anchor:
                return vid
    raise ValueError(f"anchored slot anchor={anchor} 无对应车")


def validate_ownership(plans, universe):
    """full-fleet ownership 校验：每个 universe 客户恰好出现一次。返回 (ok, duplicated, missing)。"""
    from collections import Counter
    c = Counter()
    for p in plans.values():
        for x in p.suffix:
            c[int(x)] += 1
    duplicated = [k for k, v in c.items() if v > 1]
    missing = [k for k in universe if c.get(int(k), 0) == 0]
    extra = [k for k in c if int(k) not in set(int(u) for u in universe)]
    return (not duplicated and not missing and not extra), duplicated, missing, extra


def plan_hash(plans):
    """full-fleet plan 的确定性 hash（对每辆车的有序 suffix 敏感）。"""
    h = hashlib.sha256()
    for vid in sorted(plans):
        p = plans[vid]
        h.update(str(vid).encode())
        h.update(str(p.anchor_node).encode())
        h.update(repr(tuple(p.suffix)).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 候选枚举
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    action: FleetAction
    feasible: bool
    reason: Optional[str]
    incremental_distance: float
    tw_slack: Optional[float]
    cap_slack: Optional[float]
    return_slack: Optional[float]
    plan_hash: Optional[str]


def enumerate_actions(env, inst_idx, vehicles, customer):
    """对当前 customer 枚举所有 (slot, position) 候选（含 incumbent no-op）。

    返回 (candidates, plans)。candidates 已按 (incumbent 优先, slot, position) 排序。
    plans 为 full-fleet 当前 plan（调用方可传给 validate_ownership 做 full-fleet 校验）。
    """
    plans = build_vehicle_plans(env, inst_idx, vehicles)
    anchored, new_route_available, _ = build_slots(env, inst_idx, vehicles)
    src_vid, src_idx = find_customer_slot(plans, customer)

    cands = []

    # 对每个 anchored slot
    for slot, vid, p in anchored:
        base = list(p.suffix)
        if customer in base:
            base.remove(customer)
        for pos in range(len(base) + 1):
            pred = p.anchor_node if pos == 0 else base[pos - 1]
            succ = base[pos] if pos < len(base) else 0
            trial = base[:pos] + [customer] + base[pos:]
            is_inc = (src_vid == vid and src_idx is not None
                      and trial == list(plans[src_vid].suffix))
            cert = certify_route(env, inst_idx, p.anchor_node, p.anchor_time, p.anchor_load, trial)
            incr = (_dist(env, inst_idx, pred, customer) + _dist(env, inst_idx, customer, succ)
                    - _dist(env, inst_idx, pred, succ))
            act = FleetAction(customer=customer, slot=slot, position=pos,
                              predecessor=pred, successor=succ, incumbent=is_inc)
            if cert['feasible']:
                new_plan = apply_action(plans, act, src_vid=src_vid)
                cands.append(Candidate(act, True, None, incr, cert['tw_slack'],
                                       cert['cap_slack'], cert['return_slack'],
                                       plan_hash(new_plan)))
            else:
                cands.append(Candidate(act, False, cert['reason'], incr, None, None, None, None))

    # NEW_ROUTE（匿名）
    if new_route_available:
        slot = ActionSlot('new_route', -1)
        pred, succ = 0, 0
        vid = _match_new_route(plans)  # deterministic：min idle vehicle_id
        trial = [customer]
        cert = certify_route(env, inst_idx, 0, plans[vid].anchor_time, 0.0, trial)
        incr = _dist(env, inst_idx, 0, customer) + _dist(env, inst_idx, customer, 0)
        act = FleetAction(customer=customer, slot=slot, position=0,
                          predecessor=pred, successor=succ, incumbent=False)
        if cert['feasible']:
            new_plan = apply_action(plans, act, src_vid=src_vid)
            cands.append(Candidate(act, True, None, incr, cert['tw_slack'],
                                   cert['cap_slack'], cert['return_slack'], plan_hash(new_plan)))
        else:
            cands.append(Candidate(act, False, cert['reason'], incr, None, None, None, None))

    cands.sort(key=lambda c: (not c.action.incumbent, c.action.slot.kind,
                              c.action.slot.anchor, c.action.position))
    return cands, plans
