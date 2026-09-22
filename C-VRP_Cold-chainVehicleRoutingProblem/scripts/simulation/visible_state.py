"""N0：只读冷链可见状态视图 + 可见场景评价（可部署搜索参照的基础）。

两个职责：
1. `VisibleState`：决策点的只读公开状态（字段白名单，不含未来订单）。CC-LNS-RH 与后续
   MaskCO 共用。与 CC_Compare 的 DecisionView 不同，这里补齐订单级在舱货物与每车实时
   热状态（compartment_temperature_c / zone_load / cargo_manifest / cumulative_*）。
2. `evaluate_visible_plan`：公开评价场景 `J_vis(P | S_visible)` 的确定性执行——
   从当前公开状态出发，假设没有新订单到达，执行候选计划，再用固定贪婪规则补完已知订单
   并返仓，全程用 C0 转移累计 D/Q/E。这是近视场景，不是真实终局的无偏估计；不读未来。

固定规则（本工作包一次冻结）：
  - committed leg 只在真实 committed_finish 取货一次（不从 anchor 摘要重复累计）；
  - 空 suffix 车先保留（不虚构即时返仓），由补全规则统一处理剩余已知订单后再返仓；
  - 补全 = 按 tw_end 升序把剩余已知订单贪心插入最小增量距离的可行位置；无可行插入判失败；
  - 增量 D/Q/E 从当前状态起算（各候选共享起点，共同已发生成本自然抵消）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                             transition_segment, total_quality_loss)
from coldchain_contract import compute_coldchain_cost


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VehicleVisibleState:
    vid: int
    status: str                 # idle / ready / committed / returning / closed
    current_node: int           # 真实节点 id
    ready_time: float           # 当前可离开时刻（committed 车 = depart）
    current_load: float         # 当前载重（不含 committed_next）
    committed_next: int         # -1 无
    committed_arrive: float     # nan 无
    committed_finish: float     # nan 无
    coldchain_state: object     # VehicleColdChainState 或 None（idle 未派车）


@dataclass(frozen=True)
class VisibleState:
    """决策点只读公开状态。node_ids[0] = depot；其余为已知客户与车辆锚点。"""
    inst_idx: int
    event_id: int
    clock: float
    capacity: float
    tw_speed: float
    depot_tw_end: float
    has_future_reveal: bool
    node_ids: tuple               # 真实节点 id（0 打头，无重复）
    demands: tuple                # 按 node_ids 位置
    tw_start: tuple
    tw_end: tuple
    service_time: tuple
    temp_class: tuple
    initial_quality: tuple
    dist_mat: tuple               # (len, len) float，按 node_ids 位置
    travel_mat: tuple
    vehicles: tuple               # tuple[VehicleVisibleState, ...]
    pool_customer_ids: tuple      # 可变池（可重分配客户）
    deferred_customer_ids: tuple  # 已知 deferred（补全规则处理）
    replan_ids: tuple             # 本次可写车辆

    _idx: dict = None             # 内部缓存（node_id -> 位置）

    def idx(self, node_id: int) -> int:
        if self._idx is None:
            object.__setattr__(self, '_idx', {int(n): i for i, n in enumerate(self.node_ids)})
        return self._idx[int(node_id)]


def build_visible_state(env, inst_idx, clock, event_id, vehicles, served_mask, visible_ids,
                        replan_ids, deferred):
    """从决策点车辆/环境构造只读 VisibleState（只挑公开子集，结构上不含未来订单）。"""
    replan_set = set(int(v) for v in replan_ids)
    committed_ids = set()
    anchors = []
    protected = set()
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            committed_ids.add(int(v.committed_next))
            anchors.append(int(v.committed_next))
            if int(v.current_node) != 0:
                anchors.append(int(v.current_node))  # 出发节点（committed leg 距离需要）
        elif v.status in ('idle', 'ready', 'returning') and int(v.current_node) != 0:
            anchors.append(int(v.current_node))
        if int(v.vehicle_id) not in replan_set:
            if v.committed_next not in (None, 0):
                protected.add(int(v.committed_next))
            for n in v.mutable_suffix:
                if int(n) != 0:
                    protected.add(int(n))

    known_unserved = {int(c) for c in visible_ids if not bool(served_mask[int(c)])}
    pool = sorted(known_unserved - protected - committed_ids)
    deferred_ids = sorted(int(c) for c in deferred)
    # 已知节点 = depot + 锚点 + 可变池 + deferred + protected（补全/评价需要这些客户的属性）
    node_ids = [0] + sorted(set(anchors) | set(pool) | set(deferred_ids) | set(protected))
    assert len(set(node_ids)) == len(node_ids), 'view 节点重复'
    pos = {n: i for i, n in enumerate(node_ids)}

    coords_idx = np.asarray(node_ids, dtype=np.int32)
    dist = np.asarray(env.dist_mat[inst_idx], dtype=np.float32)[coords_idx][:, coords_idx]
    travel = dist / env.tw_speed

    vviews = []
    for v in vehicles:
        cn = int(v.committed_next) if v.committed_next not in (None, 0) else -1
        ca = float(v.committed_arrive) if v.committed_arrive is not None else float('nan')
        cf = float(v.committed_finish) if v.committed_finish is not None else float('nan')
        vviews.append(VehicleVisibleState(
            vid=int(v.vehicle_id), status=str(v.status), current_node=int(v.current_node),
            ready_time=float(v.ready_time), current_load=float(v.current_load),
            committed_next=cn, committed_arrive=ca, committed_finish=cf,
            coldchain_state=v.coldchain_state))

    def _arr(key, dtype):
        return tuple(dtype(env.__getattribute__(key)[inst_idx, n]) for n in node_ids)

    return VisibleState(
        inst_idx=int(inst_idx), event_id=int(event_id), clock=float(clock),
        capacity=float(env.capacity), tw_speed=float(env.tw_speed),
        depot_tw_end=float(env.tw_end[inst_idx, 0]),
        has_future_reveal=bool(env.has_future_reveal(inst_idx, clock, served_mask)),
        node_ids=tuple(node_ids),
        demands=_arr('demands', float),
        tw_start=_arr('tw_start', float),
        tw_end=_arr('tw_end', float),
        service_time=_arr('service_time', float),
        temp_class=_arr('temp_class', int),
        initial_quality=_arr('initial_quality', float),
        dist_mat=tuple(map(tuple, dist.tolist())),
        travel_mat=tuple(map(tuple, travel.tolist())),
        vehicles=tuple(vviews),
        pool_customer_ids=tuple(pool),
        deferred_customer_ids=tuple(deferred_ids),
        replan_ids=tuple(sorted(int(x) for x in replan_ids)),
        _idx=pos,
    )


# --------------------------------------------------------------------------- #
# 评价场景
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioResult:
    J_vis: float
    D: float
    Q: float
    E: float
    feasible: bool
    finite: bool
    n_known_completed: int
    detail: dict


def _route_feasible(vis, start_idx, start_time, start_load, route):
    """从锚点 (start_idx, start_time, start_load) 重放 route（客户位置索引），检查
    TW / capacity / 返仓。返回 (feasible, reason, incr_dist)。"""
    cur = start_idx
    t = start_time
    load = start_load
    dist = 0.0
    for ci in route:
        d = float(vis.dist_mat[cur][ci])
        arrive = t + d / vis.tw_speed
        if arrive > float(vis.tw_end[ci]) + 1e-6:
            return False, 'tw', dist
        start = max(arrive, float(vis.tw_start[ci]))
        load += float(vis.demands[ci])
        if load > vis.capacity + 1e-6:
            return False, 'capacity', dist
        dist += d
        t = start + float(vis.service_time[ci])
        cur = ci
    ret = t + float(vis.dist_mat[cur][0]) / vis.tw_speed
    dist += float(vis.dist_mat[cur][0])
    if ret > float(vis.depot_tw_end) + 1e-6:
        return False, 'return', dist
    return True, None, dist


def _insert_incremental(vis, start_idx, route, pos, ci):
    """把 ci 插入 route 位置 pos 的增量距离（假设前后可行）。"""
    pred = start_idx if pos == 0 else route[pos - 1]
    succ = 0 if pos == len(route) else route[pos]
    return (float(vis.dist_mat[pred][ci]) + float(vis.dist_mat[ci][succ])
            - float(vis.dist_mat[pred][succ]))


def _advance_one(state, contract, depart, arrive, finish, customer, order_qty, temp_cls,
                 init_q, distance, return_to_depot):
    """推进一辆车一段（旅行+等待+服务+取货/返仓），必要时先派车。"""
    if state is None:
        state = dispatch_vehicle(create_vehicle_state(contract), contract)
    active = (True,) * len(contract.thermal.supported_temp_classes)
    state, _ = transition_segment(
        state, depart_time=depart, arrival_time=arrive, service_finish=finish,
        served_customer=customer, active_zone_mask=active, contract=contract,
        order_quantity=order_qty, order_temp_class=temp_cls, initial_quality=init_q,
        return_to_depot=return_to_depot, segment_distance_units=distance)
    return state


def _delta(cs0, cs, contract):
    """段增量 (quality_loss, energy_kwh)。cs0 为 None 时是整段（idle 派车后）。"""
    if cs0 is not None:
        return (total_quality_loss(cs, contract) - total_quality_loss(cs0, contract),
                cs.cumulative_energy_kwh - cs0.cumulative_energy_kwh)
    return (total_quality_loss(cs, contract), cs.cumulative_energy_kwh)


def evaluate_visible_plan(vis, plan_suffixes, contract, objective):
    """执行可见场景：候选 plan_suffixes（vid -> tuple(客户真实 id)）→ 补全 → 返仓。

    plan_suffixes 是完整车队计划（含 frozen 车原 suffix；suffix 不含尾部 0）。
    返回 ScenarioResult。不可行 = 已知订单/返仓无法完成（非部分累计）。
    closed 车无剩余工作跳过；returning 车只推进在途返仓（frozen，delta 抵消）。
    """
    K = len(vis.vehicles)
    veh = [None] * K
    D = 0.0
    Q = 0.0
    E = 0.0
    n_picked = 0
    assigned = set()
    for k in range(K):
        v = vis.vehicles[k]
        if v.status == 'closed':
            continue
        cs = v.coldchain_state
        cur = vis.idx(v.current_node)
        t = float(v.ready_time)
        load = float(v.current_load)
        # returning：在途返仓（unload 于 return_finish = committed_finish），随后完成
        if v.status == 'returning':
            d = float(vis.dist_mat[cur][0])
            depart = min(float(vis.clock), float(v.committed_finish))
            cs0 = cs
            cs = _advance_one(cs, contract, depart, float(v.committed_finish),
                              float(v.committed_finish), None, 0.0, None, 1.0, d, True)
            dq, de = _delta(cs0, cs, contract)
            Q += dq
            E += de
            D += d
            continue
        # committed leg（在途承诺；冷链状态已在 clock 推进，只从 clock 再推到 committed_finish）
        if v.committed_next > 0:
            ci = vis.idx(v.committed_next)
            d = float(vis.dist_mat[cur][ci])
            depart = min(float(vis.clock), float(v.committed_finish))
            cs0 = cs
            cs = _advance_one(cs, contract, depart, float(v.committed_finish),
                              float(v.committed_finish), v.committed_next,
                              float(vis.demands[ci]), int(vis.temp_class[ci]),
                              float(vis.initial_quality[ci]), d, False)
            dq, de = _delta(cs0, cs, contract)
            Q += dq
            E += de
            D += d
            n_picked += 1
            assigned.add(v.committed_next)
            cur = ci
            t = float(v.committed_finish)
            load += float(vis.demands[ci])
        # suffix
        for c in plan_suffixes.get(v.vid, ()):
            if int(c) <= 0:
                continue
            ci = vis.idx(int(c))
            d = float(vis.dist_mat[cur][ci])
            arrive = t + d / vis.tw_speed
            start = max(arrive, float(vis.tw_start[ci]))
            finish = start + float(vis.service_time[ci])
            cs0 = cs
            cs = _advance_one(cs, contract, t, arrive, finish, int(c),
                              float(vis.demands[ci]), int(vis.temp_class[ci]),
                              float(vis.initial_quality[ci]), d, False)
            dq, de = _delta(cs0, cs, contract)
            Q += dq
            E += de
            D += d
            n_picked += 1
            assigned.add(int(c))
            cur = ci
            t = finish
            load += float(vis.demands[ci])
        veh[k] = {'cs': cs, 'anchor': cur, 't': t, 'load': load, 'route': [],
                  'vid': v.vid, 'start_idx': cur}

    # 补全：剩余已知订单 = (可变池 ∪ deferred) 中尚未 assigned 的（去重；deferred 也在池内）
    known = set(vis.pool_customer_ids) | set(vis.deferred_customer_ids)
    remaining = [c for c in known if int(c) not in assigned]
    remaining = sorted(remaining, key=lambda c: (float(vis.tw_end[vis.idx(c)]), int(c)))
    unassigned = []
    for c in remaining:
        ci = vis.idx(c)
        best = None
        for vk in range(K):
            e = veh[vk]
            if e is None:
                continue
            route = e['route']
            start_idx = e['start_idx']
            for pos in range(len(route) + 1):
                trial = route[:pos] + [ci] + route[pos:]
                ok, _reason, _d = _route_feasible(vis, start_idx, e['t'], e['load'], trial)
                if ok:
                    incr = _insert_incremental(vis, start_idx, route, pos, ci)
                    if best is None or incr < best[0]:
                        best = (incr, vk, pos)
        if best is None:
            unassigned.append(int(c))
            continue
        _incr, vk, pos = best
        veh[vk]['route'].insert(pos, ci)
        assigned.add(int(c))

    feasible = (len(unassigned) == 0)

    # 返仓（补全后每车回 depot）并累计补全段冷量；未派车且无任务的车保持 idle，增量为 0
    for e in veh:
        if e is None:
            continue
        cur = e['anchor']
        t = e['t']
        load = e['load']
        cs = e['cs']
        if cs is None and not e['route']:
            # 从未派车（无 committed/suffix）且补全也未分配任务 → 保持未派车，不收预冷
            continue
        for ci in e['route']:
            d = float(vis.dist_mat[cur][ci])
            arrive = t + d / vis.tw_speed
            start = max(arrive, float(vis.tw_start[ci]))
            finish = start + float(vis.service_time[ci])
            c = vis.node_ids[ci]
            cs0 = cs
            cs = _advance_one(cs, contract, t, arrive, finish, c,
                              float(vis.demands[ci]), int(vis.temp_class[ci]),
                              float(vis.initial_quality[ci]), d, False)
            dq, de = _delta(cs0, cs, contract)
            Q += dq
            E += de
            D += d
            n_picked += 1
            cur = ci
            t = finish
            load += float(vis.demands[ci])
        # return
        d = float(vis.dist_mat[cur][0])
        arrive = t + d / vis.tw_speed
        cs0 = cs
        cs = _advance_one(cs, contract, t, arrive, arrive, None, 0.0, None, 1.0, d, True)
        dq, de = _delta(cs0, cs, contract)
        Q += dq
        E += de
        D += d

    J = compute_coldchain_cost(D, Q, E, objective)
    finite = all(np.isfinite(x) for x in (J, D, Q, E))
    return ScenarioResult(
        J_vis=float(J), D=float(D), Q=float(Q), E=float(E), feasible=feasible,
        finite=finite, n_known_completed=n_picked,
        detail={'unassigned': unassigned, 'n_picked': n_picked, 'n_remaining': len(remaining)})
