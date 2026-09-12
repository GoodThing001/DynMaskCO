"""独立 pickup 认证器（R0.5 Stage A，纯 NumPy/stdlib）。

从真实 anchor/time/load 出发，逐次追加时检查时间窗 / 累积容量 / 服务后按时返仓。
与 route_mapper.certify_suffix_float 同口径（depot 恒在 node_ids 索引 0）。

pickup-to-depot 语义：
  - 服务客户后载荷**增加**（取货入舱）；
  - 仅终端的 depot（尾 0）才结束本次行程；
  - 空路线 `()` = WAIT，`(0,)` = CLOSE（返仓），二者语义严格区分。
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class VehicleState:
    """单辆车当前的公开状态（不可变）。"""
    pos_node_id: int   # 当前最后节点（真实 id；anchor 起）
    time: float        # 可出发时刻
    load: float        # 当前载荷
    suffix: tuple      # 已服务客户（真实 id，无尾 0）


def _idx(view, node_id):
    return view.node_ids.index(int(node_id))


def certify_append(view, state, customer_id):
    """检查把 customer 追加到 state 是否可行（TW + 累积容量 + 服务后按时返仓）。

    返回 (ok, reason)。保守口径：假设该客户是最后一个，服务后直接返仓必须
    在 depot_tw_end 内——这是任何「含该客户的可行路线」的必要条件。
    """
    c = int(customer_id)
    if c == 0:
        return False, 'internal depot（depot 只作尾 0，不得作客户）'
    if c not in view.pool_customer_ids:
        return False, f'customer {c} 不在可变池（protected/未来/已服务）'
    if c not in view.node_ids:
        return False, f'external customer {c}'
    if c in state.suffix:
        return False, f'duplicate customer {c}'
    ic = _idx(view, c)
    ip = _idx(view, state.pos_node_id)
    arrive = state.time + float(view.travel_mat[ip][ic])
    if arrive > float(view.tw_end[ic]) + 1e-6:
        return False, f'tw: arrive {arrive:.4f} > tw_end {view.tw_end[ic]:.4f}'
    start = max(arrive, float(view.tw_start[ic]))
    new_load = state.load + float(view.demands[ic])
    if new_load > float(view.capacity) + 1e-6:
        return False, f'capacity: load {new_load:.4f} > {view.capacity}'
    ret = start + float(view.service_time[ic]) + float(view.travel_mat[ic][0])
    if ret > float(view.depot_tw_end) + 1e-6:
        return False, f'return: {ret:.4f} > depot_tw_end {view.depot_tw_end:.4f}'
    return True, None


def append_state(view, state, customer_id):
    """应用一次合法追加，返回新 state（假定已通过 certify_append）。"""
    c = int(customer_id)
    ic = _idx(view, c)
    ip = _idx(view, state.pos_node_id)
    arrive = state.time + float(view.travel_mat[ip][ic])
    start = max(arrive, float(view.tw_start[ic]))
    new_time = start + float(view.service_time[ic])
    new_load = state.load + float(view.demands[ic])
    return VehicleState(c, new_time, new_load, state.suffix + (c,))


def certify_suffix(view, vehicle, suffix):
    """权威证书：完整重放一条 suffix（真实 id，尾 0 已由调用方约定）。

    suffix 语义：
      ()       = WAIT（恒可行，原地等待）
      (0,)     = CLOSE（返仓；检查 anchor→depot 返仓时间）
      (c1..ck, 0) = 服务 c1..ck 后返仓；逐客户 TW/容量 + 终局返仓。
    返回 (ok, reason)。
    """
    suffix = tuple(int(x) for x in suffix)
    # 空路线：WAIT（()）恒可行；CLOSE（(0,)）检查返仓
    if suffix == ():
        return True, None
    if suffix == (0,):
        ip = _idx(view, vehicle.anchor_node_id)
        ret = float(vehicle.ready_time) + float(view.travel_mat[ip][0])
        if ret > float(view.depot_tw_end) + 1e-6:
            return False, f'close return: {ret:.4f} > depot_tw_end'
        return True, None
    # 非空路线必须以 0 结尾，且内部不得出现 0
    if suffix[-1] != 0:
        return False, 'non-empty suffix 未以 depot(0) 结尾'
    customers = suffix[:-1]
    if 0 in customers:
        return False, 'suffix 内部含 depot(0)'
    state = VehicleState(int(vehicle.anchor_node_id), float(vehicle.ready_time),
                         float(vehicle.load), ())
    for c in customers:
        ok, reason = certify_append(view, state, c)
        if not ok:
            return False, f'append {c}: {reason}'
        state = append_state(view, state, c)
    return True, None


def wait_or_close(view, vehicle):
    """空 suffix 的 WAIT/CLOSE 规则（对齐 OR-Tools/PyVRP empty_suffix_policy）。"""
    if not view.has_future_reveal:
        return (0,)           # 无未来揭示：返仓
    ip = _idx(view, vehicle.anchor_node_id)
    ret = float(vehicle.ready_time) + float(view.travel_mat[ip][0])
    if ret <= float(view.depot_tw_end) + 1e-6:
        return ()             # 原地等待
    return (0,)               # 必须现在返仓
