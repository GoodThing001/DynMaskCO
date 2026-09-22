"""event_plan_v2：可见执行后果投影（B 表示引擎）。

从保存的可见状态 + P0 + P 出发，用唯一 C0 转移（coldchain_state.transition_segment）与
冻结 contract/profile，把候选计划 P 的「有序路线时间/资源 + 冷链 D/Q/E/J 代理」投影出来。

严格信息边界：
  - 不读取未来 reveal、不插入未揭示客户、不猜哪段会被未来清空；
  - 不调用含真实未来的 StrictOnlineEnv.run 来造特征；
  - 不拿反事实终局 D/Q/E 当输入（那是 g_B 标签）；
  - 代理是「当前已知计划」的物理投影，不是完整 episode 成本。

固定终点为公开 depot horizon H；WAIT / 未闭合路线记录 open/WAIT 位，不虚构即时返仓。
"""
import numpy as np

from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                             transition_segment, total_quality_loss)
from coldchain_contract import compute_coldchain_cost


def _active_mask(contract):
    return (True,) * len(contract.thermal.supported_temp_classes)


def project_vehicle(env, inst_idx, v, plan, contract, has_future):
    """投影单辆车执行 plan 的物理/冷链后果（从当前可见状态推进到返仓或 WAIT）。

    plan: VehiclePlan（anchor_node / anchor_time / anchor_load / suffix，suffix 不含尾 0）。
    v:    VehicleState（含 coldchain_state；committed 车的 committed leg 单独推进一次）。

    返回 dict（绝对增量，从当前状态算）：
      valid, anchor_node, anchor_time, anchor_load, is_committed, n_customers,
      distance, return_time, min_tw_slack, max_load, wait_time, open_plan,
      d_quality, d_energy, d_distance, has_cargo, cargo_age, cargo_quality, cargo_count
    """
    horizon = float(env.tw_end[inst_idx].max())
    speed = env.tw_speed
    active = _active_mask(contract)

    anchor_node = int(plan.anchor_node)
    t = float(plan.anchor_time)
    load = float(plan.anchor_load)
    suffix = [int(x) for x in plan.suffix if int(x) != 0]

    cs = v.coldchain_state  # None（idle/未派车）或 VehicleColdChainState
    cs0 = cs

    total_dist = 0.0
    wait_total = 0.0
    min_slack = float('inf')
    max_load = load
    n_customers = 0

    cur_node = int(v.current_node)
    is_committed = (v.status == 'committed' and v.committed_next not in (None, 0))

    # 1. committed leg（在途不可撤销承诺；pickup 尚未入舱）
    if is_committed:
        node = int(v.committed_next)
        if cs is None:
            cs = dispatch_vehicle(create_vehicle_state(contract), contract)
        travel = float(env.dist_mat[inst_idx, cur_node, node])
        finish = float(v.committed_finish)
        # 冷链状态已在快照时点推进到 clock（coldchain_updated_time）；committed leg 只从
        # 当前时点再推进到 committed_finish，不能从 v.ready_time 重推整段（避免重复累计）。
        # arrival_time 只用于校验与 return 语义，这里取 finish（剩余时长 = finish - clock）。
        depart = float(v.coldchain_updated_time) if v.coldchain_updated_time is not None \
            else float(v.ready_time)
        depart = min(depart, finish)
        cs, _ = transition_segment(
            cs, depart_time=depart, arrival_time=finish, service_finish=finish,
            served_customer=node, active_zone_mask=active, contract=contract,
            order_quantity=float(env.demands[inst_idx, node]),
            order_temp_class=int(env.temp_class[inst_idx, node]),
            initial_quality=float(env.initial_quality[inst_idx, node]),
            segment_distance_units=travel)
        total_dist += travel
        wait_total += max(0.0, finish - depart)
        slack = float(env.tw_end[inst_idx, node]) - float(v.committed_arrive)
        min_slack = min(min_slack, slack)
        load += float(env.demands[inst_idx, node])
        max_load = max(max_load, load)
        n_customers += 1
        cur_node = node
        t = finish

    # 2. suffix（有序客户）
    for c in suffix:
        if cs is None:
            cs = dispatch_vehicle(create_vehicle_state(contract), contract)
        travel = float(env.dist_mat[inst_idx, cur_node, c])
        arrive = t + travel / speed
        wait = max(0.0, float(env.tw_start[inst_idx, c]) - arrive)
        service_start = max(arrive, float(env.tw_start[inst_idx, c]))
        finish = service_start + float(env.service_time[inst_idx, c])
        slack = float(env.tw_end[inst_idx, c]) - arrive
        cs, _ = transition_segment(
            cs, depart_time=t, arrival_time=arrive, service_finish=finish,
            served_customer=c, active_zone_mask=active, contract=contract,
            order_quantity=float(env.demands[inst_idx, c]),
            order_temp_class=int(env.temp_class[inst_idx, c]),
            initial_quality=float(env.initial_quality[inst_idx, c]),
            segment_distance_units=travel)
        total_dist += travel
        wait_total += wait
        load += float(env.demands[inst_idx, c])
        max_load = max(max_load, load)
        min_slack = min(min_slack, slack)
        n_customers += 1
        cur_node = c
        t = finish

    # 3. 返仓 / WAIT
    open_plan = False
    return_time = t
    if not suffix and anchor_node != 0 and has_future:
        # WAIT：无客户、在客户处、还有未来 reveal → 不返仓
        open_plan = True
    else:
        if cs is None:
            cs = dispatch_vehicle(create_vehicle_state(contract), contract)
        travel_back = float(env.dist_mat[inst_idx, cur_node, 0])
        arrive_back = t + travel_back / speed
        cs, _ = transition_segment(
            cs, depart_time=t, arrival_time=arrive_back, service_finish=arrive_back,
            served_customer=None, return_to_depot=True, active_zone_mask=active,
            contract=contract, segment_distance_units=travel_back)
        total_dist += travel_back
        return_time = arrive_back
        if min_slack == float('inf'):
            min_slack = float(env.tw_end[inst_idx, 0]) - return_time

    if min_slack == float('inf'):
        min_slack = horizon

    # 4. 冷链增量（相对当前状态 cs0）
    if cs0 is not None:
        d_q = total_quality_loss(cs, contract) - total_quality_loss(cs0, contract)
        d_e = cs.cumulative_energy_kwh - cs0.cumulative_energy_kwh
    else:
        d_q = total_quality_loss(cs, contract)
        d_e = cs.cumulative_energy_kwh
    d_d = total_dist

    # 当前 cargo 汇总（cs0 是当前状态；未派车则为空）
    cargo_count = 0
    cargo_age = 0.0
    cargo_quality = 1.0
    has_cargo = bool(cs0 is not None and getattr(cs0, 'cargo_manifest', ()))
    if has_cargo:
        lots = cs0.cargo_manifest
        cargo_count = len(lots)
        clock = float(v.coldchain_updated_time) if v.coldchain_updated_time is not None else t
        w = sum(lot.quantity for lot in lots) or 1.0
        cargo_age = sum(lot.quantity * max(0.0, clock - lot.pickup_finish_time) for lot in lots) / w
        cargo_quality = sum(lot.quantity * (lot.quality_remaining / max(lot.initial_quality, 1e-9))
                            for lot in lots) / w

    return {
        'valid': True, 'anchor_node': anchor_node, 'anchor_time': t, 'anchor_load': load,
        'is_committed': is_committed, 'n_customers': n_customers,
        'distance': total_dist, 'return_time': return_time, 'min_tw_slack': min_slack,
        'max_load': max_load, 'wait_time': wait_total, 'open_plan': open_plan,
        'd_quality': d_q, 'd_energy': d_e, 'd_distance': d_d,
        'has_cargo': has_cargo, 'cargo_age': cargo_age, 'cargo_quality': cargo_quality,
        'cargo_count': cargo_count,
    }


def project_plan(env, inst_idx, vehicles, plan, contract, has_future):
    """对 full-fleet plan 投影每辆车。返回 {vid: proj_dict}（不含 absent 车）。"""
    out = {}
    for vid, p in plan.items():
        if vid >= len(vehicles):
            continue
        out[int(vid)] = project_vehicle(env, inst_idx, vehicles[int(vid)], p, contract,
                                        has_future)
    return out


def proxy_cost(proj, objective):
    """把投影的增量 D/Q/E 合成代理 J（与 g_B 同一 J 单位）。"""
    return compute_coldchain_cost(proj['d_distance'], proj['d_quality'], proj['d_energy'],
                                  objective)
