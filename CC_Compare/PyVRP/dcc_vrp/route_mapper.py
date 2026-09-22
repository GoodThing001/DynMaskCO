"""PyVRP Solution → PlanProposal 映射（B2 步骤 3/7）。

关键约束：
  - 车辆身份通过 vehicle type 加入序索引映射（route.vehicle_type() → int idx），
    禁止按 solution route 顺序猜 vehicle ID；
  - 输出 proposal 必须包含全部 replan vehicle IDs（未使用车辆 → 空 suffix）；
  - 每个非空 suffix 在**原始浮点口径**下重放 certificate（TW/容量/返仓），
    PyVRP 整数模型可行但浮点 certificate 失败 → 拒绝（显式失败）；
  - 完整性：所有 pool 客户必须恰好出现在一条 route（PyVRP required 保证，
    此处防御性复核跨车重复与缺失）。
"""
from method_adapter import PlanProposal


class MappingError(RuntimeError):
    pass


def wait_or_close(view, v):
    """空 pool / 未使用车辆的 WAIT/CLOSE 规则（冻结候选）。

    WAIT iff：有 future reveal 且「现在从 anchor 返仓可行」
    （ready_time + travel(anchor→0) ≤ depot_tw_end）。
      - depot 等待：travel=0 → 恒可行 → WAIT（零成本：未派车无冷链状态）；
      - 客户处等待：每轮 reveal 都会重估（reveal 触发全队 replan）；
        最后一轮 reveal 后 has_future_reveal=False → 自动 CLOSE。
    CLOSE：无 future reveal / 现在返仓已不可行。

    不读取下一次 reveal 的具体时间（禁止未来信息）。hard gate 只能「发现」
    返仓失败、不能保证不发生失败；本规则作为冻结候选，必须在完整
    DEV-PROTO 9×32 上保持 100% return/service Gate，否则只能在 DEV-PROTO
    修订，不得等到 VAL/TEST。
    """
    if not view.has_future_reveal:
        return (0,)
    i_anchor = v.anchor_idx
    return_time = (float(v.ready_time)
                   + float(view.travel_mat[i_anchor][view.node_index(0)]))
    if return_time <= float(view.depot_tw_end) + 1e-6:
        return ()
    return (0,)


def certify_suffix_float(view, v, suffix_customers):
    """在原始浮点口径下重放一条 suffix（不含尾部 0）。返回 (ok, reason)。

    与 action_contract.certify_route 同语义：arrive > tw_end 判 TW 违例；
    容量按 total_capacity − 当前 load 再逐客户累加；返仓 deadline 用 depot_tw_end。
    """
    idx = {n: i for i, n in enumerate(view.node_ids)}
    cur = v.anchor_node_id
    t = v.ready_time
    load = float(v.load)
    for c in suffix_customers:
        ic = idx[c]
        travel = float(view.travel_mat[idx[cur]][ic])
        arrive = t + travel
        if arrive > float(view.tw_end[ic]) + 1e-6:
            return False, (f'tw: customer {c} arrive {arrive:.4f} > '
                           f'tw_end {view.tw_end[ic]:.4f}')
        start = max(arrive, float(view.tw_start[ic]))
        load += float(view.demands[ic])
        if load > float(view.capacity) + 1e-6:
            return False, f'capacity: customer {c} load {load:.4f} > {view.capacity}'
        t = start + float(view.service_time[ic])
        cur = c
    ret = t + float(view.travel_mat[idx[cur]][idx[0]])
    if ret > float(view.depot_tw_end) + 1e-6:
        return False, f'return: {ret:.4f} > depot_tw_end {view.depot_tw_end}'
    return True, None


def map_solution(problem, view, solution, runtime_s=0.0):
    """Solution → PlanProposal（vt 索引身份映射 + 浮点 certificate 复核）。

    deferral 语义（require_complete=False）：未覆盖客户 = deferred，留给后续事件
    （公共 ownership audit 记录 deferred 名单），不报错。
    严格模式（require_complete=True）：客户缺失 → MappingError（显式失败）。
    跨车重复 / 身份映射失败 / 浮点 certificate 失败 在任何模式下都显式失败。
    """
    suffixes = {vid: () for vid in view.replan_ids}
    seen_customers = set()
    missing = [problem.customer_id_by_client_idx[a.idx]
               for a in solution.unplanned()
               if a.idx in problem.customer_id_by_client_idx]
    if missing and getattr(problem, 'require_complete', False):
        raise MappingError(f'PyVRP 未覆盖全部客户: missing={missing}')

    vehicle_views = {v.vehicle_id: v for v in view.vehicles}

    for route in solution.routes():
        # 防御性拒绝：即使 adapter 已检查 result.is_feasible()，
        # 任何一条 infeasible route 都不得静默过滤（那是未声明的后处理）
        if not route.is_feasible():
            raise MappingError(
                'PyVRP solution 包含 infeasible route: '
                f'vehicle_type={route.vehicle_type()}, '
                f'excess_load={route.has_excess_load()}, '
                f'excess_distance={route.has_excess_distance()}, '
                f'time_warp={route.has_time_warp()}')
        vt_idx = route.vehicle_type()
        vid = problem.vid_by_vt_idx.get(vt_idx)
        if vid is None:
            raise MappingError(f'route 的 vehicle type 索引 {vt_idx} 无身份映射')
        visits = []
        for act in route:
            if act.is_client():
                node = problem.customer_id_by_client_idx.get(act.idx)
                if node is None:
                    raise MappingError(f'route 含非客户节点（client idx={act.idx}）')
                if node in seen_customers:
                    raise MappingError(f'跨车重复客户 {node}')
                seen_customers.add(node)
                visits.append(node)
        if suffixes[vid]:
            raise MappingError(f'车辆 {vid} 出现多条 route')
        suffixes[vid] = tuple(visits + [0]) if visits else wait_or_close(
            view, vehicle_views[vid])

    # 未出现在任何 route 的 replan 车辆：同样走 WAIT/CLOSE 规则
    for vid in view.replan_ids:
        if vid not in suffixes or suffixes[vid] == ():
            suffixes[vid] = wait_or_close(view, vehicle_views[vid])

    # ---- route / unplanned 精确分区（防御）----
    pool_set = set(problem.pool_customers)
    seen_set = set(seen_customers)
    unplanned_set = set(missing)
    if seen_set & unplanned_set:
        raise MappingError('客户同时出现在 route 与 unplanned: '
                           f'{sorted(seen_set & unplanned_set)}')
    if seen_set | unplanned_set != pool_set:
        raise MappingError('route/unplanned 未精确分割 pool: '
                           f'extra={(seen_set | unplanned_set) - pool_set} '
                           f'missing={pool_set - (seen_set | unplanned_set)}')

    # ---- 原始浮点 certificate 复核（整数可行 ≠ 浮点可行）----
    for vid, suffix in suffixes.items():
        v = vehicle_views[vid]
        custs = [c for c in suffix if c != 0]
        ok, reason = certify_suffix_float(view, v, custs)
        if not ok:
            raise MappingError(
                f'浮点 certificate 失败：车辆 {vid} suffix={custs} reason={reason}')

    return PlanProposal(
        suffixes=suffixes,
        model_input_customers=tuple(sorted(problem.pool_customers)),
        fallback_triggered=False,
        model_runtime_s=runtime_s,
    )
