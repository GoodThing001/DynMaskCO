"""OR-Tools Solution → PlanProposal 映射（OR4）。

  - 每辆 replan 车沿 NextVar 链取客户序列（virtual start 不入 suffix）；
  - 客户只来自 pool、无同车/跨车重复；planned ∪ dropped == pool（精确分区）；
  - 每条非空 suffix 在**原始浮点口径**重放 certificate（TW/容量/返仓）；
  - 空路线走冻结 WAIT/CLOSE 规则（与 PyVRP 同规则；不读取下次 reveal 时间）。
"""
from method_adapter import PlanProposal
from problem_builder import round_int, empty_suffix_policy


class MappingError(RuntimeError):
    def __init__(self, reason, code='MAPPING_ERROR'):
        super().__init__(reason)
        self.code = code
        self.reason = reason


def wait_or_close(view, v):
    """空 pool / 未规划车辆的 WAIT/CLOSE 规则——委托 problem_builder 的
    统一 empty_suffix_policy（builder 与 mapper 同一实现，禁止漂移）。"""
    return empty_suffix_policy(view, v)


def certify_suffix_float(view, v, suffix_customers):
    """原始浮点口径重放（与 action_contract.certify_route 同语义）。"""
    idx = {n: i for i, n in enumerate(view.node_ids)}
    cur = v.anchor_node_id
    t = v.ready_time
    load = float(v.load)
    for c in suffix_customers:
        ic = idx[c]
        arrive = t + float(view.travel_mat[idx[cur]][ic])
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


def map_solution(problem, view, solution, runtime_s=0.0, solver_status='SOLVED',
                 dropped_customers=()):
    """Solution → PlanProposal。planned ∪ dropped 必须精确等于 pool。"""
    suffixes = {vid: () for vid in view.replan_ids}
    seen = set()

    for v_idx, vid in sorted(problem.vid_by_vehicle_index.items()):
        start = problem.routing.Start(v_idx)
        node = int(solution.Value(problem.routing.NextVar(start)))
        visits = []
        while not problem.routing.IsEnd(node):
            node_number = problem.manager.IndexToNode(node)
            cust = problem.customer_id_by_node_index.get(node_number)
            if cust is None:
                raise MappingError(
                    f'route 含非客户节点 node={node_number}'
                    f'（virtual start 泄漏进 route）')
            if cust == -1:
                # dummy 终端审计（OR4.1 P0-1）：dummy 的后继必须恰为该车 End
                nxt = int(solution.Value(problem.routing.NextVar(node)))
                if nxt != problem.routing.End(v_idx):
                    raise MappingError(
                        f'dummy 不是终端节点（后继={nxt} != End({v_idx})），'
                        f'求解路线与写回路线不一致')
                node = nxt
                continue
            if cust in seen:
                raise MappingError(f'跨车重复客户 {cust}')
            seen.add(cust)
            visits.append(cust)
            node = int(solution.Value(problem.routing.NextVar(node)))
        end = problem.routing.End(v_idx)
        if node != end:
            raise MappingError(f'车辆 {vid} 的 route 未终止于 end depot')
        if problem.manager.IndexToNode(node) != 0:
            raise MappingError(f'车辆 {vid} 的 end 不是真实 depot')
        suffixes[vid] = tuple(visits + [0]) if visits else wait_or_close(
            view, next(x for x in view.vehicles if x.vehicle_id == vid))

    # ---- 精确分区：planned ∪ dropped == pool ----
    pool_set = set(problem.pool_customers)
    dropped_set = set(int(c) for c in dropped_customers)
    if seen & dropped_set:
        raise MappingError(f'客户同时 planned 与 dropped: {sorted(seen & dropped_set)}')
    if (seen | dropped_set) != pool_set:
        raise MappingError('planned/dropped 未精确分割 pool: '
                           f'missing={sorted(pool_set - (seen | dropped_set))} '
                           f'extra={sorted((seen | dropped_set) - pool_set)}')

    # ---- 浮点 certificate ----
    vehicle_views = {v.vehicle_id: v for v in view.vehicles}
    for vid, suffix in suffixes.items():
        v = vehicle_views[vid]
        custs = [c for c in suffix if c != 0]
        ok, reason = certify_suffix_float(view, v, custs)
        if not ok:
            raise MappingError(
                f'浮点 certificate 失败：车辆 {vid} suffix={custs} reason={reason}',
                code='FLOAT_CERTIFICATE_FAIL')

    # ---- 原始整数目标 vs 映射路线重算目标一致性（OR4.1 P0-1）----
    # objective = 距离和（round_int 弧长）+ drop_penalty × n_dropped
    idx = {n: i for i, n in enumerate(view.node_ids)}
    mapped_dist = 0
    for vid, suffix in suffixes.items():
        v = vehicle_views[vid]
        if suffix == ():
            continue            # WAIT：原地等待，目标贡献 0（模型空弧已置 0）
        cur = v.anchor_idx
        for c in suffix:
            if c == 0:
                break
            mapped_dist += round_int(float(view.dist_mat[cur][idx[c]]))
            cur = idx[c]
        mapped_dist += round_int(float(view.dist_mat[cur][idx[0]]))
    penalty = int(problem.scale_meta.get('drop_penalty', 0))
    # forced-dropped（INTEGERIZED_TW_EMPTY）未进模型、未付 penalty——
    # 目标只对求解器级 dropped 计惩罚
    n_solver_dropped = len(dropped_set) - len(
        getattr(problem, 'forced_dropped', []))
    expected_obj = mapped_dist + penalty * n_solver_dropped
    if expected_obj != int(solution.ObjectiveValue()):
        raise MappingError(
            f'原始目标与映射路线目标不一致: mapped={expected_obj} '
            f'solver={solution.ObjectiveValue()}',
            code='OBJECTIVE_MISMATCH')

    return PlanProposal(
        suffixes=suffixes,
        model_input_customers=tuple(sorted(problem.pool_customers)),
        fallback_triggered=False,
        model_runtime_s=runtime_s,
        solve_meta=dict(problem.scale_meta, solver_status=solver_status,
                        n_dropped=len(dropped_set)),
    )
