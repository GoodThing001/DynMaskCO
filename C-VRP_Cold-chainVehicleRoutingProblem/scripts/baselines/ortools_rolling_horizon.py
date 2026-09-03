"""
P0-2: OR-Tools Rolling Horizon —— Replanner 接入共享 strict-online env（P0-OR 修复）。

2026-08-28 修复：
  - 旧版每 event 把整条 sub_route 执行完（非严格 event-driven），导致 13.53 不能作 reference。
  - 旧版 orig==0 只 load=0 continue，漏算 return depot 距离/时间（depot bug）。
  现在 OR-Tools 只负责 replan（给定当前状态，为每辆 idle/ready 车解出下一步 suffix），
  事件循环 / commitment / 车辆时间推进 / execution trace 全部由 strict_online_env.py 负责。

用法:
    python baselines/ortools_rolling_horizon.py \
        --data dcc_50_r1_edod05_val.npz --capacity 50 \
        --time_limit_ms 500 --num_vehicles 25 --num_instances 16
"""

import sys, os, argparse, time
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_CVRPTW_SCRIPTS = os.path.join(_CVRPTW, 'scripts')
sys.path.insert(0, _CVRPTW_SCRIPTS)
sys.path.insert(0, os.path.join(_CVRPTW_SCRIPTS, 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW_SCRIPTS, 'evaluation'))

from strict_online_env import StrictOnlineEnv, Replanner
from authoritative_evaluator import evaluate_execution_trace

SCALE = 1000


def solve_vrptw_from_starts(coords, demands, tw_start, tw_end, service_time,
                            capacity, num_vehicles, time_limit_ms,
                            start_positions, start_times, start_loads):
    """
    OR-Tools 多起始点 VRPTW（含载重连续性）。

    start_loads: 每辆车的「已卸货量」（容量维度的初始累积值）。
    start_times: 每辆车的起始时间（已乘 SCALE 的 int）。
    返回 (feasible, routes)：routes 是完整节点序列 [start, n1, ..., depot]。
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    N = len(coords)
    dist_float = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(axis=-1))
    dist_int = (dist_float * SCALE).astype(int)

    manager = pywrapcp.RoutingIndexManager(
        N, num_vehicles,
        [max(0, int(p)) for p in start_positions],
        [0] * num_vehicles)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return int(dist_int[i, j])

    def time_cb(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return int(dist_int[i, j] + service_time[i])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def demand_cb(from_idx):
        node = manager.IndexToNode(from_idx)
        if node != 0:
            for v in range(num_vehicles):
                if node == start_positions[v]:
                    return int(start_loads[v])
        return int(demands[node])

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_cb),
        0, [capacity] * num_vehicles, False, "Capacity")

    horizon = int(tw_end.max())
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # start-only 节点（车辆当前位置的已服务客户）不再施加 customer TW constraint。
    # 根因（CP Solver fail）：已服务客户的 original TW（tw_end 可能 < ready_time）与
    # 车辆 start cumul 冲突，domain 交集为空 → CP Solver fail（导师 P0-CTRL 第 4/8 点）。
    start_node_set = set(int(p) for p in start_positions) - {0}

    # FAIL_STAGE 诊断①：customer TW ranges（start-only 节点已跳过）
    try:
        time_dim.CumulVar(manager.NodeToIndex(0)).SetRange(int(tw_start[0]), horizon)
        for node in range(1, N):
            if node in start_node_set:
                continue  # start-only 节点（已服务客户）不再是待服务客户，不施加 TW
            time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(
                int(tw_start[node]), int(tw_end[node]))
    except Exception as e:
        import sys
        sys.stderr.write(f"  [FAIL_STAGE=customer_tw] {e} N={N} horizon={horizon}\n")
        for v in range(num_vehicles):
            sp = int(start_positions[v])
            sys.stderr.write(f"    start[{v}]: node={sp} start_time={int(start_times[v])} "
                             f"orig_tw=[{int(tw_start[sp])},{int(tw_end[sp])}] "
                             f"is_served={sp != 0}\n")
        sys.stderr.flush()
        return False, []

    # FAIL_STAGE 诊断②：vehicle start ranges（start_time 与 horizon / 历史 TW 的冲突，第二嫌疑）
    try:
        for v in range(num_vehicles):
            st = max(0, min(int(start_times[v]), horizon))
            time_dim.CumulVar(routing.Start(v)).SetRange(st, horizon)
    except Exception as e:
        import sys
        sys.stderr.write(f"  [FAIL_STAGE=vehicle_start] {e} num_veh={num_vehicles} "
                         f"horizon={horizon}\n")
        for v in range(num_vehicles):
            sp = int(start_positions[v])
            sys.stderr.write(f"    start[{v}]: node={sp} start_time={int(start_times[v])} "
                             f"orig_tw=[{int(tw_start[sp])},{int(tw_end[sp])}] "
                             f"is_served={sp != 0}\n")
        sys.stderr.flush()
        return False, []

    # P0-OR-hard：允许 drop 时间窗已关/无法服务的客户（disjunction）。
    # 否则任一客户 tw_end < clock 会让整个 VRPTW 判 infeasible，OR-Tools 返回 no solution，
    # 该 event 的「所有」客户（不只是时间窗关的）全被放弃 —— 这是 complete=0 的根因。
    # P0-CTRL：lexicographic-safe penalty —— 严格大于任何距离上界，保证「少 drop 一个客户」
    # 永远优先于「任何距离改善」（service-first），且不用 1e9 制造极端 objective scale。
    max_arc = int(dist_int.max())
    distance_ub = (N + num_vehicles) * max_arc
    PENALTY = 10 * distance_ub + 1
    for node in range(1, N):
        if node in start_node_set:
            continue  # 车辆当前位置不是待服务客户，不加 disjunction
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromMilliseconds(time_limit_ms)
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.log_search = False
    # 注：random_seed 字段因 OR-Tools 版本而异（部分版本无此字段会抛异常），
    # 且 plan persistence 已消除碎片化主因，故不设置。如需 reproducibility 单独处理。

    # FAIL_STAGE 诊断③：solve 本身
    try:
        sol = routing.SolveWithParameters(params)
    except Exception as e:
        import sys
        sys.stderr.write(f"  [FAIL_STAGE=solve] {e}\n")
        sys.stderr.flush()
        return False, []
    if not sol:
        return False, []

    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))  # end = depot 0
        routes.append(route)
    return True, routes


class ORToolsReplanner(Replanner):
    """OR-Tools 作为 Replanner：每 event 对 idle/ready 车解 VRPTW，返回下一步 suffix。"""

    def __init__(self, capacity, num_vehicles, time_limit_ms):
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.time_limit_ms = time_limit_ms

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        # P0-B：committed 车的下一个客户不可反悔
        reserved = env.get_reserved_customers(vehicles)
        # remaining = 可见未服务 - reserved
        remaining = [int(i) for i in visible_ids
                     if not served_mask[i] and int(i) not in reserved]
        # idle/ready 车（P0-CTRL：只重规划 needs_replan 的，其余保留旧 plan）
        active = [v for v in vehicles if v.status in ('idle', 'ready')
                  and (replan_ids is None or v.vehicle_id in replan_ids)]
        if not remaining or not active:
            return

        coords = env.coords[inst_idx]
        demands = env.demands[inst_idx].astype(int)
        tw_start = (env.tw_start[inst_idx] * SCALE).astype(int)
        tw_end = (env.tw_end[inst_idx] * SCALE).astype(int)
        service_time = (env.service_time[inst_idx] * SCALE).astype(int)

        # 子图：depot + remaining + idle/ready 车当前位置（作为多起始点）
        sub_nodes = [0] + sorted(remaining)
        start_sub = []
        start_times = []
        start_loads = []
        for v in active:
            pos = int(v.current_node)
            if pos != 0 and pos not in sub_nodes:
                sub_nodes.append(pos)
            start_sub.append(sub_nodes.index(pos))
            start_times.append(int(round(v.ready_time * SCALE)))
            start_loads.append(float(v.current_load))

        sub_coords = coords[sub_nodes]
        sub_demands = demands[sub_nodes]
        sub_tw_s = tw_start[sub_nodes]
        sub_tw_e = tw_end[sub_nodes]
        sub_svc = service_time[sub_nodes].copy()

        # P0-D：作为车辆起始位置、且已服务完成的 customer，其 service time 必须清零
        #（v.ready_time 已含它的 service，time_cb 再加会重复计算一次）。
        for v, sidx in zip(active, start_sub):
            if v.current_node != 0:
                sub_svc[sidx] = 0

        # 防御：start_times 不得早于 0 或晚于子图 horizon（否则 OR-Tools
        # time_dim.CumulVar(Start).SetRange(start > end) 会 C++ 段错误）。
        # v.ready_time 是服务完成时刻，可能略超 tw_end.max()（到达截止）。
        sub_horizon = int(sub_tw_e.max())
        start_times = [max(0, min(t, sub_horizon)) for t in start_times]

        # P1：pad 到完整车队 num_vehicles，避免 num_vehicles=1（单车起始于客户）
        # 触发 OR-Tools RoutingIndexManager 段错误。parked 车（非 active）放 depot、
        # start_time=sub_horizon-1（严格小于 horizon，避免 cumul 卡在容量上界导致
        # 整体 infeasible / SetRange 边界失败），solve 后只应用 active 车的 route。
        n_park = max(0, self.num_vehicles - len(active))
        for _ in range(n_park):
            start_sub.append(0)
            start_times.append(max(0, sub_horizon - 1))
            start_loads.append(0.0)

        try:
            feas, sub_routes = solve_vrptw_from_starts(
                sub_coords, sub_demands, sub_tw_s, sub_tw_e, sub_svc,
                self.capacity, self.num_vehicles, self.time_limit_ms,
                start_sub, start_times, start_loads)
        except Exception as e:
            import sys
            # 诊断：CP Solver fail 时打印完整状态，一次性定位根因（不刷屏，只异常时触发）
            sys.stderr.write(
                f"  [solve-exception] {e} | clock={clock:.3f} N={len(sub_nodes)} "
                f"active={len(active)} start_pos={start_sub[:5]} "
                f"start_t={start_times[:5]} load={start_loads[:5]} "
                f"tw_e_range=[{int(sub_tw_e.min())},{int(sub_tw_e.max())}] "
                f"remaining_tw_e=[{min((sub_tw_e[i] for i in range(1, len(sub_nodes))), default=-1)},"
                f"{max((sub_tw_e[i] for i in range(1, len(sub_nodes))), default=-1)}]\n")
            sys.stderr.flush()
            feas, sub_routes = False, []
        if not feas:
            # 兜底：让 ready@customer 的车返回 depot，避免 stuck（disjunction 后应极少触发）
            for v in active:
                if v.status == 'ready' and v.current_node != 0:
                    v.mutable_suffix = [0]
            return

        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        # 把 OR-Tools 解映射回 suffix（去掉 start 节点）
        for idx, v in enumerate(active):
            if idx >= len(sub_routes):
                break
            sub_route = sub_routes[idx]  # [start_idx, n1, ..., depot_idx]
            if len(sub_route) < 2:
                continue
            orig_route = [int(sub_nodes[k]) for k in sub_route]
            suffix = orig_route[1:]  # [n1, n2, ..., 0]
            # P0-CTRL-4：WAIT parity —— 无客户可服务且未来有 reveal 时等待，不返回 depot
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix


def main():
    parser = argparse.ArgumentParser(
        description='P0-2: OR-Tools Rolling Horizon (shared strict-online env)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--time_limit_ms', type=int, default=500)
    parser.add_argument('--num_instances', type=int, default=16)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    N = min(args.num_instances, dataset['coords'].shape[0])

    print(f"=== OR-Tools Rolling Horizon (shared strict-online env) ===")
    print(f"  Data: {args.data} ({N} instances)")
    print(f"  Capacity={args.capacity}, Vehicles={args.num_vehicles}, "
          f"Budget={args.time_limit_ms}ms/event")

    replanner = ORToolsReplanner(args.capacity, args.num_vehicles, args.time_limit_ms)
    env = StrictOnlineEnv(dataset, args.capacity, tw_speed=1.0,
                          num_vehicles=args.num_vehicles, replanner=replanner)

    results = []
    t0 = time.time()
    total_replans = 0
    for i in range(N):
        traces, served_mask = env.run(i)
        total_replans += env.replan_count
        m = evaluate_execution_trace(
            traces, dataset['coords'][i], dataset['tw_start'][i], dataset['tw_end'][i],
            dataset.get('service_time', np.zeros_like(dataset['tw_start']))[i],
            dataset['demands'][i], args.capacity, speed=1.0)
        results.append(m)
        if (i + 1) % 4 == 0:
            comp = sum(1 for x in results if x['complete'])
            avg_c = np.mean([x['distance_cost'] for x in results])
            print(f"  {i+1}/{N} | complete={comp}/{i+1} | avg_cost={avg_c:.2f} | {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n--- OR-Tools-RH Results ---")
    print(f"  complete:  {sum(1 for x in results if x['complete'])}/{N}")
    print(f"  Avg cost:  {np.mean([x['distance_cost'] for x in results]):.2f}")
    print(f"  Avg TW viol: {np.mean([x['tw_late_count'] for x in results]):.2f}")
    print(f"  Avg unserved: {np.mean([x['n_unserved'] for x in results]):.2f}")
    print(f"  Avg duplicate: {np.mean([x['n_duplicate'] for x in results]):.2f}")
    print(f"  Avg vehicles: {np.mean([x['vehicle_count'] for x in results]):.1f}")
    print(f"  Avg replans/episode: {total_replans / max(1, N):.1f}")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
