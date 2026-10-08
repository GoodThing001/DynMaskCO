"""A-v1 外部强求解器接单决策适配（OR-Tools / PyVRP 共用接口）。

角色：外部经典滚动重优化方法在 A-v1 accept/reject 协议下的接单决策器
（论文「路由强度 vs 决策信息」隔离对照）。

决策规则（myopic feasibility + 最强路由）：
  每次揭示，对每个新订单 o（按揭示时间顺序）：
    pool = 已接受未服务 ∪ {o}（committed_next 已在途，排除在客户池外）；
    用求解器求「覆盖全部 pool」的静态 pickup-to-depot CVRPTW 最小距离计划
    （车辆锚点 = 当前位置/committed_next、锚点时间、锚点载荷、eff 容量、
      返仓截止 = depot TW end）；
    无完整解 / 求解超时 / 真实 C0 certify_plan 能耗 > B → 拒绝 o（保旧计划）；
    否则接受 o 并写回新计划。

与 SAA 臂的对齐：同一 VisibleSnapshot 信息边界（只读已揭示订单 + 车队状态，
不读未来订单/未揭示字段）、同一下游硬认证（certify_plan + A-v1 效用），
区别只在「接单可行性检查」用强求解器而非贪心插入。

整数化与建模语义镜像 CC_Compare 已冻结适配（OR-Tools-RH-D / PyVRP-RH-D）：
距离 round、travel/TW 起 ceil、TW 止/deadline/容量 floor、demand 整数化、
PATH_CHEAPEST_ARC + GLS + solution_limit、PyVRP seed=0 + MaxIterations。
本文件是 A-v1 新协议的新代码，不修改任何冻结文件。
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from run_exp_energy_c0 import C0ReserveReplanner, certify_plan

INT_SCALE = 1000
_MAX_INT = np.iinfo(np.int64).max

# OR-Tools 9.11.4210 状态码（CC_Compare/OR-Tools/dcc_vrp/ortools_adapter.py 实测约定）：
# 0 NOT_SOLVED / 1 SUCCESS / 2 PARTIAL / 3 FAIL / 4 FAIL_TIMEOUT / 5 INVALID /
# 6 INFEASIBLE / 7 OPTIMAL（证明最优，属成功）
_OT_SUCCESS_STATUSES = (1, 2, 7)


def ceil_int(x):
    return int(math.ceil(float(x) * INT_SCALE))


def floor_int(x):
    return int(math.floor(float(x) * INT_SCALE))


def round_int(x):
    return int(round(float(x) * INT_SCALE))


def _demand_int(x):
    """A-05d（2026-09-29）：需求**向上**取整到 ×1000 单位——求解器看到的装载 ≥ 真实装载，
    任何求解器可行计划（seen ≤ cap_int）在真实物理（transition _EPS=1e-9）下必可行。
    A-05c 的向下取整方向相反：求解器按 49.999 打包、真实装载可达 50.0+0.02（每单低估 ≤0.001），
    certify_plan 真实模拟必然越界。
    旧 1 单位整数 ceil（2.01→3）确有 20×3=60>50 的不公平拒绝问题（T13 原注释），
    但 ×1000 尺度 ceil 高估 ≤0.001/单，只造成边界 ±0.001 的保守拒绝，不构成不公平。"""
    return int(math.ceil(float(x) * INT_SCALE))


def _start_pin_int(load, capacity_int):
    """A-05e（2026-09-29，真正的 40/40 崩溃根因）：初始载重钉死值 = ceil_int(load) 并 clamp 到
    容量整数上界。旧代码 `math.ceil(load)` 少乘 INT_SCALE（32.1584 → 33 个单位 = 0.033 真实单位），
    求解器把在途车当成空车继续打包 → certify_plan 真实模拟越界崩溃。
    clamp：ceil_int(load) > cap_int（真实剩余空间 <0.001）时该车不可再装，钉死在 cap_int
    （钉 >cap_int 会令整个模型不可行 → 所有车全部拒绝）。"""
    return min(ceil_int(load), int(capacity_int))


class SolverAcceptReplanner(C0ReserveReplanner):
    """强求解器接单 replanner。继承持久预留/回放（plan()、export/restore）与
    _vehicle_serve_state 锚点语义；覆写 on_reveal 为「求解器完整覆盖可行性」接单。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 solver='ortools', time_limit=10.0, solution_limit=30,
                 pyvrp_max_iterations=1000):
        if solver not in ('ortools', 'pyvrp'):
            raise ValueError('solver must be ortools|pyvrp, got ' + str(solver))
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share)
        self.solver = solver
        self.time_limit = float(time_limit)
        self.solution_limit = int(solution_limit)
        self.pyvrp_max_iterations = int(pyvrp_max_iterations)
        self.timeouts = 0
        self.n_solves = 0
        self.n_fail = 0
        self.solve_time_s = 0.0
        self._reject_reasons = {}

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self.timeouts = 0
            self.n_solves = 0
            self.n_fail = 0
            self.solve_time_s = 0.0
            self._reject_reasons = {}

    # ------------------------------------------------------------------ #
    # 状态收集：车辆锚点（复用 C0ReserveReplanner 语义，与 SAA 臂同口径）
    # ------------------------------------------------------------------ #
    def _collect(self, env, inst_idx, clock, vehicles, pool):
        """返回 (sts, clients, depot_deadline)。

        sts[vid] = {anchor, anchor_time, load}：
          committed 车锚点 = committed_next（anchor_time=committed_finish，
          load 已含其 pickup）；ready = current_node/max(ready,clock)；
          idle = depot/clock。returning/closed 车不参与。
        clients = pool 排除所有 committed_next（在途，不可重排）。
        """
        sts = {}
        committed_next = set()
        for v in vehicles:
            pos = self._vehicle_serve_state(env, inst_idx, v, clock)
            if pos is None:
                continue
            cur, t, load = pos
            cc = v.coldchain_state
            if cc is not None and not cc.closed:
                # A-05b：货物清单 float64 精确和（float32 running-sum 低估 ~1e-4 会在执行期越界崩溃）
                load = float(sum(lot.quantity for lot in cc.cargo_manifest))
                if v.status == 'committed' and v.committed_next not in (None, 0):
                    load += float(env.demands[inst_idx, v.committed_next])
            if v.status == 'committed' and v.committed_next not in (None, 0):
                committed_next.add(int(v.committed_next))
            sts[v.vehicle_id] = dict(anchor=int(cur), anchor_time=float(t),
                                     load=float(load))
        clients = [int(o) for o in pool if int(o) not in committed_next]
        depot_deadline = float(env.tw_end[inst_idx, 0])
        return sts, clients, depot_deadline

    def _int_tw_empty(self, env, inst_idx, clients):
        """整数化后 TW 为空（ceil(start) > floor(end)）的客户：完整覆盖不可能。"""
        for o in clients:
            if ceil_int(env.tw_start[inst_idx, o]) > floor_int(env.tw_end[inst_idx, o]):
                return int(o)
        return None

    # ------------------------------------------------------------------ #
    # OR-Tools 后端（镜像冻结 OR-Tools-RH-D 的建模：virtual start / 容量钉死 /
    # dummy 终端 / service-first drop penalty）
    # ------------------------------------------------------------------ #
    def _solve_ortools(self, env, inst_idx, clock, vehicles, pool, deadline):
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        sts, clients, depot_deadline = self._collect(env, inst_idx, clock, vehicles, pool)
        vids = sorted(sts)
        V, C = len(vids), len(clients)
        if V == 0:
            return None, 'no_vehicle'
        if self._int_tw_empty(env, inst_idx, clients) is not None:
            return None, 'int_tw_empty'

        n_nodes = 1 + V + C + V
        coord_idx = {0: 0}                       # depot
        for k, vid in enumerate(vids):
            coord_idx[1 + k] = sts[vid]['anchor']          # virtual start
        client_node = {1 + V + k: o for k, o in enumerate(clients)}  # 模型节点 → 真实客户
        for k, o in enumerate(clients):
            coord_idx[1 + V + k] = o
        for k in range(V):
            coord_idx[1 + V + C + k] = 0                   # dummy（depot 坐标）

        D = env.dist_mat[inst_idx]
        inv_speed = 1.0 / float(env.tw_speed)
        dist = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        trav = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        for i in range(n_nodes):
            for j in range(n_nodes):
                d = float(D[coord_idx[i], coord_idx[j]])
                dist[i, j] = round_int(d)
                trav[i, j] = ceil_int(d * inv_speed)
        service = np.zeros(n_nodes, dtype=np.int64)
        for k, o in enumerate(clients):
            service[1 + V + k] = ceil_int(env.service_time[inst_idx, o])
        transit = trav + service[:, None]
        demand = np.zeros(n_nodes, dtype=np.int64)
        for k, o in enumerate(clients):
            demand[1 + V + k] = _demand_int(env.demands[inst_idx, o])

        manager = pywrapcp.RoutingIndexManager(
            n_nodes, V, list(range(1, 1 + V)), [0] * V)
        routing = pywrapcp.RoutingModel(manager)

        def dist_cb(i, j):
            return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

        def transit_cb(i, j):
            return int(transit[manager.IndexToNode(i)][manager.IndexToNode(j)])

        def demand_cb(i, j):
            return int(demand[manager.IndexToNode(j)])

        routing.SetArcCostEvaluatorOfAllVehicles(
            routing.RegisterTransitCallback(dist_cb))

        # 容量：start cumul 自由变量 → 逐车 SetValue 钉死初始载重
        # A-05d：capacity_int = floor((cap − 1e-4)×1000) = 49999；需求 ceil 保证 seen ≥ real，
        # 故 seen ≤ 49999 ⇒ real ≤ 49.999 < 50，执行期必可行。
        capacity_int = int(math.floor((float(self.capacity) - 1e-4) * INT_SCALE))   # A-05：×1000 + 保守余量 1e-4（与 R2 一致，防边界载重执行期崩溃）
        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterTransitCallback(demand_cb), 0,
            [capacity_int] * V, False, 'Capacity')
        capacity_dim = routing.GetDimensionOrDie('Capacity')
        for k, vid in enumerate(vids):
            capacity_dim.CumulVar(routing.Start(k)).SetValue(
                _start_pin_int(sts[vid]['load'], capacity_int))

        # 时间：global horizon = floor(depot deadline)；virtual start 单点钉死；
        # 客户 TW；dummy 全时段。不对 end depot 设 cumul（horizon 已约束 + 版本坑）。
        horizon = floor_int(depot_deadline)
        routing.AddDimension(routing.RegisterTransitCallback(transit_cb),
                             horizon, horizon, False, 'Time')
        td = routing.GetDimensionOrDie('Time')
        for k, vid in enumerate(vids):
            td.CumulVar(manager.NodeToIndex(1 + k)).SetRange(
                ceil_int(sts[vid]['anchor_time']), ceil_int(sts[vid]['anchor_time']))
        for k, o in enumerate(clients):
            td.CumulVar(manager.NodeToIndex(1 + V + k)).SetRange(
                ceil_int(env.tw_start[inst_idx, o]), floor_int(env.tw_end[inst_idx, o]))
        for k in range(V):
            td.CumulVar(manager.NodeToIndex(1 + V + C + k)).SetRange(0, horizon)

        # dummy：每车专属、后继必为 End（空路线 = start→dummy→end，弧成本 = 真实返仓）
        for k in range(V):
            dummy_idx = manager.NodeToIndex(1 + V + C + k)
            routing.SetAllowedVehiclesForIndex([k], dummy_idx)
            routing.solver().Add(routing.NextVar(dummy_idx) == routing.End(k))

        # service-first drop penalty（覆盖任意距离差）
        max_arc = max(int(dist.max()), 1)
        drop_penalty = (C + V) * max_arc + 1
        for k in range(C):
            routing.AddDisjunction([manager.NodeToIndex(1 + V + k)],
                                   int(drop_penalty))

        remaining = deadline - time.perf_counter()   # Q-03：单调绝对 deadline 剩余预算
        if remaining <= 0:
            return None, 'timeout'
        sp = pywrapcp.DefaultRoutingSearchParameters()
        sp.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        sp.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        sp.solution_limit = self.solution_limit
        sp.time_limit.FromMilliseconds(max(1, int(math.ceil(remaining * 1000))))
        sp.log_search = False
        solution = routing.SolveWithParameters(sp)
        status = routing.status()
        if solution is None or status not in _OT_SUCCESS_STATUSES:
            return None, 'timeout' if status == 4 else 'no_solution'

        # drop 检测：NextVar(idx) == idx（node_number 是模型节点号，不是真实客户 id）
        dropped = []
        for node_number, o in client_node.items():
            idx = manager.NodeToIndex(node_number)
            if solution.Value(routing.NextVar(idx)) == idx:
                dropped.append(o)
        if dropped:
            return None, 'dropped'

        plan = {}
        for k, vid in enumerate(vids):
            idx = routing.Start(k)
            seq = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node in client_node:
                    seq.append(client_node[node])
                idx = solution.Value(routing.NextVar(idx))
            if seq:
                plan[vid] = seq
        covered = [o for r in plan.values() for o in r]
        if sorted(covered) != sorted(clients):
            return None, 'coverage_mismatch'
        return plan, 'solved'

    # ------------------------------------------------------------------ #
    # PyVRP 后端（镜像冻结 PyVRP-RH-D 的建模：独立 start depot + vehicle type +
    # required 客户 + seed=0；0.11.3 API：add_client/add_depot 直接收坐标，
    # add_vehicle_type 收 initial_load——与 OR-Tools 臂的 start-pin 语义一致）
    # ------------------------------------------------------------------ #
    def _solve_pyvrp(self, env, inst_idx, clock, vehicles, pool, deadline):
        from pyvrp import Model
        from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria

        sts, clients, depot_deadline = self._collect(env, inst_idx, clock, vehicles, pool)
        vids = sorted(sts)
        if not vids:
            return None, 'no_vehicle'
        if self._int_tw_empty(env, inst_idx, clients) is not None:
            return None, 'int_tw_empty'

        X = env.coords[inst_idx]
        D = env.dist_mat[inst_idx]
        inv_speed = 1.0 / float(env.tw_speed)

        # A-05e：capacity 与 initial_load 与 OR-Tools 臂同一整数口径；
        # 需求 ceil（A-05d：seen ≥ real ⇒ 求解可行必物理可行）。
        capacity_int = int(math.floor((float(self.capacity) - 1e-4) * INT_SCALE))

        m = Model()
        end_depot = m.add_depot(round_int(X[0, 0]), round_int(X[0, 1]),
                                tw_early=0, tw_late=floor_int(depot_deadline))

        client_idx_of = {}
        client_locs = []
        for k, o in enumerate(clients):
            cl = m.add_client(round_int(X[o, 0]), round_int(X[o, 1]),
                              pickup=_demand_int(env.demands[inst_idx, o]),
                              service_duration=ceil_int(env.service_time[inst_idx, o]),
                              tw_early=ceil_int(env.tw_start[inst_idx, o]),
                              tw_late=floor_int(env.tw_end[inst_idx, o]),
                              required=True, prize=0)
            # 0.11.3：Route 迭代给出 ProblemData 位置索引，布局 = 全部 depot 在前
            # （1 end + V start）、客户随后；客户 k 的 data 索引 = (1 + V) + k。
            client_idx_of[(1 + len(vids)) + k] = o
            client_locs.append(cl)

        vid_by_vt = {}
        start_locs = []
        for k, vid in enumerate(vids):
            s = sts[vid]
            ia = s['anchor']
            start_depot = m.add_depot(round_int(X[ia, 0]), round_int(X[ia, 1]),
                                      tw_early=ceil_int(s['anchor_time']),
                                      tw_late=_MAX_INT)
            vt = m.add_vehicle_type(1, capacity=capacity_int, start_depot=start_depot,
                                    end_depot=end_depot, fixed_cost=0, tw_early=0,
                                    tw_late=floor_int(depot_deadline),
                                    unit_distance_cost=1, unit_duration_cost=0,
                                    initial_load=_start_pin_int(s['load'], capacity_int),
                                    name='vehicle_%d' % vid)
            vid_by_vt[k] = vid      # VehicleType 不可哈希（0.11.3），按创建序 k 存 vid
            start_locs.append(start_depot)

        # 边：end depot / clients / start depots 全对（自环 0/0）
        all_locs = [end_depot] + client_locs + start_locs
        coord_of = [0] + [int(o) for o in clients] + [int(sts[vid]['anchor'])
                                                       for vid in vids]
        for a_i, a in enumerate(all_locs):
            for b_i, b in enumerate(all_locs):
                if a_i == b_i:
                    m.add_edge(a, b, 0, 0)
                    continue
                d = float(D[coord_of[a_i], coord_of[b_i]])
                m.add_edge(a, b, round_int(d), ceil_int(d * inv_speed))

        remaining = deadline - time.perf_counter()   # Q-03：单调绝对 deadline 剩余预算
        if remaining <= 0:
            return None, 'timeout'
        stop = MultipleCriteria([MaxIterations(self.pyvrp_max_iterations),
                                 MaxRuntime(max(remaining, 0.05))])
        res = m.solve(stop=stop, seed=0)
        if res is None or res.best is None or not res.best.is_feasible():
            return None, 'infeasible'

        plan = {}
        covered = []
        for route in res.best.routes():
            vt = route.vehicle_type()          # 0.11.3：返回 vehicle type 的 int 索引（创建序）
            if not isinstance(vt, int) or vt < 0 or vt >= len(vid_by_vt):
                return None, 'unexpected_vehicle_type'
            vid = vid_by_vt[vt]
            seq = [client_idx_of[a] for a in route if a in client_idx_of]
            if seq:
                plan[vid] = seq
            covered.extend(seq)
        if sorted(covered) != sorted(clients):
            return None, 'coverage_mismatch'
        return plan, 'solved'

    # ------------------------------------------------------------------ #
    # 在线决策
    # ------------------------------------------------------------------ #
    def _solve(self, env, inst_idx, clock, vehicles, served_mask, pool, deadline):
        self.n_solves += 1
        t_s = time.perf_counter()
        try:
            if self.solver == 'ortools':
                plan, reason = self._solve_ortools(env, inst_idx, clock, vehicles,
                                                   pool, deadline)
            else:
                plan, reason = self._solve_pyvrp(env, inst_idx, clock, vehicles,
                                                 pool, deadline)
        except Exception as e:  # noqa: BLE001 —— 记录异常类型与信息，拒绝而非中断
            self.n_fail += 1
            if self.n_fail <= 3:
                import traceback
                print('SOLVER_EXC[%s #%d]:' % (self.solver, self.n_fail),
                      traceback.format_exc(limit=8), file=sys.stderr, flush=True)
            return None, 'exception:%s:%r' % (type(e).__name__, e)
        self.solve_time_s += time.perf_counter() - t_s
        return plan, reason

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = {int(i) for i in visible_ids if not served_mask[i]
                   and int(i) not in reserved and int(i) not in self._accepted
                   and int(i) not in self._rejected}
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            # Q-03/A-02（2026-09-29 核查修复）：单调 perf_counter + 绝对 deadline，
            # 构造/求解/解码/认证全计入；认证通过后提交前**再次复查**——越限 = 超时
            # （保旧计划 + 拒单 + 计次）。旧实现只把剩余时间传给后端、返回后不再检查，
            # 注入探针（限额 0.01s、认证结束 0.06s）证明存在超时漏报路径。
            deadline = time.perf_counter() + self.time_limit
            pool = sorted(set(int(x) for x in self._accepted
                              if not served_mask[x]) | {int(o)})
            saved_plan = {k: list(v) for k, v in self._plan.items()}
            plan, reason = self._solve(env, inst_idx, clock, vehicles, served_mask,
                                       pool, deadline)
            if plan is None:
                self._plan = saved_plan
                self._rejected.add(o)
                if reason == 'timeout':
                    self.timeouts += 1
                self._reject_reasons[reason] = self._reject_reasons.get(reason, 0) + 1
                continue
            ok, _energy = certify_plan(env, inst_idx, clock, vehicles, served_mask,
                                       plan, self.contract, self.budget)
            if time.perf_counter() > deadline:   # Q-03：认证后再查（最后认证越限不得接单）
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                self._reject_reasons['timeout'] = self._reject_reasons.get('timeout', 0) + 1
                continue
            if not ok:
                self._plan = saved_plan
                self._rejected.add(o)
                self._reject_reasons['certify_budget'] = \
                    self._reject_reasons.get('certify_budget', 0) + 1
                continue
            self._plan = plan
            self._accepted.add(o)
