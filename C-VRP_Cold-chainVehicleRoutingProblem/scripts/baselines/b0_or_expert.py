"""
B0 Teacher-Value Diagnostic —— OR-fixed vs OR-joint 求解器。

B0 回答：OR-Tools 剩余 17.6% gap 中，多少来自「单车 suffix sequencing」、多少来自「fleet
allocation」？三层 comparator（导师 §9）：

  NN           —— 当前 NN 贪心 incumbent（GreedyReplanner('nn')）
  OR-fixed     —— 冻结 customer→vehicle 归属，OR-Tools 只优化同车 suffix sequencing（本文件）
  OR-joint     —— 释放 mutable ownership，OR-Tools 跨车重分配（复用 ortools_rolling_horizon）

本文件提供：
  - solve_or_fixed()：单车辆 TSPTW（frozen assignment，只排同一辆车的客户顺序）
  - ORFixedReplanner：NN 贪心决定 assignment + 单车 TSPTW 优化 sequencing 的 Replanner

注意：OR-Tools 是 CPU 求解，不抢 GPU；可在全量 R1.7-4 运行期间并行跑 B0 的 OR-fixed/OR-joint。
"""

import numpy as np
import sys, os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from strict_online_env import Replanner, GreedyReplanner

SCALE = 1000


def solve_or_fixed(coords, demands, tw_start, tw_end, service_time, capacity,
                   time_limit_ms, current_node, ready_time, current_load,
                   assigned_customers, dist_mat=None, tw_speed=1.0):
    """单车辆 TSPTW：优化一辆车 assigned_customers 的 suffix sequencing（frozen assignment）。

    P0-R1 修复合同（02 手册 §P0-R1）：
      feasible=True 当且仅当 returned_customer_multiset == assigned_customer_multiset，
      且无重复，且整条 route 的 TW / capacity / return 全可行。

    k>8 冻结策略（三选一中的第 3 项）：无 no-drop 求解器 → 显式 unavailable，
    绝不返回部分 route（禁止 partial success）。k<=8 精确穷举排列（每辆车 assigned
    客户通常 2-5 个，阶乘可解）。无 OR-Tools 依赖（规避 num_vehicles=1 的 segfault）。
    time_limit_ms 仅作兼容参数（穷举不需要）。

    距离口径：dist_mat 提供时用 dist_mat（与 authoritative_evaluator / env 完全一致，含
    asymmetric 网络）；否则回退 Euclidean。travel time = dist / tw_speed，cost = dist。
    当前标准 DCC 数据为对称 Euclidean + speed=1，两种口径等价。

    返回 (feasible, route, status)：
      feasible=True  → route 完整覆盖 assigned 集合；status ∈ {'empty', 'optimal'}。
      feasible=False → route=None；status ∈ {'unavailable_k_over8', 'unavailable_infeasible'}。
    """
    import itertools
    _c = np.asarray(coords)
    _tw_s = np.asarray(tw_start)
    _tw_e = np.asarray(tw_end)
    _sv = np.asarray(service_time)
    _dm = np.asarray(demands)

    def _dist(i, j):
        if dist_mat is not None:
            return float(dist_mat[int(i), int(j)])
        return float(np.linalg.norm(_c[int(i)] - _c[int(j)]))

    custs = sorted(int(c) for c in assigned_customers if int(c) != 0)
    if not custs:
        # 空分配：只验证返回 depot 的 TW（ready_time + travel(current_node,0) <= tw_end[0]）。
        arrive0 = float(ready_time) + _dist(int(current_node), 0) / tw_speed
        if arrive0 > float(_tw_e[0]) + 1e-6:
            return False, None, 'unavailable_infeasible'
        return True, [int(current_node), 0], 'empty'

    def _cost_and_feasible(perm):
        # 与 authoritative_evaluator 的 TW 语义一致：先判 arrival > tw_end，
        # 再等 tw_start（waiting），再累加 service_time。全可行才返回 cost，否则 None。
        cur = int(current_node)
        t = float(ready_time)
        load = float(current_load)
        c = 0.0
        for j in perm:
            arrive = t + _dist(cur, j) / tw_speed
            if arrive > float(_tw_e[j]) + 1e-6:
                return None
            start = max(arrive, float(_tw_s[j]))
            load += float(_dm[j])
            if load > capacity + 1e-6:
                return None
            c += _dist(cur, j)
            t = start + float(_sv[j])
            cur = j
        arrive0 = t + _dist(cur, 0) / tw_speed
        if arrive0 > float(_tw_e[0]) + 1e-6:
            return None
        c += _dist(cur, 0)
        return c

    if len(custs) > 8:
        return False, None, 'unavailable_k_over8'

    best = None
    best_cost = float('inf')
    for perm in itertools.permutations(custs):
        c = _cost_and_feasible(perm)
        if c is not None and c < best_cost:
            best_cost = c
            best = perm
    if best is None:
        return False, None, 'unavailable_infeasible'

    route = [int(current_node)] + list(best) + [0]
    return True, route, 'optimal'


class ORFixedReplanner(GreedyReplanner):
    """OR-fixed：NN 贪心决定 assignment，OR-Tools 单车 TSPTW 只优化 sequencing。

    与 OR-joint（ortools_rolling_horizon.ORToolsReplanner）的区别：本 replanner 不跨车重分配客户，
    只在 NN 贪心分配的客户集内优化每辆车的服务顺序。

    P0-R1：solve_or_fixed 返回 (feasible, route, status)；unavailable 时保持 greedy（绝不 partial）。
    solve_status_log 记录每次求解的 assigned/returned set 与 status（per-instance，供 coverage 报告）。
    """

    def __init__(self, capacity, time_limit_ms):
        super().__init__('nn')
        self.capacity = capacity
        self.time_limit_ms = time_limit_ms
        self.solve_status_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        # 1. NN 贪心分配（决定每辆车 claim 哪些客户）
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
        greedy_suffix = {}
        for v in vehicles:
            if v.status in ('idle', 'ready'):
                greedy_suffix[v.vehicle_id] = list(v.mutable_suffix)

        has_future = env.has_future_reveal(inst_idx, clock, served_mask)

        # 2. 每辆车用单车 TSPTW 重新 sequencing（frozen assignment）
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = [o for o in greedy_suffix.get(v.vehicle_id, []) if o != 0]
            if not assigned:
                continue  # 无客户，保持 greedy（[0] 或 []）
            feasible, route, status = solve_or_fixed(
                env.coords[inst_idx], env.demands[inst_idx], env.tw_start[inst_idx],
                env.tw_end[inst_idx], env.service_time[inst_idx], env.capacity,
                self.time_limit_ms, int(v.current_node), v.ready_time, v.current_load,
                assigned, dist_mat=env.dist_mat[inst_idx], tw_speed=env.tw_speed)
            if not feasible:
                # unavailable → 保持 greedy（客户仍被 greedy 服务，绝不 partial），显式记录
                self.solve_status_log.append({
                    'inst_idx': int(inst_idx), 'vehicle_id': int(v.vehicle_id),
                    'k': len(assigned), 'status': status,
                    'assigned_set': tuple(sorted(assigned)),
                    'returned_set': tuple(sorted(assigned)),
                })
                continue
            returned = tuple(sorted(int(n) for n in route
                                    if n not in (int(v.current_node), 0)))
            self.solve_status_log.append({
                'inst_idx': int(inst_idx), 'vehicle_id': int(v.vehicle_id),
                'k': len(assigned), 'status': status,
                'assigned_set': tuple(sorted(assigned)), 'returned_set': returned,
            })
            suffix = [n for n in route if n != int(v.current_node)]
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []  # WAIT（与 GreedyReplanner 一致）
            else:
                v.mutable_suffix = suffix
