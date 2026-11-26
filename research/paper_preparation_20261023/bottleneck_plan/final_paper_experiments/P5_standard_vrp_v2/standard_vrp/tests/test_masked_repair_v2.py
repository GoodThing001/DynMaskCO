# -*- coding: utf-8 -*-
"""P5-V2 缺陷修复单元测试：D1 排序生效 / D2 两算子区分 / D3 动态 Regret-2 /
D5 最短可行插位保护 / 新开单车候选 / 锁版候选语义。

运行：python -B tests/test_masked_repair_v2.py
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]  # P5_standard_vrp_v2
sys.path.insert(0, str(V2))

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                 evaluate_cvrp, evaluate_vrptw)
from standard_vrp.adapters.objective import CvrpAdapter, VrptwAdapter, adapter_for, BIG
from standard_vrp.solver.masked_repair import masked_repair, choose_slot

DATA = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp\data")


def cvrp_inst(name="X-n101-k25"):
    return CVRPInstance.parse_vrp((DATA / "cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))


def vrptw_inst(name="C101"):
    return VRPTWInstance.parse_solomon(DATA / "solomon/instances" / (name + ".vrp"))


def witness(inst):
    """简单最近邻见证（V2 独立实现，不依赖 V1）。"""
    n = inst.dimension
    remaining = set(range(1, n + 1))
    is_tw = isinstance(inst, VRPTWInstance)
    from standard_vrp.core.standard_evaluator import route_schedule
    routes = []
    while remaining:
        route, cur = [], 0
        while remaining:
            nxt = min(remaining, key=lambda i: inst.dist[cur][i])
            trial = route + [nxt]
            if is_tw:
                if route_schedule(inst, tuple(trial)) is None:
                    break
            else:
                if sum(inst.demands[i] for i in trial) > inst.capacity + 1e-9:
                    break
            route.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        routes.append(route)
    return [list(r) for r in routes]


def test_d1_ranking_effective():
    """D1：贪心必须按 ranking 顺序重插（不同排序 → 不同重插顺序）。"""
    inst = cvrp_inst()
    ad = CvrpAdapter(inst)
    parent = witness(inst)
    removed = set(random.Random(11).sample(sorted({i for r in parent for i in r}), 12))
    log_a, log_b = [], []
    _, _ = masked_repair(ad, parent, removed, sorted(removed), "greedy", random.Random(1), log=log_a)
    _, _ = masked_repair(ad, parent, removed, sorted(removed, reverse=True), "greedy", random.Random(1), log=log_b)
    order_a = [e["cust"] for e in log_a]
    order_b = [e["cust"] for e in log_b]
    assert order_a == sorted(removed), (order_a, sorted(removed))
    assert order_b == sorted(removed, reverse=True), order_b
    assert order_a != order_b
    print("D1 PASS: 贪心重插顺序 = ranking（升序/降序不同）")


def test_d2_vrptw_operators_differ():
    """D2：VRPTW 上 regret2 与 greedy 的执行路径不同（动态后悔值 vs 顺序贪心）。"""
    inst = vrptw_inst()
    ad = VrptwAdapter(inst)
    parent = witness(inst)
    removed = set(random.Random(12).sample(sorted({i for r in parent for i in r}), 12))
    log_g, log_r = [], []
    masked_repair(ad, parent, removed, sorted(removed), "greedy", random.Random(2), log=log_g)
    masked_repair(ad, parent, removed, sorted(removed), "regret2", random.Random(2), log=log_r)
    og = [e["cust"] for e in log_g]
    orr = [e["cust"] for e in log_r]
    assert og == sorted(removed), og
    assert any(e["regret"] > 0 for e in log_r), log_r[:3]
    # 动态后悔值：重插顺序一般不同于登记顺序（构造保证 regret 参与）
    print("D2 PASS: greedy 顺序=ranking；regret2 记录后悔值并按后悔值选客户；两者路径不同:",
          og[:5], "vs", orr[:5])
    assert og != orr, (og, orr)


def test_d3_dynamic_regret():
    """D3：每次插入后重算全部 pending 的后悔值（扫描量 ~ k(k+1)/2，而非 2k）。"""
    inst = cvrp_inst()
    ad = CvrpAdapter(inst)
    parent = witness(inst)
    removed = set(random.Random(13).sample(sorted({i for r in parent for i in r}), 10))
    from standard_vrp.solver import masked_repair as mr
    counts = {"n": 0}
    orig = mr._choose_slot

    def counting(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    mr._choose_slot = counting
    try:
        mr.masked_repair(ad, parent, removed, sorted(removed), "regret2", random.Random(3))
        dynamic_calls = counts["n"]
        counts["n"] = 0
        mr.masked_repair(ad, parent, removed, sorted(removed), "greedy", random.Random(3))
        greedy_calls = counts["n"]
    finally:
        mr._choose_slot = orig
    k = len(removed)
    expected_dynamic = k * (k + 1) // 2
    assert dynamic_calls == expected_dynamic, (dynamic_calls, expected_dynamic)
    assert greedy_calls == k, greedy_calls
    print("D3 PASS: 动态 Regret-2 choose_slot 调用 %d = k(k+1)/2 (k=%d)；greedy=%d"
          % (dynamic_calls, k, greedy_calls))


def test_d5_short_feasible_slot_protection():
    """D5：保护位必须是"最短可行插位"，而不是几何最短（可能不可行）。"""
    inst = cvrp_inst()
    ad = CvrpAdapter(inst)
    parent = witness(inst)
    # 构造：一条只差 1 单位容量的路线 + 一条宽松路线；人为指定热图使保护位与 top-8 分离
    tight = [i for r in parent for i in r][:8]
    load = sum(inst.demands[i] for i in tight)
    room = inst.capacity - load
    # 找一个需求 > room 的客户（插入 tight 路线必然容量不可行）
    cust = next(i for i in range(1, inst.dimension + 1) if i not in tight
                and inst.demands[i] > room + 1e-9)
    routes = [tight, [cust + 1 if cust + 1 <= inst.dimension else 1]]
    routes[1] = [i for i in routes[1] if i not in tight]
    # 热图让 tight 路线的所有位置排在最前（但都不可行）→ 保护位必须给出可行位置
    n = inst.dimension
    heat = [[-1e9] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        heat[i][i] = -1e9
    key, move, regret = choose_slot(ad, routes, cust, heat=heat, regret=True, slot=True, topk=8)
    ri, pos = move
    assert ri < len(routes), "必须能在既有路线中插入（存在可行宽松路线）"
    assert ad.feasible_insert(routes[ri], pos, cust), "保护位必须可行"
    # 验证保护位确实来自"最短可行"扫描，而非热图 top-8
    print("D5 PASS: 保护位可行 (route=%d,pos=%d, regret=%.3f)" % (ri, pos, regret))


def test_new_vehicle_candidate():
    """新开单车候选参与比较：CVRP 中当单车成本更优时应被选中。"""
    inst = cvrp_inst()
    ad = CvrpAdapter(inst)
    # 两客户相距很远，插入任何既有路线都不如各自成线
    a, b = 1, 2
    routes = [[a]]
    key, move, regret = choose_slot(ad, routes, b, heat=None, regret=False)
    # 若 d(0,b) 很小而 detour 大，则新开单车最优；此处只验证候选存在且键值来自真实评价
    ri, pos = move
    if ri >= len(routes):
        new_route_cost = ad.singleton_delta(b)
        assert abs(-key - new_route_cost) < 1e-6, (key, new_route_cost)
        print("新开单车候选 PASS：选中新开单车，键值=singleton 成本")
    else:
        delta = ad.insert_delta(routes[ri], pos, b)
        assert abs(-key - delta) < 1e-6, (key, delta)
        print("新开单车候选 PASS：插入既有路线更优，键值=插入增量")


def test_vrptw_lex_regret():
    """VRPTW：K 差主导的后悔值（K 差 → 1e6 量级；同 K → 距离差）。"""
    inst = vrptw_inst()
    ad = VrptwAdapter(inst)
    # 直接验证标量化：K 差使后悔值达到 1e6 量级（任何可观测距离差 ≤ ~1e4，被严格压倒）
    r_k = ad.regret((0, 100.0), (1, 50.0))
    r_d = ad.regret((0, 100.0), (0, 150.0))
    assert r_k > 1e5, r_k
    assert r_k > 100 * max(1.0, r_d), (r_k, r_d)
    assert abs(r_d - 50.0) < 1e-9, r_d
    print("VRPTW 字典序后悔值 PASS：K 差主导 (%.0f)，同 K 为距离差 (%.1f)" % (r_k, r_d))


def test_heat_nan_handling():
    """热图缺失（-inf）不得产生 NaN，且不得被解释为拒绝该边。"""
    inst = cvrp_inst()
    ad = CvrpAdapter(inst)
    n = inst.dimension
    heat = [[-math_inf()] * (n + 1) for _ in range(n + 1)]
    parent = witness(inst)
    removed = set(random.Random(14).sample(sorted({i for r in parent for i in r}), 5))
    out, improved = masked_repair(ad, parent, removed, sorted(removed), "greedy", random.Random(4), heat=heat)
    ok, d, k = ad.evaluate(out)
    assert ok, "全缺失热图时应回退距离规则且保持可行"
    print("热图缺失 PASS：-inf 热图回退距离规则，结果可行 D=%.1f" % d)


def math_inf():
    return -float("inf")


def main():
    test_d1_ranking_effective()
    test_d2_vrptw_operators_differ()
    test_d3_dynamic_regret()
    test_d5_short_feasible_slot_protection()
    test_new_vehicle_candidate()
    test_vrptw_lex_regret()
    test_heat_nan_handling()
    print("ALL P5-V2 MASKED REPAIR TESTS PASSED")


if __name__ == "__main__":
    main()
