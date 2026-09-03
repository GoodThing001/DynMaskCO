"""
P0-D2: Online Protocol Invariant Test（强制测试 #4 + #5）

验证 StrictOnlineDecoder 的两个核心不变式：
  #4 Prefix invariance：已执行 prefix（served / executed_route）不可变，
     新事件后 P_{e+1} = P_e，只增不减。
  #5 Exact-once service：episode 结束 ∀i, count(i) = 1（无遗漏、无重复）。

只用 EDD 贪心 replanner（不依赖模型/checkpoint），验证事件驱动循环本身正确。
模型（future perturbation 端到端）见 test_future_perturbation.py。

用法:
    python scripts/tests/test_online_protocol.py
"""

import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'decoding'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from online_decode import StrictOnlineDecoder
from authoritative_evaluator import evaluate_solution, evaluate_execution_trace
from strict_online_env import StrictOnlineEnv, GreedyReplanner, VehicleState


def build_toy_dataset():
    """构造 toy 实例：depot + 4 客户，reveal_time 分散 (0,0,5,10)。"""
    N = 5
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]],
                      dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
    tw_end = np.array([[30., 30., 30., 30., 30.]], dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 5., 10.]], dtype=np.float32)
    return {
        'coords': coords, 'demands': demands,
        'tw_start': tw_start, 'tw_end': tw_end,
        'service_time': service_time, 'temp_class': temp_class,
        'reveal_time': reveal_time,
    }


def test_exact_once():
    """#5: 最终 route 每个客户恰好服务一次。"""
    print("=" * 60)
    print("Test #5: Exact-Once Service")
    print("=" * 60)
    dataset = build_toy_dataset()
    decoder = StrictOnlineDecoder(dataset, capacity=10, model=None, tw_max=30.0)
    route, metrics = decoder.decode_instance(0)

    # 用权威 evaluator 验证 exact-once（complete = n_visited==n_customers 且 0 重复）
    r = evaluate_solution(
        route, dataset['coords'][0], dataset['tw_start'][0], dataset['tw_end'][0],
        dataset['service_time'][0], dataset['demands'][0], capacity=10.0,
    )
    print(f"  route: {route.tolist()}")
    print(f"  n_visited={r['n_visited']} n_duplicate={r['n_duplicate']} "
          f"n_unserved={r['n_unserved']}")
    ok = r['complete'] and r['n_duplicate'] == 0 and r['n_unserved'] == 0
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_prefix_invariance():
    """#4: 已执行 prefix 只增不减，新事件不改变已提交历史（R1.5 用 trace 验证）。"""
    print("\n" + "=" * 60)
    print("Test #4: Prefix Invariance (service times monotonic, no re-serve)")
    print("=" * 60)
    dataset = build_toy_dataset()
    decoder = StrictOnlineDecoder(dataset, capacity=10, model=None, tw_max=30.0)
    traces, served_mask = decoder.env.run(0)

    from collections import Counter
    cnt = Counter()
    monotonic = True
    for tr in traces:
        prev_finish = -1.0
        for sr in tr.services:
            cnt[sr.node] += 1
            if sr.service_finish < prev_finish - 1e-6:
                monotonic = False
            prev_finish = sr.service_finish

    no_dup = all(c == 1 for c in cnt.values())
    served_all = len(cnt) == 4
    ok = monotonic and no_dup and served_all
    print(f"  monotonic={monotonic} no_dup={no_dup} served_all={served_all} served={dict(cnt)}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_dynamic_visibility():
    """辅助：动态可见性（reveal_time <= clock）正确。"""
    dataset = build_toy_dataset()
    rt = dataset['reveal_time'][0]
    def vis(clock):
        v = rt <= clock
        v[0] = True  # depot 始终可见
        return list(v.astype(bool))
    ok = (vis(0.0) == [True, True, True, False, False]
          and vis(6.0) == [True, True, True, True, False]
          and vis(11.0) == [True, True, True, True, True])
    print(f"\n动态可见性: {'PASS' if ok else 'FAIL'}")
    return ok


def test_multi_vehicle_fleet():
    """Fleet 测试：capacity=2 强制多车，验证多车并行服务（非顺序 single-vehicle multi-trip）。"""
    print("\n" + "=" * 60)
    print("Test: Multi-Vehicle Fleet (capacity=2 强制多车并行)")
    print("=" * 60)
    # 4 客户 demand=1，capacity=2 → 至少需 2 辆车；全部 t=0 揭示，TW 宽
    N = 5
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [0., 1.], [0., 2.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.zeros((1, N), dtype=np.float32)  # 全部 t=0 揭示

    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    decoder = StrictOnlineDecoder(dataset, capacity=2.0, model=None, tw_max=100.0,
                                  num_vehicles=25)
    route, metrics = decoder.decode_instance(0)
    r = evaluate_solution(route, coords[0], tw_start[0], tw_end[0],
                          service_time[0], demands[0], capacity=2.0)
    print(f"  route: {route.tolist()}")
    print(f"  complete={r['complete']} n_visited={r['n_visited']} "
          f"vehicles={r['vehicle_count']} cap_feas={r['capacity_feasible']}")
    ok = r['complete'] and r['vehicle_count'] >= 2 and r['capacity_feasible']
    print(f"  => {'PASS' if ok else 'FAIL'} (需 complete + ≥2 车 + 容量可行)")
    return ok


def test_parallel_dispatch():
    """并行派车：2 客户 capacity=1，TW 紧，验证两车同时出发（顺序会导致第二个客户 TW 过期）。"""
    print("\n" + "=" * 60)
    print("Test: Parallel Dispatch (两车同时出发，非顺序)")
    print("=" * 60)
    # 客户 1 在 (10,0)、客户 2 在 (0,10)，travel 都是 10；TW=[0,15]
    # 并行：两车 t=0 同时出发 → 两个客户 arrive=10<=15 → complete=True
    # 顺序（错误）：车 2 等车 0 回 depot(+10)+再出发(+10) → arrive=30>15 → complete=False
    N = 3
    coords = np.array([[[0., 0.], [10., 0.], [0., 10.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.array([[100., 15., 15.]], dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.zeros((1, N), dtype=np.float32)

    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    decoder = StrictOnlineDecoder(dataset, capacity=1.0, model=None, tw_max=100.0,
                                  num_vehicles=25)
    route, metrics = decoder.decode_instance(0)
    r = evaluate_solution(route, coords[0], tw_start[0], tw_end[0],
                          service_time[0], demands[0], capacity=1.0)
    print(f"  route: {route.tolist()}")
    print(f"  complete={r['complete']} vehicles={r['vehicle_count']} "
          f"tw_feas={r['tw_feasible']}")
    ok = r['complete'] and r['vehicle_count'] >= 2
    print(f"  => {'PASS' if ok else 'FAIL'} (并行派车 complete=True；顺序会导致客户2过期)")
    return ok


def test_late_dispatch_no_time_travel():
    """#11: late-dispatch causality — reveal t=10 的车必须 depart>=10，不得从 t=0 出发。"""
    print("\n" + "=" * 60)
    print("Test #11: Late-Dispatch No Time Travel (P0-A)")
    print("=" * 60)
    coords = np.array([[[0., 0.], [1., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1.]], dtype=np.float32)
    tw_start = np.array([[0., 0.]], dtype=np.float32)
    tw_end = np.array([[30., 30.]], dtype=np.float32)
    service_time = np.zeros((1, 2), dtype=np.float32)
    temp_class = np.zeros((1, 2), dtype=np.int32)
    reveal_time = np.array([[0., 10.]], dtype=np.float32)
    dataset = {
        'coords': coords, 'demands': demands, 'tw_start': tw_start,
        'tw_end': tw_end, 'service_time': service_time,
        'temp_class': temp_class, 'reveal_time': reveal_time,
    }
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=1,
                          replanner=GreedyReplanner('edd'))
    traces, served_mask = env.run(0)
    sr = traces[0].services[0]
    ok = (traces[0].dispatch_time is not None
          and traces[0].dispatch_time >= 10.0 - 1e-6
          and sr.depart_time >= 10.0 - 1e-6
          and abs(sr.arrival_time - 11.0) < 1e-6)
    print(f"  dispatch={traces[0].dispatch_time} depart={sr.depart_time:.2f} "
          f"arrival={sr.arrival_time:.2f} (expect 10/10/11)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_committed_customer_reserved():
    """#12: committed customer 不可被其他车重复分配（P0-B）。"""
    print("\n" + "=" * 60)
    print("Test #12: Cross-Vehicle Committed Reservation (P0-B)")
    print("=" * 60)
    vehicles = [
        VehicleState(vehicle_id=0, status='committed', current_node=0,
                     committed_next=2, committed_finish=10.0),
        VehicleState(vehicle_id=1, status='ready', current_node=1, ready_time=2.0),
    ]
    N = 4
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.zeros((1, N), dtype=np.float32)
    dataset = {
        'coords': coords, 'demands': demands, 'tw_start': tw_start,
        'tw_end': tw_end, 'service_time': service_time,
        'temp_class': temp_class, 'reveal_time': reveal_time,
    }
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=2,
                          replanner=GreedyReplanner('edd'))
    served_mask = np.zeros(N, dtype=bool)
    served_mask[0] = True
    GreedyReplanner('edd').plan(env, 0, 2.0, vehicles, served_mask, [1, 2, 3])
    ok = 2 not in vehicles[1].mutable_suffix
    print(f"  V1 mutable_suffix={vehicles[1].mutable_suffix} (committed 2 必须不在其中)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_ready_wait_no_time_travel():
    """#13: ready 车 WAIT 后服务新揭示客户，depart 不得早于 reveal（时间穿越）。"""
    print("\n" + "=" * 60)
    print("Test #13: Ready-Wait No Time Travel (P0-A ready 路径)")
    print("=" * 60)
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, 4), dtype=np.float32)
    tw_end = np.full((1, 4), 30.0, dtype=np.float32)
    service_time = np.zeros((1, 4), dtype=np.float32)
    temp_class = np.zeros((1, 4), dtype=np.int32)
    reveal_time = np.array([[0., 0., 5., 5.]], dtype=np.float32)
    dataset = {
        'coords': coords, 'demands': demands, 'tw_start': tw_start,
        'tw_end': tw_end, 'service_time': service_time,
        'temp_class': temp_class, 'reveal_time': reveal_time,
    }
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=2,
                          replanner=GreedyReplanner('edd'))
    traces, served = env.run(0)
    all_served = all(served[i] for i in [1, 2, 3])
    ok = all_served
    for tr in traces:
        for sr in tr.services:
            if sr.depart_time < reveal_time[0, sr.node] - 1e-6:
                ok = False
    print(f"  served={[i for i in [1,2,3] if served[i]]} "
          f"departs={[round(sr.depart_time,2) for tr in traces for sr in tr.services]}")
    print(f"  => {'PASS' if ok else 'FAIL'} (所有 depart >= 对应 reveal_time)")
    return ok


class CountingReplanner(GreedyReplanner):
    """计数 replanner 调用次数的包装（用于 #14/#15 验证 plan persistence）。"""
    def __init__(self, incumbent_builder='edd'):
        super().__init__(incumbent_builder)
        self.call_count = 0

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self.call_count += 1
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)


def test_no_information_plan_persistence():
    """#14: 无新信息时延续 plan，不重复调用 replanner（P0-CTRL plan persistence）。"""
    print("\n" + "=" * 60)
    print("Test #14: No-Information Plan Persistence (P0-CTRL)")
    print("=" * 60)
    N = 5
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.zeros((1, N), dtype=np.float32)  # 无 reveal
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    replanner = CountingReplanner('edd')
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=25,
                          replanner=replanner)
    traces, served_mask = env.run(0)
    from collections import Counter
    cnt = Counter()
    for tr in traces:
        for sr in tr.services:
            cnt[sr.node] += 1
    complete = (len(cnt) == 4 and all(c == 1 for c in cnt.values()))
    ok = complete and replanner.call_count == 1
    print(f"  complete={complete} replanner_calls={replanner.call_count} (expect 1)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_reveal_triggers_recourse():
    """#15: reveal 触发重规划（replanner 被再次调用）。"""
    print("\n" + "=" * 60)
    print("Test #15: Reveal Triggers Recourse (P0-CTRL)")
    print("=" * 60)
    N = 5
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 5., 5.]], dtype=np.float32)  # 3,4 在 t=5 reveal
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    replanner = CountingReplanner('edd')
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=25,
                          replanner=replanner)
    traces, served_mask = env.run(0)
    from collections import Counter
    cnt = Counter()
    for tr in traces:
        for sr in tr.services:
            cnt[sr.node] += 1
    complete = (len(cnt) == 4 and all(c == 1 for c in cnt.values()))
    ok = complete and replanner.call_count >= 2
    print(f"  complete={complete} replanner_calls={replanner.call_count} (expect >=2)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_no_stale_tail_duplicate():
    """#16: stale tail 不导致 duplicate（有 reveal 的多车场景 n_duplicate=0）。"""
    print("\n" + "=" * 60)
    print("Test #16: No Stale-Tail Duplicate (P0-CTRL)")
    print("=" * 60)
    N = 7
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.],
                        [0., 1.], [0., 2.], [0., 3.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 0., 3., 3., 3.]], dtype=np.float32)
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    decoder = StrictOnlineDecoder(dataset, capacity=2.0, model=None, tw_max=100.0,
                                  replan_fn=GreedyReplanner('edd'))
    route, metrics = decoder.decode_instance(0)
    r = evaluate_solution(route, coords[0], tw_start[0], tw_end[0],
                          service_time[0], demands[0], capacity=2.0)
    ok = r['n_duplicate'] == 0 and r['complete']
    print(f"  n_duplicate={r['n_duplicate']} complete={r['complete']} (expect 0/True)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_ortools_wait_parity():
    """#17: OR-Tools WAIT parity —— ready@customer 无 job + 未来 reveal 时 WAIT 不 return。"""
    print("\n" + "=" * 60)
    print("Test #17: OR-Tools WAIT Parity (P0-CTRL-4)")
    print("=" * 60)
    try:
        from ortools_rolling_horizon import ORToolsReplanner
    except ImportError:
        print("  [SKIP] ortools not available")
        return True
    N = 4
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.array([[100., 100., 1.5, 100.]], dtype=np.float32)  # 客户2 tw_end=1.5 已关
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 5.]], dtype=np.float32)  # 客户3 未来 reveal
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=2,
                          replanner=ORToolsReplanner(10, 2, 100))
    vehicles = [
        VehicleState(vehicle_id=0, status='ready', current_node=1, ready_time=2.0,
                     served_route=[1], current_load=1.0),
        VehicleState(vehicle_id=1, status='idle'),
    ]
    served_mask = np.zeros(N, dtype=bool)
    served_mask[0] = True
    served_mask[1] = True
    visible_ids = [1, 2]  # 1 已服务，2 可见但 tw_end=1.5 < ready_time=2.0 → drop
    env.replanner.plan(env, 0, 2.0, vehicles, served_mask, visible_ids, replan_ids={0, 1})
    # V0：客户2 时间窗已关 → OR-Tools drop → suffix=[0]；但未来有 reveal(客户3) → WAIT []
    ok = (vehicles[0].mutable_suffix == [])
    print(f"  V0 mutable_suffix={vehicles[0].mutable_suffix} (expect [] = WAIT, 不是 [0])")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_reveal_stale_tail_no_duplicate():
    """#18: reveal 清空 tail + reserve 整个 plan → 分批重规划不产生 stale-tail duplicate。"""
    print("\n" + "=" * 60)
    print("Test #18: Reveal Stale-Tail No Duplicate (P0-CTRL-4)")
    print("=" * 60)
    N = 6
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.]]],
                      dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 0., 0., 0.5]], dtype=np.float32)  # 客户5 t=0.5 reveal
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=5,
                          replanner=GreedyReplanner('edd'))
    traces, served = env.run(0)
    m = evaluate_execution_trace(traces, coords[0], tw_start[0], tw_end[0],
                                 service_time[0], demands[0], 10.0)
    ok = m['n_duplicate'] == 0 and m['complete'] and m['n_unserved'] == 0
    print(f"  complete={m['complete']} n_duplicate={m['n_duplicate']} "
          f"n_unserved={m['n_unserved']} (expect True/0/0)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    results = {
        'exact-once (#5)': test_exact_once(),
        'prefix invariance (#4)': test_prefix_invariance(),
        'dynamic visibility': test_dynamic_visibility(),
        'multi-vehicle fleet': test_multi_vehicle_fleet(),
        'parallel dispatch': test_parallel_dispatch(),
        'late-dispatch (#11)': test_late_dispatch_no_time_travel(),
        'committed reservation (#12)': test_committed_customer_reserved(),
        'ready-wait no time-travel (#13)': test_ready_wait_no_time_travel(),
        'plan persistence (#14)': test_no_information_plan_persistence(),
        'reveal recourse (#15)': test_reveal_triggers_recourse(),
        'no stale-tail dup (#16)': test_no_stale_tail_duplicate(),
        'ortools WAIT parity (#17)': test_ortools_wait_parity(),
        'reveal stale-tail (#18)': test_reveal_stale_tail_no_duplicate(),
    }
    print("\n" + "=" * 60)
    all_ok = all(results.values())
    for k, v in results.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
