"""
R1.5-M Model-Utility Measurement Blocking Tests（M1-M7）。

验证 R1.5 的「模型效用测量链」正确，不依赖模型/checkpoint：
  M1 PlanExhaustion —— 车辆服务完最后一个 plan 节点（无尾计划）必须触发 replan
  M2 score multiset —— H3 shuffle 严格保持可行候选边 score multiset 不变
  M3 candidate-set invariance —— logit_mode 不参与 feasibility（real/shuffle 同候选集）
  M4 exclusion —— future / served / capacity / TW infeasible 节点不进 shuffle 候选集
  M5 deterministic —— 同 seed 同 state 下 H3 shuffle 确定性
  M6 real-mode regression —— 修复不改变 H2 real Resource Beam 输出
  M7 real-vs-shuffle action split —— 宽 TW 下 shuffle 至少能产生 first-action 分叉

用法:
    python scripts/tests/test_r1_5_model_utility.py
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

from strict_online_env import StrictOnlineEnv, Replanner, GreedyReplanner
from resource_beam import ResourceBeamSearcher, BeamState, derive_seed
from authoritative_evaluator import evaluate_execution_trace


def _first(route):
    """route 第一个非 0 节点（0=return depot，-1=None）。"""
    if route is None:
        return -1
    for n in route:
        n = int(n)
        if n != 0:
            return n
    return 0


# ============================================================================
# M1: PlanExhaustion
# ============================================================================

class TwoPhaseReplanner(Replanner):
    """第一次 plan=[1]（故意不带 depot），第二次 plan=[2,0]。"""

    def __init__(self):
        self.call_count = 0

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self.call_count += 1
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            if self.call_count == 1:
                v.mutable_suffix = [1]      # 服务 1 后 plan 用完（无 depot 尾）
            else:
                v.mutable_suffix = [2, 0]   # 服务 2 后返回 depot


def test_m1_plan_exhaustion():
    print("=" * 60)
    print("M1: PlanExhaustion triggers replan (no tail → needs_replan)")
    print("=" * 60)
    N = 3
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, N), dtype=np.float32)
    tw_end = np.full((1, N), 100.0, dtype=np.float32)
    service_time = np.zeros((1, N), dtype=np.float32)
    temp_class = np.zeros((1, N), dtype=np.int32)
    reveal_time = np.zeros((1, N), dtype=np.float32)  # 1,2 均 t=0 可见
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
               'tw_end': tw_end, 'service_time': service_time,
               'temp_class': temp_class, 'reveal_time': reveal_time}

    replanner = TwoPhaseReplanner()
    env = StrictOnlineEnv(dataset, capacity=10, tw_speed=1.0, num_vehicles=1,
                          replanner=replanner)
    traces, served = env.run(0)
    m = evaluate_execution_trace(traces, coords[0], tw_start[0], tw_end[0],
                                 service_time[0], demands[0], 10.0)
    ok = (replanner.call_count >= 2 and m['complete']
          and m['n_duplicate'] == 0 and m['n_unserved'] == 0)
    print(f"  replanner_calls={replanner.call_count} (expect >=2)")
    print(f"  complete={m['complete']} dup={m['n_duplicate']} unserved={m['n_unserved']}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 共享 toy builder（M2/M3/M5/M6/M7 用）
# ============================================================================

def build_wide_searcher(num_customers=5, capacity=10, depot_dispatch_time=0.0):
    """depot 在原点 + num_customers 个沿 x 轴排列的客户，宽 TW、全可见。"""
    N = num_customers + 1
    coords = np.zeros((N, 2), dtype=np.float32)
    coords[1:, 0] = np.arange(1, num_customers + 1, dtype=np.float32)
    demands = np.zeros(N, dtype=np.float32)
    demands[1:] = 1.0
    tw_start = np.zeros(N, dtype=np.float32)
    tw_end = np.full(N, 100.0, dtype=np.float32)
    service_time = np.zeros(N, dtype=np.float32)
    visible_mask = np.ones(N, dtype=bool)
    searcher = ResourceBeamSearcher(
        coords, tw_start, tw_end, service_time, demands, capacity,
        speed=1.0, K=8, visible_mask=visible_mask,
        depot_dispatch_time=depot_dispatch_time)
    served_mask = np.zeros(N, dtype=bool)
    served_mask[0] = True
    return searcher, served_mask


# ============================================================================
# M2: score multiset preservation
# ============================================================================

def test_m2_score_multiset():
    print("\n" + "=" * 60)
    print("M2: H3 shuffle preserves score multiset per candidate set")
    print("=" * 60)
    searcher, served_mask = build_wide_searcher(num_customers=5)
    N = 6
    edge_logits = np.random.default_rng(0).standard_normal((N, N)).astype(np.float32)
    _, success, _ = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=False,
        logit_mode='shuffle', shuffle_seed=42, return_meta=True,
        record_shuffle_audit=True)
    audit = searcher.last_shuffle_audit
    ok = success and len(audit) > 0
    max_err = 0.0
    for entry in audit:
        before = np.asarray(entry['scores_before'], dtype=np.float64)
        after = np.asarray(entry['scores_after'], dtype=np.float64)
        np.testing.assert_allclose(np.sort(before), np.sort(after), atol=1e-7)
        max_err = max(max_err, float(np.abs(np.sort(before) - np.sort(after)).max()))
    print(f"  success={success} shuffle_events={len(audit)} max_multiset_err={max_err:.2e}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# M3: candidate-set invariance
# ============================================================================

def test_m3_candidate_set_invariance():
    print("\n" + "=" * 60)
    print("M3: logit_mode does not participate in feasibility")
    print("=" * 60)
    searcher, served_mask = build_wide_searcher(num_customers=5)
    N = 6
    edge_logits = np.random.default_rng(1).standard_normal((N, N)).astype(np.float32)
    _, _, meta_real = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=True,
        logit_mode='real', return_meta=True)
    _, _, meta_shuffle = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=True,
        logit_mode='shuffle', shuffle_seed=42, return_meta=True)
    ok = (meta_real['initial_actionable_count'] == meta_shuffle['initial_actionable_count'])
    print(f"  real actionable={meta_real['initial_actionable_count']} "
          f"shuffle actionable={meta_shuffle['initial_actionable_count']}")
    print(f"  => {'PASS' if ok else 'FAIL'} (须相等)")
    return ok


# ============================================================================
# M4: exclusion (future/served/capacity/TW)
# ============================================================================

def test_m4_exclusion():
    print("\n" + "=" * 60)
    print("M4: future/served/capacity/TW-infeasible nodes excluded from candidates")
    print("=" * 60)
    N = 7
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.],
                       [4., 0.], [5., 0.], [0., 1.]], dtype=np.float32)
    demands = np.array([0., 1., 1., 1., 100., 1., 1.], dtype=np.float32)  # 4 容量不可行
    tw_start = np.zeros(N, dtype=np.float32)
    tw_end = np.full(N, 100.0, dtype=np.float32)
    tw_end[5] = 0.5  # 5 TW 已关（从 depot travel=5 > 0.5）
    service_time = np.zeros(N, dtype=np.float32)
    visible_mask = np.ones(N, dtype=bool)
    visible_mask[3] = False  # 3 future
    searcher = ResourceBeamSearcher(
        coords, tw_start, tw_end, service_time, demands, 10.0,
        speed=1.0, K=8, visible_mask=visible_mask)

    beam = BeamState(route=[0], current_node=0, current_time=0.0,
                     current_load=0.0, visited={0, 2})  # 2 已服务
    init_visited = {0, 2}
    all_unvisited = {j for j in range(1, N) if visible_mask[j]} - init_visited
    cands = searcher._feasible_next_nodes(beam, all_unvisited, init_visited,
                                          single_route=True)
    cand_set = set(cands)
    ok = (cand_set == {1, 6}
          and 2 not in cand_set and 3 not in cand_set
          and 4 not in cand_set and 5 not in cand_set)
    print(f"  candidate set={sorted(cand_set)} (expect [1, 6])")
    print(f"  excluded: 2(served)={'ok' if 2 not in cand_set else 'FAIL'} "
          f"3(future)={'ok' if 3 not in cand_set else 'FAIL'} "
          f"4(cap)={'ok' if 4 not in cand_set else 'FAIL'} "
          f"5(tw)={'ok' if 5 not in cand_set else 'FAIL'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# M5: deterministic shuffle
# ============================================================================

def test_m5_deterministic():
    print("\n" + "=" * 60)
    print("M5: H3 shuffle deterministic under same seed/state")
    print("=" * 60)
    searcher, served_mask = build_wide_searcher(num_customers=5)
    N = 6
    edge_logits = np.random.default_rng(2).standard_normal((N, N)).astype(np.float32)
    ra, sa = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=False,
        logit_mode='shuffle', shuffle_seed=42)
    rb, sb = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=False,
        logit_mode='shuffle', shuffle_seed=42)
    ok = (ra == rb and sa == sb)
    print(f"  route_a==route_b={ra == rb} success_a==success_b={sa == sb}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# M6: real-mode regression
# ============================================================================

def test_m6_real_mode_regression():
    print("\n" + "=" * 60)
    print("M6: real-mode unchanged by H3 control (no regression)")
    print("=" * 60)
    searcher, served_mask = build_wide_searcher(num_customers=5)
    N = 6
    edge_logits = np.random.default_rng(3).standard_normal((N, N)).astype(np.float32)
    r_default, s_default = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=False)
    r_real, s_real = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=False, logit_mode='real')
    ok = (r_default == r_real and s_default == s_real)
    print(f"  default==real route={r_default == r_real} success={s_default == s_real}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# M7: real-vs-shuffle action split
# ============================================================================

def test_m7_real_vs_shuffle_split():
    print("\n" + "=" * 60)
    print("M7: shuffle produces first-action split vs real (wide TW)")
    print("=" * 60)
    searcher, served_mask = build_wide_searcher(num_customers=5)
    N = 6
    edge_logits = np.zeros((N, N), dtype=np.float32)
    edge_logits[0, 1:] = np.array([10., 8., 6., 4., 2.], dtype=np.float32)  # unique

    route_real, _ = searcher.generate_from_state(
        edge_logits, 0, 0.0, 0.0, served_mask, single_route=True, logit_mode='real')
    real_first = _first(route_real)

    found_split = False
    shuffle_first_seen = set()
    for seed in range(20):
        route_s, _ = searcher.generate_from_state(
            edge_logits, 0, 0.0, 0.0, served_mask, single_route=True,
            logit_mode='shuffle', shuffle_seed=seed)
        sf = _first(route_s)
        shuffle_first_seen.add(sf)
        if sf != real_first:
            found_split = True
            break
    ok = real_first == 1 and found_split
    print(f"  real_first={real_first} (expect 1=argmax depot logit)")
    print(f"  shuffle_firsts sampled={sorted(shuffle_first_seen)} "
          f"split_found={found_split}")
    print(f"  => {'PASS' if ok else 'FAIL'} (须 real_first=1 且存在 shuffle 分叉)")
    return ok


# ============================================================================
# M8: guarded selection rule（R1.6）
# ============================================================================

def test_m8_guarded_selection():
    print("\n" + "=" * 60)
    print("M8: guarded selection rule (service-first lexicographic)")
    print("=" * 60)
    from online_decode import _apply_selection
    cases = []
    # (selection_mode, candidate_exists, cand_n, inc_n, cand_cost, inc_cost) -> (accepted, reason)
    cases.append((('guarded', True, 5, 3, 10.0, 12.0), (True, 'more_service')))
    cases.append((('guarded', True, 2, 3, 8.0, 12.0), (False, 'less_service')))
    cases.append((('guarded', True, 3, 3, 8.0, 10.0), (True, 'lower_cost_same_service')))
    cases.append((('guarded', True, 3, 3, 12.0, 10.0), (False, 'not_better')))
    cases.append((('guarded', True, 3, 3, 10.0, 10.0), (False, 'not_better')))  # 相等不超 1e-9
    cases.append((('guarded', False, 0, 3, float('nan'), 10.0), (False, 'candidate_missing')))
    cases.append((('raw', True, 3, 3, 12.0, 10.0), (True, 'raw_accept')))  # raw 无条件接受
    cases.append((('raw', False, 0, 3, float('nan'), 10.0), (False, 'candidate_missing')))
    ok = True
    for args, expected in cases:
        got = _apply_selection(*args)
        if got != expected:
            ok = False
            print(f"  FAIL: _apply_selection{args} = {got}, expect {expected}")
    print(f"  {len(cases)} cases, all_match={ok}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    results = {
        'M1 PlanExhaustion': test_m1_plan_exhaustion(),
        'M2 score multiset': test_m2_score_multiset(),
        'M3 candidate-set invariance': test_m3_candidate_set_invariance(),
        'M4 exclusion': test_m4_exclusion(),
        'M5 deterministic': test_m5_deterministic(),
        'M6 real-mode regression': test_m6_real_mode_regression(),
        'M7 real-vs-shuffle split': test_m7_real_vs_shuffle_split(),
        'M8 guarded selection': test_m8_guarded_selection(),
    }
    print("\n" + "=" * 60)
    all_ok = all(results.values())
    for k, v in results.items():
        print(f"  {k:<26} {'PASS' if v else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS (8/8)' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
