"""
P0-R comparator protocol 测试（02 手册 §P0-R1 / §P0-R3 验收）。

覆盖：
  1. solve_or_fixed 的 5 类 synthetic counterexample（dead-end / 永久不可行 / k=0,1,8,9 /
     nonzero service+waiting+return / returned set equality）；
  2. service_first 统计原语（hard_outcome / service_equivalent_idx / bootstrap / win-tie-loss /
     aggregate_solver_status 的 n_partial==0）；
  3. 真实 VAL 数据上 ORFixedReplanner 与 NN greedy 的 served-set 不变式（无 silent drop）。

纯 NumPy（无 JAX / OR-Tools），可本地跑：
    python scripts/tests/test_comparator_protocol.py

产物：results/p0r/evaluation_protocol_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))          # scripts/tests
_SCRIPTS = os.path.dirname(_BASE)                            # scripts
_CVRPTW = os.path.dirname(_SCRIPTS)                          # C-VRP.../
for p in ('simulation', 'evaluation', 'baselines'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv, GreedyReplanner
from authoritative_evaluator import evaluate_execution_trace
from b0_or_expert import solve_or_fixed, ORFixedReplanner
from service_first import (hard_outcome, service_equivalent_idx, paired_bootstrap_ci,
                           win_tie_loss, aggregate_solver_status)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _sf(assigned, coords, tw_end, tw_start=None, service=None, demands=None,
        capacity=10.0, current_node=0, ready_time=0.0, current_load=0.0):
    n = coords.shape[0]
    tw_start = np.zeros(n) if tw_start is None else np.asarray(tw_start, dtype=float)
    service = np.zeros(n) if service is None else np.asarray(service, dtype=float)
    demands = np.ones(n) if demands is None else np.asarray(demands, dtype=float)
    return solve_or_fixed(np.asarray(coords, dtype=float), demands, tw_start,
                          np.asarray(tw_end, dtype=float), service, capacity,
                          0.0, current_node, ready_time, current_load, assigned)


def test_dead_end_vs_feasible():
    # 最近邻（node1 距 depot 1）先走会堵死 node2（tw_end=3，绕行 4.16>3），
    # 但先 node2 后 node1 完整可行。旧贪心回退会 drop node2；修复后应返回 optimal 全序。
    coords = np.array([[0., 0.], [0., 1.], [3., 0.]])   # depot, near(0,1), far(3,0)
    tw_end = np.array([100., 100., 3.0])
    feasible, route, status = _sf([1, 2], coords, tw_end)
    ok = (feasible and status == 'optimal'
          and route == [0, 2, 1, 0] and sorted(route[1:-1]) == [1, 2])
    record('dead_end_vs_feasible', ok, f"route={route}, status={status}")


def test_permanently_infeasible():
    # 客户 dist=10 > tw_end=5，永久不可行 → unavailable_infeasible（绝不 partial）。
    coords = np.array([[0., 0.], [10., 0.]])
    tw_end = np.array([100., 5.0])
    feasible, route, status = _sf([1], coords, tw_end)
    ok = (not feasible and status == 'unavailable_infeasible' and route is None)
    record('permanently_infeasible', ok, f"status={status}, route={route}")


def test_k_boundaries():
    # k=0 → empty
    coords1 = np.array([[0., 0.], [1., 0.]])
    tw1 = np.array([100., 100.])
    f, r, s = _sf([], coords1, tw1)
    ok0 = (f and s == 'empty' and r == [0, 0])
    record('k=0_empty', ok0, f"status={s}, route={r}")

    # k=1 → optimal
    f, r, s = _sf([1], coords1, tw1)
    ok1 = (f and s == 'optimal' and sorted(r[1:-1]) == [1])
    record('k=1_optimal', ok1, f"status={s}")

    # k=8 → optimal（8! 全排列）
    coords8 = np.array([[float(i), 0.] for i in range(9)])
    tw8 = np.full(9, 1000.0)
    f, r, s = _sf(list(range(1, 9)), coords8, tw8, demands=np.ones(9), capacity=100.0)
    ok8 = (f and s == 'optimal' and sorted(r[1:-1]) == list(range(1, 9)))
    record('k=8_optimal', ok8, f"status={s}, n_returned={len(r)-2}")

    # k=9 → unavailable_k_over8（冻结策略：不 partial，不 drop）
    coords9 = np.array([[float(i), 0.] for i in range(10)])
    tw9 = np.full(10, 1000.0)
    f, r, s = _sf(list(range(1, 10)), coords9, tw9, demands=np.ones(10), capacity=100.0)
    ok9 = (not f and s == 'unavailable_k_over8' and r is None)
    record('k=9_unavailable', ok9, f"status={s}, route={r}")


def test_nonzero_service_waiting_return():
    # service time 必须正确累加：先 1（service=5）会堵死 2（tw_end=6），先 2 后 1 可行。
    coords = np.array([[0., 0.], [1., 0.], [2., 0.]])
    tw_end = np.array([100., 100., 6.0])
    service = np.array([0., 5., 0.])
    f, r, s = _sf([1, 2], coords, tw_end, service=service)
    ok_svc = (f and s == 'optimal' and r == [0, 2, 1, 0])
    record('nonzero_service', ok_svc, f"route={r}")

    # waiting（tw_start 5）仍可行
    tw_start = np.array([0., 5., 0.])
    tw_end_w = np.array([100., 100., 100.])
    f, r, s = _sf([1, 2], coords, tw_end_w, tw_start=tw_start, service=np.zeros(3))
    record('nonzero_waiting', f and s == 'optimal', f"status={s}")

    # return horizon：depot tw_end=4；服务客户 2（x=5）返回 10>4 → infeasible
    coords2 = np.array([[0., 0.], [1., 0.], [5., 0.]])
    tw_end_r = np.array([4., 100., 100.])
    f, r, s = _sf([1], coords2, tw_end_r)
    ok_ret1 = (f and s == 'optimal')
    f2, r2, s2 = _sf([2], coords2, tw_end_r)
    ok_ret2 = (not f2 and s2 == 'unavailable_infeasible')
    record('return_horizon', ok_ret1 and ok_ret2, f"s1={s}, s2={s2}")


def test_returned_set_equality_sweep():
    # 小规模 battery：feasible 时 returned multiset == assigned multiset 且无重复。
    rng = np.random.default_rng(7)
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [0., 3.]])
    fails = 0
    for trial in range(200):
        k = int(rng.integers(1, 6))
        assigned = list(rng.choice([1, 2, 3, 4, 5], size=k, replace=False))
        tw_end = rng.uniform(1.0, 30.0, size=6)
        tw_end[0] = 100.0
        service = rng.uniform(0.0, 2.0, size=6); service[0] = 0.0
        demands = rng.uniform(0.5, 2.0, size=6); demands[0] = 0.0
        f, r, s = _sf(assigned, coords, tw_end, service=service, demands=demands, capacity=6.0)
        if f:
            if s != 'optimal':
                fails += 1
                continue
            got = sorted(r[1:-1])
            if got != sorted(assigned) or len(got) != len(set(got)):
                fails += 1
        else:
            if s not in ('unavailable_infeasible', 'unavailable_k_over8') or r is not None:
                fails += 1
    record('returned_set_equality_sweep', fails == 0, f"fails={fails}/200")


def test_service_first_primitives():
    # hard_outcome / service_equivalent_idx
    m = lambda u, d, tw, cap, ret: {'n_unserved': u, 'n_duplicate': d, 'tw_feasible': tw,
                                    'capacity_feasible': cap, 'depot_return_feasible': ret,
                                    'complete': (u == 0 and d == 0), 'distance_cost': 10.0}
    results = {'A': [m(0, 0, True, True, True), m(1, 0, True, True, True)],
               'B': [m(0, 0, True, True, True), m(0, 0, True, True, True)]}
    idx = service_equivalent_idx(results, ['A', 'B'], 2)
    ok_idx = (idx == [0])  # instance 0 service-eq，instance 1 not
    record('service_equivalent_idx', ok_idx, f"idx={idx}")

    # paired_bootstrap_ci / win_tie_loss
    lo, hi = paired_bootstrap_ci([-1.0, -2.0, -3.0], seed=42)
    ok_ci = (lo < 0 and hi < 0)
    w, t, l = win_tie_loss([-1.0, -2.0, 0.0, 1.0])
    ok_wtl = (w == 2 and t == 1 and l == 1)
    record('bootstrap_win_tie_loss', ok_ci and ok_wtl, f"CI=({lo:.2f},{hi:.2f}), WTL=({w},{t},{l})")

    # aggregate_solver_status：optimal/unavailable 都 returned==assigned → 不计 partial；
    # greedy_seq 有 drop → n_partial=1, n_dropped=1, solver_status='has_drop'
    log = [
        {'inst_idx': 0, 'vehicle_id': 0, 'k': 2, 'status': 'optimal',
         'assigned_set': (1, 2), 'returned_set': (1, 2)},
        {'inst_idx': 0, 'vehicle_id': 1, 'k': 3, 'status': 'unavailable_k_over8',
         'assigned_set': (3, 4, 5), 'returned_set': (3, 4, 5)},
        {'inst_idx': 0, 'vehicle_id': 2, 'k': 3, 'status': 'greedy_seq',
         'assigned_set': (6, 7, 8), 'returned_set': (6, 7)},
    ]
    s = aggregate_solver_status(log, 0)
    ok_agg = (s['solver_status'] == 'has_drop' and s['n_partial'] == 1
              and s['n_dropped'] == 1 and s['n_optimal'] == 1
              and s['n_unavailable'] == 1 and s['n_k_over8'] == 1
              and abs(s['coverage'] - 7.0 / 8.0) < 1e-9)
    record('aggregate_solver_status', ok_agg,
           f"solver_status={s['solver_status']}, n_partial={s['n_partial']}, "
           f"n_dropped={s['n_dropped']}, coverage={s['coverage']:.4f}")

    # no_solves → coverage=nan（而非 1.0）
    s0 = aggregate_solver_status([], 0)
    ok_ns = (s0['solver_status'] == 'no_solves' and s0['n_solve_calls'] == 0
             and np.isnan(s0['coverage']))
    record('aggregate_solver_status_no_solves', ok_ns, f"coverage={s0['coverage']}")


def test_integration_no_silent_drop():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    if not os.path.exists(data_path):
        record('integration_no_silent_drop', True, 'SKIP: data not found')
        return
    dataset = dict(np.load(data_path))
    n = 4
    cap, nv = 50, 25

    def run(replanner):
        env = StrictOnlineEnv(dataset, cap, 1.0, nv, replanner=replanner)
        out = []
        for i in range(n):
            traces, served = env.run(i)
            m = evaluate_execution_trace(traces, env.coords[i], env.tw_start[i], env.tw_end[i],
                                         env.service_time[i], env.demands[i], cap, speed=1.0,
                                         dist_mat=env.dist_mat[i])
            out.append(m)
        return out

    nn = run(GreedyReplanner('nn'))
    orf_rp = ORFixedReplanner(cap, 500)
    orf = run(orf_rp)

    ok_set = True
    for i in range(n):
        if nn[i]['n_unserved'] != orf[i]['n_unserved'] or nn[i]['n_duplicate'] != orf[i]['n_duplicate']:
            ok_set = False
            break
    ok_partial = all(aggregate_solver_status(orf_rp.solve_status_log, i)['n_partial'] == 0
                     for i in range(n))
    record('integration_no_silent_drop', ok_set and ok_partial,
           f"n_unserved NN={[nn[i]['n_unserved'] for i in range(n)]} "
           f"OR-fixed={[orf[i]['n_unserved'] for i in range(n)]}, n_partial=0={ok_partial}")


def test_empty_assignment_return_tw():
    # 空分配也要查返回 depot 的 TW（contract：feasible=True iff return 可行）。
    coords = np.array([[0., 0.], [10., 0.]])   # depot, customer at x=10
    tw_end_tight = np.array([4., 100.])        # depot 关闭于 4，返回 10 > 4 → 不可行
    f, r, s = _sf([], coords, tw_end_tight, current_node=1)
    ok_inf = (not f and s == 'unavailable_infeasible' and r is None)
    record('empty_assignment_return_tw_infeasible', ok_inf, f"status={s}")

    tw_end_wide = np.array([100., 100.])
    f2, r2, s2 = _sf([], coords, tw_end_wide, current_node=1)
    ok_feas = (f2 and s2 == 'empty' and r2 == [1, 0])
    record('empty_assignment_return_tw_ok', ok_feas, f"status={s2}, route={r2}")


def test_dist_mat_threading():
    # 验证 solve_or_fixed 使用传入的 dist_mat（asymmetric），而非回退 Euclidean。
    coords = np.array([[0., 0.], [1., 0.], [2., 0.]])
    tw_start = np.zeros(3); tw_end = np.array([100., 100., 1.5]); service = np.zeros(3)
    demands = np.ones(3)
    # Euclidean 下 (1,2) 到 c2=2 > 1.5 不可行；asymmetric d(1,2)=0.1 使 (1,2) 可行（arrive c2=1.1）。
    dm = np.array([[0., 1., 2.],
                   [1., 0., 0.1],
                   [2., 0.1, 0.]], dtype=float)
    f, r, s = solve_or_fixed(coords, demands, tw_start, tw_end, service, 10.0, 0.0,
                             0, 0.0, 0.0, [1, 2], dist_mat=dm)
    ok = (f and s == 'optimal' and r == [0, 1, 2, 0])
    record('dist_mat_threading', ok, f"route={r}, status={s}")


def main():
    print("=== P0-R comparator protocol tests ===")
    test_dead_end_vs_feasible()
    test_permanently_infeasible()
    test_k_boundaries()
    test_nonzero_service_waiting_return()
    test_returned_set_equality_sweep()
    test_service_first_primitives()
    test_empty_assignment_return_tw()
    test_dist_mat_threading()
    test_integration_no_silent_drop()

    out_dir = os.path.join(_CVRPTW, 'results', 'p0r')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'evaluation_protocol_tests.json')
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    n_pass = sum(1 for r in RESULTS if r['status'] == 'PASS')
    n_fail = sum(1 for r in RESULTS if r['status'] == 'FAIL')
    print(f"\n  total={len(RESULTS)} pass={n_pass} fail={n_fail}")
    print(f"  saved: {out_path}")
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
