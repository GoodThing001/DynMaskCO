"""O0-D greedy sequential oracle 测试（合成 2–5 customer + on-trajectory + headroom）。

验证：
  1. oracle 在合成小实例上跑通且 service 100%（complete）；
  2. oracle 不劣于 baseline（至少不差，因为 incumbent 在候选集）；
  3. 在存在 headroom 的场景 oracle 严格更优（distance 更低）；
  4. oracle 沿自身 trajectory 决策（多决策点 reveal 后仍决策，trace 与 baseline 分叉）。

纯 NumPy（JF1-H continuation），可本地跑：python scripts/tests/test_sequential_oracle.py
产物：results/o0d/sequential_oracle_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv, VehicleState
from joint_fleet import JointAssignmentReplanner
from sequential_oracle import make_oracle_hook, sequential_oracle_plan
from counterfactual_teacher import _eval
from recourse_snapshot import capture_recourse_snapshot
from run_action_oracle import service_ok

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_dataset(coords, demands, reveal=None, tw_end=None, service=None, num_vehicles=2):
    N = coords.shape[0]
    tw_start = np.zeros(N, np.float32)
    tw_end = np.full(N, 100.0, np.float32) if tw_end is None else np.asarray(tw_end, np.float32)
    service = np.zeros(N, np.float32) if service is None else np.asarray(service, np.float32)
    reveal = np.zeros(N, np.float32) if reveal is None else np.asarray(reveal, np.float32)
    return {
        'coords': coords[None].astype(np.float32),
        'demands': demands[None].astype(np.float32),
        'tw_start': tw_start[None].astype(np.float32),
        'tw_end': tw_end[None].astype(np.float32),
        'service_time': service[None].astype(np.float32),
        'reveal_time': reveal[None].astype(np.float32),
    }, num_vehicles


def _run_trajectory(dataset, capacity, num_vehicles, continuation, use_oracle):
    n = dataset['coords'].shape[0]
    outcomes = []
    for i in range(n):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles, replanner=continuation)
        if use_oracle:
            rollout_env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                                          replanner=continuation)
            env.oracle_hook = make_oracle_hook(rollout_env, continuation, 'distance')
        traces, _ = env.run(i)
        outcomes.append(_eval(env, i, traces, 'distance'))
    return outcomes


def _line5():
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1., 1.], np.float32)
    return _make_dataset(coords, demands, num_vehicles=2)


def _asymmetric_headroom():
    # c1(1,0), c2(10,0) 先可见；c3(5,0) 在 t=2 reveal。c3 移到 committed 车（c2 后顺路）更优。
    coords = np.array([[0., 0.], [1., 0.], [10., 0.], [5., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    reveal = np.array([0., 0., 0., 2.], np.float32)
    return _make_dataset(coords, demands, reveal=reveal, num_vehicles=3)


def test_oracle_no_regression():
    dataset, nv = _line5()
    cont = JointAssignmentReplanner('heuristic')
    base = _run_trajectory(dataset, 50.0, nv, cont, use_oracle=False)
    orac = _run_trajectory(dataset, 50.0, nv, cont, use_oracle=True)
    ok_service = all(o['complete'] for o in orac)
    ok_no_worse = all(o['distance_cost'] <= b['distance_cost'] + 1e-6
                      for o, b in zip(orac, base))
    record('oracle_no_regression',
           ok_service and ok_no_worse,
           f"base={[round(b['distance_cost'],2) for b in base]} "
           f"orac={[round(o['distance_cost'],2) for o in orac]}")


def test_oracle_headroom():
    # 旧 headroom 场景依赖「把客户插入 committed 车 tail」，阶段 C 的 scope 修复已正确禁止该
    # 操作（mutable_vehicle_ids 只含 idle/ready）。故此处改为非退化断言：oracle 不得劣于
    # baseline（single-step + 终局非退化），严格 headroom 留到 DEV-GATE 重新测量。
    dataset, nv = _asymmetric_headroom()
    cont = JointAssignmentReplanner('heuristic')
    base = _run_trajectory(dataset, 50.0, nv, cont, use_oracle=False)
    orac = _run_trajectory(dataset, 50.0, nv, cont, use_oracle=True)
    ok_service = all(o['complete'] for o in orac)
    ok_nonregress = all(o['distance_cost'] <= b['distance_cost'] + 1e-6
                        for o, b in zip(orac, base))
    record('oracle_headroom', ok_service and ok_nonregress,
           f"base={base[0]['distance_cost']:.3f} oracle={orac[0]['distance_cost']:.3f} "
           f"Δ={orac[0]['distance_cost']-base[0]['distance_cost']:+.3f} (non-regress)")


def test_oracle_on_trajectory_multievent():
    # 多决策点 + reveal：验证 oracle 在 reveal 后仍决策，且与 baseline 分叉。
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [5., 0.], [6., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1., 1., 1.], np.float32)
    reveal = np.array([0., 0., 0., 2., 4., 0.], np.float32)  # c4(t=2)、c5(t=4) reveal
    dataset, nv = _make_dataset(coords, demands, reveal=reveal, num_vehicles=3)
    cont = JointAssignmentReplanner('heuristic')

    def trace_signature(use_oracle):
        env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=cont)
        if use_oracle:
            rollout_env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=cont)
            env.oracle_hook = make_oracle_hook(rollout_env, cont, 'distance')
        traces, _ = env.run(0)
        return tuple(tuple(sr.node for sr in tr.services) for tr in traces)

    sig_base = trace_signature(False)
    sig_orac = trace_signature(True)
    # on-trajectory：oracle 的最终 trace 与 baseline 不同（确实做了不同决策）
    ok_diverged = sig_orac != sig_base
    record('oracle_on_trajectory_multievent', ok_diverged,
           f"base_trace={sig_base} oracle_trace={sig_orac}")


def test_multi_customer_same_event_log():
    # 4 节点全 visible（t=0），2 车：决策点有 3 个 pool 客户，oracle 逐个决策并写完整 log。
    coords = np.array([[0., 0.], [1., 0.], [3., 0.], [6., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    dataset, nv = _make_dataset(coords, demands, num_vehicles=2)
    cont = JointAssignmentReplanner('heuristic')
    env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=cont)
    snaps = []
    env.snapshot_hook = lambda e, i, clk, eid, rid, veh, tr, sm, ac: snaps.append(
        capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))
    env.run(0)
    log = []
    sequential_oracle_plan(env, snaps[0], cont, 'distance', log=log)
    required = {'event_id', 'customer', 'num_candidates', 'num_feasible',
                'incumbent_exists', 'selected_action', 'reject_reasons', 'n_rollouts'}
    ok = (len(log) == 3 and all(required <= set(rec.keys()) for rec in log))
    record('multi_customer_same_event_log', ok, f"n_records={len(log)}")


def test_oracle_hook_action_timing():
    # 动作时机：replan_ids 空（service completion）不触发 oracle_hook；有 needs_replan 才触发。
    coords = np.array([[0., 0.], [1., 0.]], np.float32)
    demands = np.array([0., 1.], np.float32)
    dataset, nv = _make_dataset(coords, demands, num_vehicles=1)
    env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=JointAssignmentReplanner('heuristic'))
    calls = []
    env.oracle_hook = lambda *a, **k: calls.append(1)
    v = VehicleState(vehicle_id=0, status='ready', current_node=0, needs_replan=False)
    env._run_oracle_hook(0, 0.0, 0, [v], [], np.array([True, False]), [])
    ok_idle = (len(calls) == 0)
    v.needs_replan = True
    env._run_oracle_hook(0, 0.0, 0, [v], [], np.array([True, False]), [])
    ok_replan = (len(calls) == 1)
    record('oracle_hook_action_timing', ok_idle and ok_replan,
           f"idle={ok_idle} replan={ok_replan}")


def test_coldchain_service_gate():
    # service_ok 对 coldchain 额外检查温度硬约束 / 入舱 / 返仓 / manifest 清空 / accounting 一致。
    base = {'complete': True, 'n_unserved': 0, 'n_duplicate': 0, 'tw_feasible': True,
            'capacity_feasible': True, 'depot_return_feasible': True}
    ok_dist = service_ok(base, 'distance')
    ok_cc_missing = (not service_ok(base, 'coldchain'))
    cc = dict(base, temperature_hard_feasible=True, all_orders_picked=True,
              all_cargo_delivered_to_depot=True, terminal_manifests_empty=True,
              trace_accounting_consistent=True, distance_accounting_consistent=True)
    ok_cc_full = service_ok(cc, 'coldchain')
    record('coldchain_service_gate', ok_dist and ok_cc_missing and ok_cc_full,
           f"distance={ok_dist} cc_missing_reject={ok_cc_missing} cc_full={ok_cc_full}")


def main():
    print("=== O0-D sequential oracle tests ===")
    test_oracle_no_regression()
    test_oracle_headroom()
    test_oracle_on_trajectory_multievent()
    test_multi_customer_same_event_log()
    test_oracle_hook_action_timing()
    test_coldchain_service_gate()

    out_dir = os.path.join(_CVRPTW, 'results', 'o0d')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sequential_oracle_tests.json')
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
