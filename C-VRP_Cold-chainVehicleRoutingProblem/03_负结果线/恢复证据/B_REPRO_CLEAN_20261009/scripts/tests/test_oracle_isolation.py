"""阶段 C 步骤 1：continuation 状态隔离 + 候选顺序不变性测试。

验证：
  1. baseline 确定性：同一实例跑两次（各自独立 continuation）→ 相同 outcome；
  2. 同一 snapshot + action 重复 rollout → 相同 outcome（无跨分支 deferred 状态串扰）；
  3. candidate 正序/倒序评估 → 相同最优选择（顺序无关）。

纯 NumPy。用法：python scripts/tests/test_oracle_isolation.py
产物：results/o0cc/oracle_isolation_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from sequential_oracle import mutable_vehicle_ids, decision_pool, customer_order_key
from counterfactual_teacher import _incumbent_plans, rollout_action, _eval, _snapshot_with_force
from action_contract import enumerate_actions_from_plans
from recourse_snapshot import capture_recourse_snapshot
from coldchain_contract import default_pilot_contract
from coldchain_evaluator import evaluate_coldchain_trace

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_dataset(coords, demands, reveal=None, tw_end=None, num_vehicles=2, temp=None):
    N = coords.shape[0]
    tw_start = np.zeros(N, np.float32)
    tw_end = np.full(N, 100.0, np.float32) if tw_end is None else np.asarray(tw_end, np.float32)
    reveal = np.zeros(N, np.float32) if reveal is None else np.asarray(reveal, np.float32)
    temp = np.zeros(N, np.int32) if temp is None else np.asarray(temp, np.int32)
    return {
        'coords': np.asarray(coords, np.float32)[None],
        'demands': np.asarray(demands, np.float32)[None],
        'tw_start': tw_start[None],
        'tw_end': tw_end[None],
        'service_time': np.zeros((1, N), np.float32),
        'reveal_time': reveal[None],
        'temp_class': temp[None],
        'initial_quality': np.ones((1, N), np.float32),
    }, num_vehicles


def _run(env, inst_idx):
    traces, served = env.run(inst_idx)
    return evaluate_coldchain_trace(traces, {
        'coords': env.coords[inst_idx], 'tw_start': env.tw_start[inst_idx],
        'tw_end': env.tw_end[inst_idx], 'service_time': env.service_time[inst_idx],
        'demands': env.demands[inst_idx], 'dist_mat': env.dist_mat[inst_idx],
        'temp_class': env.temp_class[inst_idx], 'initial_quality': env.initial_quality[inst_idx],
        'speed': env.tw_speed}, default_pilot_contract())


def test_baseline_determinism():
    dataset, nv = _make_dataset(
        np.array([[0., 0.], [1., 0.], [10., 0.], [5., 0.]], np.float32),
        np.array([0., 1., 1., 1.], np.float32),
        reveal=np.array([0., 0., 0., 2.], np.float32), num_vehicles=3)
    contract = default_pilot_contract()
    outs = []
    for _ in range(2):
        env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=make_continuation(),
                              coldchain_contract=contract)
        outs.append(_run(env, 0))
    a, b = outs
    ok = (a['distance_cost'] == b['distance_cost'] and a['complete'] == b['complete']
          and a['quality_loss'] == b['quality_loss'] and a['energy_kwh'] == b['energy_kwh'])
    record('baseline_determinism', ok, f"dist={a['distance_cost']:.3f} vs {b['distance_cost']:.3f}")
    return ok


def test_candidate_reproducible_and_order_invariant():
    dataset, nv = _make_dataset(
        np.array([[0., 0.], [1., 0.], [10., 0.], [5., 0.]], np.float32),
        np.array([0., 1., 1., 1.], np.float32),
        reveal=np.array([0., 0., 0., 2.], np.float32), num_vehicles=3)
    contract = default_pilot_contract()
    env = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=make_continuation(),
                          coldchain_contract=contract)
    snapshots = []
    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        if {v.vehicle_id for v in veh if v.status in ('idle', 'ready') and v.needs_replan}:
            snapshots.append(capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac))
    env.snapshot_hook = hook
    env.run(0)
    if not snapshots:
        record('candidate_reproducible', False, 'no eligible snapshot')
        return False

    snap = None
    pool = []
    renv = StrictOnlineEnv(dataset, 50.0, 1.0, nv, replanner=make_continuation(),
                           coldchain_contract=contract)
    for s in reversed(snapshots):
        _pool = sorted(decision_pool(s), key=customer_order_key(renv, 0))
        if _pool:
            snap, pool = s, _pool
            break
    if snap is None:
        record('candidate_reproducible', False, 'no snapshot with non-empty pool')
        return False

    inst_idx = 0
    incumbent = _incumbent_plans(renv, snap, renv.replanner)
    mutable_ids = mutable_vehicle_ids(snap)
    customer = pool[0]
    cands, _ = enumerate_actions_from_plans(renv, inst_idx, incumbent, customer,
                                            allowed_vehicle_ids=mutable_ids)
    feasible = [c for c in cands if c.feasible]

    # 1. 同一 action 重复评估结果相同
    reproducible = True
    for c in feasible[:3]:
        o1, h1 = rollout_action(renv, snap, c.action, renv.replanner, incumbent,
                                objective='coldchain', allowed_vehicle_ids=mutable_ids,
                                mutable_ids=mutable_ids)
        o2, h2 = rollout_action(renv, snap, c.action, renv.replanner, incumbent,
                                objective='coldchain', allowed_vehicle_ids=mutable_ids,
                                mutable_ids=mutable_ids)
        if o1['coldchain_cost'] != o2['coldchain_cost'] or h1 != h2:
            reproducible = False
    record('candidate_reproducible', reproducible, f'{len(feasible)} feasible, re-eval identical')

    # 2. 顺序无关：正序/倒序评估 → 相同最优 action（cost + action_id 稳定 tie-break）
    def best_action(order):
        best_key = None
        best_a = None
        for c in order:
            o, _ = rollout_action(renv, snap, c.action, renv.replanner, incumbent,
                                  objective='coldchain', allowed_vehicle_ids=mutable_ids,
                                  mutable_ids=mutable_ids)
            key = (o['coldchain_cost'], c.action.action_id())
            if best_key is None or key < best_key:
                best_key, best_a = key, c.action.action_id()
        return best_a
    fwd = best_action(feasible)
    rev = best_action(list(reversed(feasible)))
    order_invariant = (fwd == rev)
    record('candidate_order_invariant', order_invariant, f"fwd={fwd} rev={rev}")
    return reproducible and order_invariant


def main():
    ok1 = test_baseline_determinism()
    ok2 = test_candidate_reproducible_and_order_invariant()
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'oracle_isolation_tests.json'), 'w') as f:
        json.dump({'all_pass': ok1 and ok2, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if (ok1 and ok2) else 'FAIL'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
