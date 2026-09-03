"""
P0-U counterfactual utility teacher 测试（02 手册 §P0-U / 进度表 §6 / Gate U）。

覆盖：
  1. 相同 state/action 重复 rollout → outcome/plan_hash 完全相同（deterministic）；
  2. incumbent（no-op）action 的 outcome == baseline rollout outcome（delta=0）；
  3. force_plan_hash 可追溯（incumbent hash == incumbent plan hash；non-incumbent 不同）；
  4. lexicographic delta_cost 有区分度（存在更优/更差的 feasible counterfactual）。

纯 NumPy（JF1-H continuation），可本地跑：python scripts/tests/test_counterfactual_teacher.py
产物：results/p0u/counterfactual_teacher_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from joint_fleet import JointAssignmentReplanner
from recourse_snapshot import capture_recourse_snapshot, restore_recourse_snapshot
from action_contract import enumerate_actions
from counterfactual_teacher import (rollout_baseline, rollout_action,
                                    enumerate_counterfactuals, lex_key)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_env_and_snapshot():
    # 4 节点：depot(0,0), c1(1,0), c2(10,0), c3(5,0)。c3 在 t=2 reveal；3 辆车。
    # t=2：车A ready@c1(anchor=1)、车B ready@c2(anchor=10)、车C idle。
    # continuation 把 c3 分到车A（dist 4 最小）；移 c3 到车B（anchor=10）更优、到 NEW_ROUTE（车C）更贵
    # → delta_cost 有正有负。
    coords = np.array([[0., 0.], [1., 0.], [10., 0.], [5., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    tw_start = np.zeros(4, np.float32)
    tw_end = np.full(4, 100., np.float32)
    service = np.zeros(4, np.float32)
    reveal = np.array([0., 0., 0., 2.], np.float32)
    dataset = {'coords': coords[None], 'demands': demands[None], 'tw_start': tw_start[None],
               'tw_end': tw_end[None], 'service_time': service[None], 'reveal_time': reveal[None]}
    cont = JointAssignmentReplanner('heuristic')
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 3, replanner=cont)
    snaps = []

    def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))

    env.snapshot_hook = hook
    env.run(0)
    snap = next(s for s in snaps if float(s['clock']) >= 2.0 - 1e-6
                and any(int(n) != 0 for n in s['vehicle_node']))
    return env, cont, snap


def _incumbent_actions(env, snap, cont, customer):
    """在决策点跑 continuation 得到 incumbent vehicles，再枚举 customer 的 action（返回 action 对象）。"""
    inst_idx = int(snap['instance_id'])
    clock = float(snap['clock'])
    served = snap['served_mask']
    visible = [int(c) for c in snap['customer_universe'] if bool(snap['visible_mask'][int(c)])]
    vehicles, _, _ = restore_recourse_snapshot(snap)
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    cont.plan(env, inst_idx, clock, vehicles, served, visible, replan_ids=replan_ids)
    cands, _ = enumerate_actions(env, inst_idx, vehicles, customer)
    return cands


def test_deterministic_repeat():
    env, cont, snap = _make_env_and_snapshot()
    cands = _incumbent_actions(env, snap, cont, customer=3)
    feas = [c for c in cands if c.feasible]
    ok = True
    for c in feas[:3]:  # 抽 3 个 action 各跑两次
        o1, h1 = rollout_action(env, snap, c.action, cont)
        o2, h2 = rollout_action(env, snap, c.action, cont)
        ok = ok and (o1 == o2 and h1 == h2)
    record('deterministic_repeat', ok, f"tested {min(3, len(feas))} actions")


def test_incumbent_delta_zero():
    env, cont, snap = _make_env_and_snapshot()
    rows, baseline = enumerate_counterfactuals(env, snap, cont, customer=3)
    inc = [r for r in rows if r['incumbent']]
    ok = (len(inc) == 1)
    if ok:
        r = inc[0]
        ok = (r['delta_lex'] == 0 and abs(r['delta_cost']) < 1e-9
              and lex_key(r['outcome']) == lex_key(baseline))
    record('incumbent_delta_zero', ok,
           f"incumbent={len(inc)} baseline_cost={baseline['distance_cost']:.3f}")


def test_force_plan_hash_traceability():
    env, cont, snap = _make_env_and_snapshot()
    rows, baseline = enumerate_counterfactuals(env, snap, cont, customer=3)
    feas = [r for r in rows if r['feasible']]
    inc = [r for r in feas if r['incumbent']]
    noninc = [r for r in feas if not r['incumbent']]
    ok_hash = all(isinstance(r['force_plan_hash'], str) and r['force_plan_hash'] for r in feas)
    # 存在 non-incumbent 且其 plan_hash != incumbent（action 真实改变了 full-fleet plan）
    ok_diff = (inc and noninc
               and any(r['force_plan_hash'] != inc[0]['force_plan_hash'] for r in noninc))
    ok_id = len({r['action_id'] for r in rows}) == len(rows)
    record('force_plan_hash_traceability', ok_hash and ok_diff and ok_id,
           f"feasible={len(feas)} hash_ok={ok_hash} diff={ok_diff} id_unique={ok_id}")


def test_lexicographic_delta_cost():
    env, cont, snap = _make_env_and_snapshot()
    rows, baseline = enumerate_counterfactuals(env, snap, cont, customer=3)
    feas = [r for r in rows if r['feasible'] and not r['incumbent']]
    deltas = [r['delta_cost'] for r in feas]
    # counterfactual 有区分度：存在 non-incumbent feasible 且 delta_cost 非零
    ok = (len(feas) >= 1 and any(abs(d) > 1e-9 for d in deltas))
    record('lexicographic_delta_cost', ok,
           f"n={len(feas)} deltas={[round(d,2) for d in deltas]} "
           f"baseline_cost={baseline['distance_cost']:.1f}")


def main():
    print("=== P0-U counterfactual teacher tests ===")
    test_deterministic_repeat()
    test_incumbent_delta_zero()
    test_force_plan_hash_traceability()
    test_lexicographic_delta_cost()

    out_dir = os.path.join(_CVRPTW, 'results', 'p0u')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'counterfactual_teacher_tests.json')
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
