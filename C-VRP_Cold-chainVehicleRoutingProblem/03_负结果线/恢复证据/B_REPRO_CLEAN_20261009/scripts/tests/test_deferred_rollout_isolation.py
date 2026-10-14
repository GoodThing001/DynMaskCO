"""阶段 C：deferred 非空 snapshot 的 A→B→A 实际 coldchain rollout 隔离测试。

验证：同一 deferred snapshot 上，分支 A（baseline rollout）→ 分支 B（候选 rollout）→ 再分支 A，
两次 A 的终局 outcome 完全一致（候选分支的 deferred/审计状态不污染后续分支）。

用法：python scripts/tests/test_deferred_rollout_isolation.py
产物：results/o0cc/deferred_rollout_isolation_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from recourse_snapshot import capture_recourse_snapshot
from sequential_oracle import decision_pool, mutable_vehicle_ids
from counterfactual_teacher import rollout_baseline, _incumbent_plans, rollout_action
from action_contract import enumerate_actions_from_plans
from coldchain_contract import default_pilot_contract
from hard_gate import hard_vector_from_outcome

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    dataset = dict(np.load(data_path))
    contract = default_pilot_contract()

    rp = make_continuation()
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 25, replanner=rp, coldchain_contract=contract)
    snaps = []

    def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))

    env.snapshot_hook = hook
    env.run(0)

    deferred_snaps = [s for s in snaps
                      if s.get('replanner_state') and s['replanner_state'].get('deferred_customers')]
    if not deferred_snaps:
        record('deferred_rollout_isolation', False, '未找到 deferred 非空 snapshot')
        out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'deferred_rollout_isolation_tests.json'), 'w') as f:
            json.dump({'all_pass': False, 'results': RESULTS}, f, indent=2)
        return 1
    snap = deferred_snaps[0]

    # 复用同一个 rollout env + continuation（生产候选循环的对象复用模式）
    renv = StrictOnlineEnv(dataset, 50.0, 1.0, 25, replanner=make_continuation(),
                           coldchain_contract=contract)

    def branch_a():
        out = rollout_baseline(renv, snap, 'coldchain')
        # 比较完整 hard vector + D/Q/E + 恢复后决策状态 + 分支审计
        return {
            'outcome': (out['coldchain_cost'], out['distance_km'], out['quality_loss'],
                        out['energy_kwh']),
            'hard_vector': hard_vector_from_outcome(out),
            'repair': (out.get('ownership_violations', -1), out.get('terminal_unresolved', -1)),
            'decision_state': sorted(int(c) for c in renv.replanner.deferred_customers),
            'audit_attempts': int(renv.replanner.repair_stats.get('attempts', -1)),
        }

    # 分支 B：从 incumbent 枚举一个可行的非 no-op 候选 action
    incumbent = _incumbent_plans(renv, snap, renv.replanner)
    mutable_ids = mutable_vehicle_ids(snap)
    pool = decision_pool(snap)
    action = None
    for customer in pool:
        cands, _ = enumerate_actions_from_plans(renv, 0, incumbent, customer,
                                                allowed_vehicle_ids=mutable_ids)
        feas = [c for c in cands if c.feasible and not c.action.incumbent]
        if feas:
            action = feas[0].action
            break
    if action is None:
        record('deferred_rollout_isolation', False, '未找到可行的非 no-op 候选 action')
        out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'deferred_rollout_isolation_tests.json'), 'w') as f:
            json.dump({'all_pass': False, 'results': RESULTS}, f, indent=2)
        return 1

    a1 = branch_a()
    # B：同一 renv 上执行候选 rollout（restore 自同一 snapshot）
    b_outcome, _ = rollout_action(renv, snap, action, renv.replanner, incumbent,
                                  objective='coldchain', allowed_vehicle_ids=mutable_ids,
                                  mutable_ids=mutable_ids)
    a2 = branch_a()

    ok = (a1 == a2)
    record('deferred_rollout_isolation', ok,
           f"a1==a2={ok} b_cost={b_outcome['coldchain_cost']:.4f} "
           f"a1_hard_ok={a1['hard_vector']==a2['hard_vector']} "
           f"a1_repair={a1['repair']} a1_state={a1['decision_state']}")
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'deferred_rollout_isolation_tests.json'), 'w') as f:
        json.dump({'all_pass': ok, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
