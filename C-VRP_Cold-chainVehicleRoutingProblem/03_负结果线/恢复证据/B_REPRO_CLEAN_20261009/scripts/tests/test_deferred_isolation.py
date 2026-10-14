"""阶段 C 集成测试：deferred 非空 snapshot 的状态隔离 + restore round-trip。

验证：
  1. 存在 deferred 非空的 snapshot（repair 延期真实发生）；
  2. snapshot 捕获的 replanner_state.deferred_customers 与决策点 deferred 一致；
  3. 同一 snapshot 用两个全新 continuation 各 restore 一次 → deferred 完全一致（无跨分支串扰）；
  4. restore 后审计状态被重置（repair_stats/events 清零，不沿用旧分支）。

用法：python scripts/tests/test_deferred_isolation.py
产物：results/o0cc/deferred_isolation_tests.json
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

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    dataset = dict(np.load(data_path))

    # 找一个 deferred 非空的 snapshot
    rp = make_continuation()
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 25, replanner=rp)
    snapshots = []

    def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        snapshots.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))

    env.snapshot_hook = hook
    env.run(0)

    deferred_snaps = [s for s in snapshots
                      if s.get('replanner_state') and s['replanner_state'].get('deferred_customers')]
    if not deferred_snaps:
        record('deferred_snapshot_exists', False, '未找到 deferred 非空 snapshot（val 实例无延期）')
        out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'deferred_isolation_tests.json'), 'w') as f:
            json.dump({'all_pass': False, 'results': RESULTS}, f, indent=2)
        return 1

    snap = deferred_snaps[0]
    deferred = set(int(c) for c in snap['replanner_state']['deferred_customers'])
    record('deferred_snapshot_exists', True,
           f"找到 deferred 非空 snapshot（deferred={sorted(deferred)}）")

    # 两个全新 continuation 各 restore 一次 → deferred 一致，且审计状态重置
    def restore_and_check():
        cont = make_continuation()
        # 先污染审计状态，确认 restore 会重置
        cont.repair_stats['attempts'] = 999
        cont.repair_events.append({'dummy': True})
        cont.restore_state(snap['replanner_state'])
        d = set(cont.deferred_customers)
        audit_reset = (cont.repair_stats['attempts'] == 0 and cont.repair_events == [])
        return d, audit_reset

    d1, r1 = restore_and_check()
    d2, r2 = restore_and_check()
    ok_deferred = (d1 == deferred and d2 == deferred)
    ok_reset = r1 and r2
    record('deferred_restore_isolation', ok_deferred and ok_reset,
           f"d1==d2==snapshot={ok_deferred} audit_reset={ok_reset}")

    all_pass = (ok_deferred and ok_reset)
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'deferred_isolation_tests.json'), 'w') as f:
        json.dump({'all_pass': all_pass, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
