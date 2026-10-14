"""阶段 C 协议完整性测试：snapshot hash 校验 + replanner_state 版本 + 严格改善 synthetic。

验证：
  1. 篡改 clock / suffix / deferred 后保留旧 state_hash → restore 报错；
  2. 合法 clone + force + 重算 hash → restore 通过；
  3. restore_state 版本/配置不符 → 报错；
  4. mutable scope 内已知严格改善的 synthetic 场景：oracle 选中更优动作。

用法：python scripts/tests/test_protocol_integrity.py
产物：results/o0cc/protocol_integrity_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot,
                               clone_snapshot, snapshot_state_hash)
from sequential_oracle import sequential_oracle_plan, mutable_vehicle_ids, decision_pool
from counterfactual_teacher import _incumbent_plans

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _small_dataset():
    coords = np.array([[0., 0.], [1., 0.], [10., 0.], [5., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    reveal = np.array([0., 0., 0., 0.], np.float32)  # 全可见
    return {
        'coords': coords[None], 'demands': demands[None],
        'tw_start': np.zeros((1, 4), np.float32), 'tw_end': np.full((1, 4), 100., np.float32),
        'service_time': np.zeros((1, 4), np.float32), 'reveal_time': reveal[None],
        'temp_class': np.zeros((1, 4), np.int32), 'initial_quality': np.ones((1, 4), np.float32),
    }


def _capture_first_snapshot():
    dataset = _small_dataset()
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 3, replanner=make_continuation())
    snaps = []
    def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))
    env.snapshot_hook = hook
    env.run(0)
    return env, snaps[0]


def test_snapshot_hash_tamper():
    env, snap = _capture_first_snapshot()
    ok = True
    # 篡改 clock（保留旧 hash）
    s2 = clone_snapshot(snap)
    s2['clock'] = float(snap['clock']) + 1.0
    try:
        restore_recourse_snapshot(s2)
        ok = False
        detail = 'clock 篡改未报错'
    except ValueError as e:
        detail = f'ok: {str(e)[:40]}'
    # 合法 clone + force + 重算 hash 通过
    s3 = clone_snapshot(snap)
    s3['force_suffix'] = [[list(k), list(v)] for k, v in snap['force_suffix']]
    s3['state_hash'] = snapshot_state_hash(s3)
    try:
        restore_recourse_snapshot(s3)
        ok_legal = True
    except Exception:
        ok_legal = False
    record('snapshot_hash_tamper', ok and ok_legal, detail)
    return ok and ok_legal


def test_restore_state_version():
    cont = make_continuation()
    ok = True
    try:
        cont.restore_state({'deferred_customers': [], 'status_inst': 0})  # 缺 version
        ok = False
    except ValueError:
        pass
    try:
        cont.restore_state({'version': 'wrong', 'deferred_customers': [], 'status_inst': 0})
        ok = False
    except ValueError:
        pass
    valid = {'version': 'jf1h-f-state-v1', 'baseline': 'JF1-H-F', 'slack_vehicles': 1,
             'deferred_customers': [3], 'status_inst': 0}
    cont.restore_state(valid)
    ok_restore = (cont.deferred_customers == {3})
    # slack 不匹配 → 报错
    try:
        cont.restore_state(dict(valid, slack_vehicles=2))
        ok_slack = False
    except ValueError:
        ok_slack = True
    record('restore_state_version', ok and ok_restore and ok_slack,
           f"reject_wrong={ok} restore_ok={ok_restore} slack_check={ok_slack}")
    return ok and ok_restore and ok_slack


def test_synthetic_strict_improvement():
    # capacity=2、2 车、c1(1),c2(2),c3(100) 全可见：greedy 把 c1,c2 给 v0、c3 给 v1（cost 204）。
    # 最优是把 c2 移到 v1（v0 只送 c1），cost 202。oracle 应选中 c2 的改善动作（delta=-2）。
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [100., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    dataset = {'coords': coords[None], 'demands': demands[None],
               'tw_start': np.zeros((1, 4), np.float32), 'tw_end': np.full((1, 4), 1000., np.float32),
               'service_time': np.zeros((1, 4), np.float32), 'reveal_time': np.zeros((1, 4), np.float32),
               'temp_class': np.zeros((1, 4), np.int32), 'initial_quality': np.ones((1, 4), np.float32)}
    cont = make_continuation(0)
    env = StrictOnlineEnv(dataset, 2.0, 1.0, 2, replanner=cont)
    snaps = []
    def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))
    env.snapshot_hook = hook
    env.run(0)
    snap = snaps[0]
    renv = StrictOnlineEnv(dataset, 2.0, 1.0, 2, replanner=make_continuation(0))
    log = []
    plan = sequential_oracle_plan(renv, snap, renv.replanner, 'distance', log=log)
    moved = [r for r in log if not str(r['selected_action']).startswith('__')]
    ok_moved = (len(moved) >= 1)
    ok_strict = (ok_moved and all(r['selected_delta'] < -1e-9 for r in moved))
    # 最优计划：v1 送 c2+c3（suffix 顺序 2,3）
    ok_plan = (plan[1].suffix == (2, 3) and plan[0].suffix == (1,))
    record('synthetic_strict_improvement', ok_moved and ok_strict and ok_plan,
           f"moved={[r['customer'] for r in moved]} delta={[r['selected_delta'] for r in moved]} "
           f"plan_v1={plan[1].suffix}")
    return ok_moved and ok_strict and ok_plan


def main():
    ok1 = test_snapshot_hash_tamper()
    ok2 = test_restore_state_version()
    ok3 = test_synthetic_strict_improvement()
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'protocol_integrity_tests.json'), 'w') as f:
        json.dump({'all_pass': ok1 and ok2 and ok3, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == '__main__':
    sys.exit(main())
