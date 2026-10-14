"""
P0-S RecourseSnapshotV2 round-trip parity 测试（02 手册 §P0-S4 / Gate S）。

验证：
  1. 真实 VAL 数据 3 实例 × 全部决策点：capture → resume 后缀 trace/cost exact parity；
  2. 六类事件覆盖（initial / reveal / plan_exhaustion / committed / WAIT / returning），
     WAIT/returning/exhaustion 用合成场景保证出现；
  3. asymmetric dist_mat 实例 parity；
  4. clone 深拷贝独立性 + state_hash 确定性 + hash 对 mutable_suffix 顺序敏感；
  5. restore 后 fleet/trace 与 snapshot 逐字段相等。

纯 NumPy，可本地跑：python scripts/tests/test_recourse_snapshot.py
产物：results/p0s/snapshot_roundtrip_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv, GreedyReplanner
from authoritative_evaluator import evaluate_execution_trace
from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot,
                               clone_snapshot, snapshot_state_hash, resume_from_snapshot,
                               SNAPSHOT_SCHEMA_VERSION)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _eq(a, b):
    """支持 None/nan 的精确相等。"""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
    return a == b


def _norm(x):
    """把 nan 归一化为 None（snapshot 用 nan 编码 None，restore 还原为 None）。"""
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def traces_equal(t1, t2):
    if len(t1) != len(t2):
        return False
    for a, b in zip(t1, t2):
        if a.vehicle_id != b.vehicle_id:
            return False
        if len(a.services) != len(b.services):
            return False
        for sa, sb in zip(a.services, b.services):
            for f in ('vehicle_id', 'prev_node', 'node', 'depart_time', 'arrival_time',
                      'service_start', 'service_finish'):
                if not _eq(getattr(sa, f), getattr(sb, f)):
                    return False
        for f in ('dispatch_time', 'return_depart', 'return_arrival'):
            if not _eq(getattr(a, f), getattr(b, f)):
                return False
    return True


def run_with_hooks(dataset, replanner, inst_idx, capacity=50, tw_speed=1.0, num_vehicles=25,
                   force_suffix=None):
    snaps = []

    def hook(env, i, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers):
        snaps.append(capture_recourse_snapshot(env, i, clock, event_id, reveal_idx,
                                               vehicles, traces, served_mask, all_customers))

    env = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles, replanner=replanner)
    env.snapshot_hook = hook
    traces, served = env.run(inst_idx, force_suffix=force_suffix)
    return env, traces, served, snaps


def resume_and_check(dataset, snap, replanner, traces_full, served_full,
                     capacity=50, tw_speed=1.0, num_vehicles=25):
    """从 snapshot 恢复并断言后缀 trace / served_mask / evaluator 指标 exact parity。"""
    env2 = resume_from_snapshot(dataset, snap, replanner, capacity=capacity,
                                tw_speed=tw_speed, num_vehicles=num_vehicles)
    traces2, served2 = env2.run_resumed(snap)
    if not (traces_equal(traces_full, traces2) and np.array_equal(served_full, served2)):
        return False
    # belt-and-braces：权威 evaluator 输出也必须一致
    i = int(snap['instance_id'])
    m1 = evaluate_execution_trace(traces_full, env2.coords[i], env2.tw_start[i], env2.tw_end[i],
                                  env2.service_time[i], env2.demands[i], capacity, speed=tw_speed,
                                  dist_mat=env2.dist_mat[i])
    m2 = evaluate_execution_trace(traces2, env2.coords[i], env2.tw_start[i], env2.tw_end[i],
                                  env2.service_time[i], env2.demands[i], capacity, speed=tw_speed,
                                  dist_mat=env2.dist_mat[i])
    return m1 == m2


def classify(snap):
    classes = []
    if snap['event_id'] == 0:
        classes.append('initial')
    if any(r == 'reveal' for r in snap['replan_reason']):
        classes.append('reveal')
    if any(r == 'plan_exhaustion' for r in snap['replan_reason']):
        classes.append('exhaustion')
    if any(s == 'committed' for s in snap['vehicle_status']):
        classes.append('committed')
    if any(s == 'returning' for s in snap['vehicle_status']):
        classes.append('returning')
    for k in range(snap['num_vehicles']):
        if (snap['vehicle_status'][k] == 'ready' and int(snap['vehicle_node'][k]) != 0
                and int(snap['mutable_suffix_len'][k]) == 0):
            classes.append('wait')
            break
    return classes


def test_roundtrip_parity_real():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    if not os.path.exists(data_path):
        record('roundtrip_parity_real', True, 'SKIP: data not found')
        return
    dataset = dict(np.load(data_path))
    seen = {}
    fails = 0
    n_snaps = 0
    for i in range(3):
        env, traces_full, served_full, snaps = run_with_hooks(dataset, GreedyReplanner('nn'), i)
        for s in snaps:
            n_snaps += 1
            for c in classify(s):
                seen[c] = seen.get(c, 0) + 1
            if not resume_and_check(dataset, s, GreedyReplanner('nn'), traces_full, served_full):
                fails += 1
                if fails <= 2:
                    record('roundtrip_parity_real', False,
                           f"inst {i} event {s['event_id']} resume mismatch")
    ok = fails == 0
    record('roundtrip_parity_real', ok, f"snapshots={n_snaps} fails={fails}")
    required = {'initial', 'reveal', 'committed'}
    missing = required - set(seen)
    record('event_class_coverage_real', not missing,
           f"missing={missing or 'none'}, seen={seen}")


def test_asymmetric_parity():
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.], [0., 1.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, 4), dtype=np.float32)
    tw_end = np.array([[100., 100., 100., 100.]], dtype=np.float32)
    service_time = np.zeros((1, 4), dtype=np.float32)
    reveal_time = np.array([[0., 0., 0., 5.]], dtype=np.float32)   # node3 未来揭示
    dist_mat = np.array([[[0., 1., 2., 1.],
                          [2., 0., 1., 2.],
                          [3., 2., 0., 1.],
                          [1., 3., 2., 0.]]], dtype=np.float32)   # 非对称
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
               'service_time': service_time, 'reveal_time': reveal_time, 'dist_mat': dist_mat}
    env, traces_full, served_full, snaps = run_with_hooks(dataset, GreedyReplanner('nn'), 0,
                                                          capacity=50, tw_speed=1.0, num_vehicles=2)
    fails = sum(1 for s in snaps
                if not resume_and_check(dataset, s, GreedyReplanner('nn'), traces_full,
                                        served_full, capacity=50, tw_speed=1.0, num_vehicles=2))
    record('asymmetric_parity', fails == 0, f"snapshots={len(snaps)} fails={fails}")


def test_wait_returning_scenario():
    # 单车辆：A(1,0) 先服务；F1(8,0) reveal 0.5 且从 A 不可达（tw_end 11 < 5+7）→ 在 reveal 后
    # 贪心无可行客户 → WAIT（suffix=[]）；F2(9,0) reveal 10 可服务；F3 reveal 20 制造 returning 快照。
    coords = np.array([[[0., 0.], [1., 0.], [8., 0.], [9., 0.], [9.5, 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, 5), dtype=np.float32)
    tw_end = np.array([[100., 100., 11., 100., 100.]], dtype=np.float32)
    service_time = np.zeros((1, 5), dtype=np.float32)
    reveal_time = np.array([[0., 0., 0.5, 10., 20.]], dtype=np.float32)
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
               'service_time': service_time, 'reveal_time': reveal_time}
    env, traces_full, served_full, snaps = run_with_hooks(dataset, GreedyReplanner('nn'), 0,
                                                          capacity=50, tw_speed=1.0, num_vehicles=1)
    seen = {}
    fails = 0
    for s in snaps:
        for c in classify(s):
            seen[c] = seen.get(c, 0) + 1
        if not resume_and_check(dataset, s, GreedyReplanner('nn'), traces_full, served_full,
                                capacity=50, tw_speed=1.0, num_vehicles=1):
            fails += 1
    ok_classes = ('wait' in seen and 'returning' in seen and 'committed' in seen
                  and 'reveal' in seen)
    record('wait_returning_scenario', fails == 0 and ok_classes,
           f"snapshots={len(snaps)} fails={fails} seen={seen}")


def test_exhaustion_scenario():
    # 两客户 B(1,0)、C(2,0)，force_suffix 在 event 1 强制 suffix=[2]（无尾部 0）→
    # commit C 后 suffix=[] → C 完成时触发 plan_exhaustion。
    coords = np.array([[[0., 0.], [1., 0.], [2., 0.]]], dtype=np.float32)
    demands = np.array([[0., 1., 1.]], dtype=np.float32)
    tw_start = np.zeros((1, 3), dtype=np.float32)
    tw_end = np.array([[100., 100., 100.]], dtype=np.float32)
    service_time = np.zeros((1, 3), dtype=np.float32)
    reveal_time = np.zeros((1, 3), dtype=np.float32)
    dataset = {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
               'service_time': service_time, 'reveal_time': reveal_time}
    env, traces_full, served_full, snaps = run_with_hooks(dataset, GreedyReplanner('nn'), 0,
                                                          capacity=50, tw_speed=1.0,
                                                          num_vehicles=1,
                                                          force_suffix={(1, 0): [2]})
    seen = {}
    fails = 0
    for s in snaps:
        for c in classify(s):
            seen[c] = seen.get(c, 0) + 1
        if not resume_and_check(dataset, s, GreedyReplanner('nn'), traces_full, served_full,
                                capacity=50, tw_speed=1.0, num_vehicles=1):
            fails += 1
    record('exhaustion_scenario', fails == 0 and 'exhaustion' in seen,
           f"snapshots={len(snaps)} fails={fails} seen={seen}")


def test_clone_and_hash():
    base = {
        'schema_version': SNAPSHOT_SCHEMA_VERSION,
        'instance_id': 0, 'event_id': 1, 'replan_count': 1, 'clock': 2.5, 'reveal_idx': 1,
        'num_vehicles': 1, 'num_nodes': 6,
        'served_mask': np.array([True, True, False, False, False, False]),
        'visible_mask': np.array([True, True, True, True, False, False]),
        'customer_universe': np.array([1, 2, 3, 4, 5], dtype=np.int32),
        'vehicle_status': ['ready'],
        'vehicle_node': np.array([0], np.int32),
        'vehicle_ready': np.array([2.5]),
        'vehicle_load': np.array([0.]),
        'committed_next': np.array([-1], np.int32),
        'committed_arrive': np.array([np.nan]),
        'committed_finish': np.array([np.nan]),
        'needs_replan': np.array([False]),
        'replan_reason': [None],
        'mutable_suffix': np.array([[3, 4, 5]], np.int32),
        'mutable_suffix_len': np.array([3], np.int32),
        'served_route': np.array([[1]], np.int32),
        'served_route_len': np.array([1], np.int32),
        'dispatch_time': np.array([0.]),
        'return_finish': np.array([np.nan]),
        'trace_dispatch_time': np.array([0.]),
        'trace_return_depart': np.array([np.nan]),
        'trace_return_arrival': np.array([np.nan]),
        'services_prev_node': np.array([[0]], np.int32),
        'services_node': np.array([[1]], np.int32),
        'services_depart': np.array([[0.]]),
        'services_arrival': np.array([[1.]]),
        'services_start': np.array([[1.]]),
        'services_finish': np.array([[1.]]),
        'services_len': np.array([1], np.int32),
        'force_suffix': [],
    }
    base['state_hash'] = snapshot_state_hash(base)
    h1 = base['state_hash']

    c = clone_snapshot(base)
    ok_det = (snapshot_state_hash(c) == h1)

    # 顺序敏感：同集合不同顺序 → hash 变（关键：hash 对 route 顺序敏感）
    c1 = clone_snapshot(base)
    c1['mutable_suffix'] = np.array([[4, 3, 5]], np.int32)
    ok_order = (snapshot_state_hash(c1) != h1)
    c2 = clone_snapshot(base)
    c2['mutable_suffix'] = np.array([[5, 3, 4]], np.int32)
    ok_order2 = (snapshot_state_hash(c2) != h1)

    # 深拷贝独立性：修改 clone 不影响原 snapshot
    c3 = clone_snapshot(base)
    c3['mutable_suffix'][0, 0] = 999
    ok_indep = (base['mutable_suffix'][0, 0] == 3)

    record('clone_and_hash', ok_det and ok_order and ok_order2 and ok_indep,
           f"deterministic={ok_det} order_sensitive={ok_order and ok_order2} independent={ok_indep}")


def test_restore_field_equality():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    if not os.path.exists(data_path):
        record('restore_field_equality', True, 'SKIP: data not found')
        return
    dataset = dict(np.load(data_path))
    # 取一个非初始、含服务记录的 snapshot（event_id 最大者，状态最丰富）
    env, traces, served, snaps = run_with_hooks(dataset, GreedyReplanner('nn'), 0)
    s = max(snaps, key=lambda x: x['event_id'])
    vehicles, traces2, served_mask = restore_recourse_snapshot(s)
    ok = True
    for k, v in enumerate(vehicles):
        ok &= (v.status == s['vehicle_status'][k])
        ok &= (v.current_node == int(s['vehicle_node'][k]))
        ok &= _eq(_norm(v.ready_time), _norm(float(s['vehicle_ready'][k])))
        ok &= _eq(_norm(v.current_load), _norm(float(s['vehicle_load'][k])))
        cn = -1 if v.committed_next is None else v.committed_next
        ok &= (cn == int(s['committed_next'][k]))
        ok &= _eq(_norm(v.committed_arrive), _norm(float(s['committed_arrive'][k])))
        ok &= _eq(_norm(v.committed_finish), _norm(float(s['committed_finish'][k])))
        ok &= (v.needs_replan == bool(s['needs_replan'][k]))
        ok &= (v.replan_reason == s['replan_reason'][k])
        lt = int(s['mutable_suffix_len'][k])
        ok &= (v.mutable_suffix == [int(x) for x in s['mutable_suffix'][k][:lt]])
        ls = int(s['served_route_len'][k])
        ok &= (v.served_route == [int(x) for x in s['served_route'][k][:ls]])
        ok &= _eq(_norm(v.dispatch_time), _norm(float(s['dispatch_time'][k])))
        ok &= _eq(_norm(v.return_finish), _norm(float(s['return_finish'][k])))
    for k, tr in enumerate(traces2):
        ok &= _eq(_norm(tr.dispatch_time), _norm(float(s['trace_dispatch_time'][k])))
        ok &= _eq(_norm(tr.return_depart), _norm(float(s['trace_return_depart'][k])))
        ok &= _eq(_norm(tr.return_arrival), _norm(float(s['trace_return_arrival'][k])))
        ok &= (len(tr.services) == int(s['services_len'][k]))
        for si, sr in enumerate(tr.services):
            ok &= (sr.prev_node == int(s['services_prev_node'][k][si]))
            ok &= (sr.node == int(s['services_node'][k][si]))
            ok &= _eq(_norm(sr.depart_time), _norm(float(s['services_depart'][k][si])))
            ok &= _eq(_norm(sr.arrival_time), _norm(float(s['services_arrival'][k][si])))
            ok &= _eq(_norm(sr.service_start), _norm(float(s['services_start'][k][si])))
            ok &= _eq(_norm(sr.service_finish), _norm(float(s['services_finish'][k][si])))
    ok &= np.array_equal(served_mask, s['served_mask'])
    record('restore_field_equality', ok, f"snapshot event_id={s['event_id']}")


def main():
    print("=== P0-S RecourseSnapshotV2 round-trip tests ===")
    test_roundtrip_parity_real()
    test_asymmetric_parity()
    test_wait_returning_scenario()
    test_exhaustion_scenario()
    test_clone_and_hash()
    test_restore_field_equality()

    out_dir = os.path.join(_CVRPTW, 'results', 'p0s')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'snapshot_roundtrip_tests.json')
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
