"""
P0-A full-fleet Action Contract v1 测试（02 手册 §P0-A / 进度表 §5）。

覆盖：
  1. anchored slot 稳定身份 = anchor_node（非 physical vehicle id）；
  2. 匿名 NEW_ROUTE + deterministic physical matching；
  3. physical vehicle permutation equivariance；
  4. 枚举所有 insertion position + predecessor/successor + stable action_id；
  5. 从原 route 移除 customer + 安装完整 full-fleet plan（其他车不变）；
  6. ownership exactly once；
  7. current-route TW/capacity/return certificate + reject reason histogram；
  8. incumbent（no-op）永远在候选集且 apply 后 plan 不变；
  9. brute-force false-negative audit。

纯 NumPy，可本地跑：python scripts/tests/test_action_contract.py
产物：results/p0a/action_contract_tests.json
"""
import sys, os, json
from collections import Counter

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv, VehicleState
from action_contract import (ActionSlot, FleetAction, build_vehicle_plans, build_slots,
                             certify_route, apply_action, validate_ownership, plan_hash,
                             enumerate_actions, enumerate_actions_from_plans, find_customer_slot)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_env(coords, demands, tw_start, tw_end, service_time, capacity=50.0,
              num_vehicles=3, reveal_time=None):
    N = coords.shape[0]
    dataset = {
        'coords': np.asarray(coords, np.float32)[None],
        'demands': np.asarray(demands, np.float32)[None],
        'tw_start': np.asarray(tw_start, np.float32)[None],
        'tw_end': np.asarray(tw_end, np.float32)[None],
        'service_time': np.asarray(service_time, np.float32)[None],
        'reveal_time': np.asarray(reveal_time if reveal_time is not None else np.zeros(N),
                                  np.float32)[None],
    }
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles, replanner=None)


def _veh(vid, status='ready', node=0, ready_time=0.0, load=0.0, suffix=None,
         committed_next=None, committed_finish=None):
    v = VehicleState(vehicle_id=vid)
    v.status = status
    v.current_node = node
    v.ready_time = ready_time
    v.current_load = load
    v.mutable_suffix = list(suffix or [])
    v.committed_next = committed_next
    v.committed_finish = committed_finish
    return v


def _line_env(capacity=50.0, num_vehicles=3):
    # 5 节点一条线：depot(0,0), c1..c4 在 x=1..4
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    demands = np.array([0., 1., 1., 1., 1.])
    tw_start = np.zeros(5)
    tw_end = np.full(5, 100.)
    service_time = np.zeros(5)
    return _make_env(coords, demands, tw_start, tw_end, service_time, capacity, num_vehicles)


def test_enumeration_positions_new_route():
    env = _line_env()
    vehicles = [_veh(0, 'ready', node=1, ready_time=1.0, suffix=[3, 4]),
                _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    cands, plans = enumerate_actions(env, 0, vehicles, customer=2)

    # anchored(anchor=1, base=[3,4]) → 3 positions；NEW_ROUTE → 1
    ok_count = (len(cands) == 4)
    pos_anchored = sorted(c.action.position for c in cands if c.action.slot.kind == 'anchored')
    ok_pos = (pos_anchored == [0, 1, 2])
    nr = [c for c in cands if c.action.slot.kind == 'new_route']
    ok_nr = (len(nr) == 1 and nr[0].action.slot.anchor == -1 and nr[0].action.position == 0)
    ok_feas = all(c.feasible for c in cands)

    # incremental_distance 抽查：anchored pos0 = d(1,2)+d(2,3)-d(1,3) = 1+1-2 = 0
    c0 = [c for c in cands if c.action.slot.kind == 'anchored' and c.action.position == 0][0]
    ok_incr = abs(c0.incremental_distance - 0.0) < 1e-6
    record('enumeration_positions_new_route',
           ok_count and ok_pos and ok_nr and ok_feas and ok_incr,
           f"n={len(cands)} pos={pos_anchored} new_route={ok_nr} incr={c0.incremental_distance:.3f}")


def test_incumbent_preserved():
    env = _line_env()
    vehicles = [_veh(0, 'ready', node=1, ready_time=1.0, suffix=[2, 3, 4]),
                _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    plans = build_vehicle_plans(env, 0, vehicles)
    h0 = plan_hash(plans)
    cands, plans2 = enumerate_actions(env, 0, vehicles, customer=2)

    inc = [c for c in cands if c.action.incumbent]
    ok_inc = (len(inc) == 1)
    ok_top = (cands[0].action.incumbent)  # 排序后 incumbent 第一
    if ok_inc:
        a = inc[0].action
        ok_inc = (a.position == 0 and a.predecessor == 1 and a.successor == 3)
        new_plan = apply_action(plans2, a, src_vid=0)
        ok_inc = ok_inc and (plan_hash(new_plan) == h0)
    record('incumbent_preserved', ok_inc and ok_top,
           f"incumbent={ok_inc} top={ok_top}")


def test_physical_vehicle_permutation():
    env = _line_env()
    v0 = _veh(0, 'idle', node=0, ready_time=0.0, suffix=[])
    v1 = _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])
    # 交换 vehicle_id
    v0s = _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])
    v1s = _veh(0, 'idle', node=0, ready_time=0.0, suffix=[])

    slots_a, nr_a, idle_a = build_slots(env, 0, [v0, v1])
    slots_b, nr_b, idle_b = build_slots(env, 0, [v0s, v1s])
    # 语义身份：anchored 空 + NEW_ROUTE 可用（两集合一致）
    ok_slots = (len(slots_a) == 0 and len(slots_b) == 0 and nr_a and nr_b)

    # NEW_ROUTE action 应用到两份 → route multiset 相同（customer=2 进一条 depot 出发 route）
    plans_a = build_vehicle_plans(env, 0, [v0, v1])
    plans_b = build_vehicle_plans(env, 0, [v0s, v1s])
    act = FleetAction(customer=2, slot=ActionSlot('new_route', -1), position=0,
                      predecessor=0, successor=0)
    new_a = apply_action(plans_a, act)
    new_b = apply_action(plans_b, act)
    def multisets(plans):
        c = Counter()
        for p in plans.values():
            for x in p.suffix:
                c[int(x)] += 1
        return c
    ok_mset = (multisets(new_a) == multisets(new_b) == Counter({2: 1}))
    # min-vid deterministic matching：各自选最小 idle 车
    ok_match = (find_customer_slot(new_a, 2) == (0, 0)) and (find_customer_slot(new_b, 2) == (0, 0))
    record('physical_vehicle_permutation', ok_slots and ok_mset and ok_match,
           f"slots={ok_slots} multiset={ok_mset} match={ok_match}")


def test_certificate_reject_histogram():
    # capacity reject
    env = _line_env(capacity=2.0)
    vehicles = [_veh(0, 'ready', node=1, ready_time=1.0, load=1.9, suffix=[3]),
                _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    cands, _ = enumerate_actions(env, 0, vehicles, customer=2)
    reasons = Counter(c.reason for c in cands if not c.feasible)
    ok_cap = (reasons.get('capacity', 0) >= 1)

    # tw reject：customer=4 tw_end 极紧
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    demands = np.array([0., 1., 1., 1., 1.])
    tw_start = np.zeros(5)
    tw_end = np.array([100., 100., 100., 100., 5.])
    env2 = _make_env(coords, demands, tw_start, tw_end, np.zeros(5), 50.0, 3)
    vehicles2 = [_veh(0, 'ready', node=1, ready_time=10.0, suffix=[3]),
                 _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    cands2, _ = enumerate_actions(env2, 0, vehicles2, customer=4)
    reasons2 = Counter(c.reason for c in cands2 if not c.feasible)
    ok_tw = (reasons2.get('tw', 0) >= 1)

    # return reject：depot tw_end 极紧
    tw_end3 = np.array([2.5, 100., 100., 100., 100.])
    env3 = _make_env(coords, demands, tw_start, tw_end3, np.zeros(5), 50.0, 3)
    vehicles3 = [_veh(0, 'ready', node=1, ready_time=1.0, suffix=[2]),
                 _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    cands3, _ = enumerate_actions(env3, 0, vehicles3, customer=2)
    reasons3 = Counter(c.reason for c in cands3 if not c.feasible)
    ok_ret = (reasons3.get('return', 0) >= 1)

    record('certificate_reject_histogram', ok_cap and ok_tw and ok_ret,
           f"capacity={ok_cap} tw={ok_tw} return={ok_ret} "
           f"(cap:{dict(reasons)}, tw:{dict(reasons2)}, ret:{dict(reasons3)})")


def test_ownership_exactly_once():
    env = _line_env()
    # 两辆 ready@customer 车；customer=2 从 v0 移到 v1
    vehicles = [_veh(0, 'ready', node=1, ready_time=1.0, suffix=[2, 3]),
                _veh(1, 'ready', node=4, ready_time=4.0, suffix=[])]
    plans = build_vehicle_plans(env, 0, vehicles)
    act = FleetAction(customer=2, slot=ActionSlot('anchored', 4), position=0,
                      predecessor=4, successor=0)
    new = apply_action(plans, act, src_vid=0)
    ok, dup, missing, extra = validate_ownership(new, [2, 3])
    ok_move = (ok and find_customer_slot(new, 2) == (1, 0) and find_customer_slot(new, 3) == (0, 0))
    record('ownership_exactly_once', ok_move, f"dup={dup} missing={missing} extra={extra}")


def test_committed_anchor():
    env = _line_env()
    # committed 车：anchor = committed_next=2, time=committed_finish=5.0, load += demand[2]
    vehicles = [_veh(0, 'committed', node=1, ready_time=1.0, load=0.0, suffix=[3],
                     committed_next=2, committed_finish=5.0),
                _veh(1, 'idle', node=0, ready_time=0.0, suffix=[])]
    slots, nr, _ = build_slots(env, 0, vehicles)
    ok_anchor = (len(slots) == 1 and slots[0][0].anchor == 2 and slots[0][2].anchor_time == 5.0
                 and abs(slots[0][2].anchor_load - 1.0) < 1e-6)
    cands, _ = enumerate_actions(env, 0, vehicles, customer=4)
    # customer=4 插入 committed 车（anchor=2）的 suffix [3] → positions 0..1
    ok_enum = (len(cands) == 3)  # anchored 2 positions + NEW_ROUTE 1
    record('committed_anchor', ok_anchor and ok_enum, f"anchor={ok_anchor} enum={ok_enum}")


def test_bruteforce_no_false_negative():
    env = _line_env()
    vehicles = [_veh(0, 'ready', node=1, ready_time=1.0, suffix=[3, 4]),
                _veh(1, 'ready', node=2, ready_time=2.0, suffix=[]),
                _veh(2, 'idle', node=0, ready_time=0.0, suffix=[])]
    cands, plans = enumerate_actions(env, 0, vehicles, customer=2)

    # 独立 brute-force：对所有 slot × position 组合重放 certificate
    brute = set()
    anchored, nr, _ = build_slots(env, 0, vehicles)
    src_vid, _ = find_customer_slot(plans, 2)
    for slot, vid, p in anchored:
        base = list(p.suffix)
        if 2 in base:
            base.remove(2)
        for pos in range(len(base) + 1):
            trial = base[:pos] + [2] + base[pos:]
            cert = certify_route(env, 0, p.anchor_node, p.anchor_time, p.anchor_load, trial)
            if cert['feasible']:
                pred = p.anchor_node if pos == 0 else base[pos - 1]
                succ = base[pos] if pos < len(base) else 0
                brute.add(FleetAction(2, slot, pos, pred, succ).action_id())
    if nr:
        cert = certify_route(env, 0, 0, 0.0, 0.0, [2])
        if cert['feasible']:
            brute.add(FleetAction(2, ActionSlot('new_route', -1), 0, 0, 0).action_id())

    enum_feas = {c.action.action_id() for c in cands if c.feasible}
    ok = (enum_feas == brute)
    record('bruteforce_no_false_negative', ok,
           f"enum={len(enum_feas)} brute={len(brute)} diff_extra={enum_feas-brute} diff_miss={brute-enum_feas}")


def test_new_route_scoped_matching():
    """NEW_ROUTE 的 scoped physical matching：枚举/apply 都只在 allowed 空 idle 车里选。"""
    env = _line_env(num_vehicles=6)
    # v0 idle 空路线（不在 allowed），v5 idle 空路线（在 allowed）；allowed={5}
    vehicles = [_veh(0, 'idle', node=0, ready_time=0.0, suffix=[]),
                _veh(5, 'idle', node=0, ready_time=0.0, suffix=[])]
    plans = build_vehicle_plans(env, 0, vehicles)
    cands, _ = enumerate_actions_from_plans(env, 0, plans, customer=2, allowed_vehicle_ids={5})
    nr = [c for c in cands if c.action.slot.kind == 'new_route' and c.feasible]
    if len(nr) != 1:
        record('new_route_scoped_matching', False, f'expected 1 feasible NEW_ROUTE, got {len(nr)}')
        return False
    c = nr[0]
    applied = apply_action(plans, c.action, allowed_vehicle_ids={5})
    ok_v5 = (2 in applied[5].suffix)
    ok_v0 = (2 not in applied[0].suffix)
    ok_hash = (c.plan_hash is not None and c.plan_hash == plan_hash(applied))
    ok = ok_v5 and ok_v0 and ok_hash
    record('new_route_scoped_matching', ok,
           f"v5_written={ok_v5} v0_unchanged={ok_v0} hash_match={ok_hash}")
    return ok


def main():
    print("=== P0-A Action Contract v1 tests ===")
    test_enumeration_positions_new_route()
    test_incumbent_preserved()
    test_physical_vehicle_permutation()
    test_certificate_reject_histogram()
    test_ownership_exactly_once()
    test_committed_anchor()
    test_bruteforce_no_false_negative()
    test_new_route_scoped_matching()

    out_dir = os.path.join(_CVRPTW, 'results', 'p0a')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'action_contract_tests.json')
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
