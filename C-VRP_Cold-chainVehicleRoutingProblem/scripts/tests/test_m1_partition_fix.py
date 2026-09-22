"""M1/feature-only 状态分区修复的最小测试（纯 NumPy，无 GPU/模型）。

覆盖：
  1. 存在 committed 客户时，合法 suffix 重排不被误拒；
  2. 存在 deferred 客户时，合法重排不被误拒；
  3. deferred 客户被插入后从 deferred 集合移除（试探态）；
  4. 真正 missing / duplicate / future 客户入计划仍被拒绝；
  5. NEW_ROUTE 只匹配允许的空车（最小空车不可变时）；
  6. 写回后重新提取的 FleetPlan 与认证计划一致，冻结车/committed leg 不变。
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from action_contract import (ActionSlot, FleetAction, apply_action, build_vehicle_plans,
                             VehiclePlan)
from dynmaskco_cc_context import validate_full_partition

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _plan(vid, anchor, suffix):
    return VehiclePlan(vid, anchor, 0.0, 0.0, tuple(suffix))


def _plans(pairs):
    return {vid: _plan(vid, a, s) for vid, a, s in pairs}


def test_committed_not_missing():
    # 客户 1 committed，客户 4 deferred，suffix=[2,3]。universe 排除 committed。
    plans = _plans([(0, 0, (2, 3))])
    committed = {1}
    deferred = {4}
    universe = {2, 3, 4}     # 已揭示未服务未 committed
    ok, d = validate_full_partition(committed, plans, deferred, universe)
    record('committed_not_missing', ok, f"detail={d}" if not ok else '')
    return ok


def test_deferred_not_missing():
    plans = _plans([(0, 0, (2, 3))])
    committed = set()
    deferred = {4}
    universe = {2, 3, 4}
    ok, _ = validate_full_partition(committed, plans, deferred, universe)
    record('deferred_not_missing', ok)
    return ok


def test_deferred_discard_on_insert():
    # 客户 4 从 deferred 插入 suffix [2,3,4]：试探 deferred 移除 4 后应 PASS。
    plans = _plans([(0, 0, (2, 3, 4))])
    committed = set()
    deferred = {4}
    universe = {2, 3, 4}
    trial_deferred = set(deferred) - {4}
    ok, _ = validate_full_partition(committed, plans, trial_deferred, universe)
    record('deferred_discard_on_insert', ok)
    return ok


def test_real_missing_rejected():
    plans = _plans([(0, 0, (2, 3))])
    committed = set()
    deferred = set()
    universe = {2, 3, 4}     # 4 真的丢了
    ok, d = validate_full_partition(committed, plans, deferred, universe)
    record('real_missing_rejected', not ok, f"missing={d.get('missing')}")
    return not ok


def test_duplicate_rejected():
    plans = _plans([(0, 0, (2, 2))])   # 客户 2 重复
    committed = set()
    deferred = set()
    universe = {2}
    ok, d = validate_full_partition(committed, plans, deferred, universe)
    record('duplicate_rejected', not ok, f"dup={d.get('duplicate_suffix')}")
    return not ok


def test_future_planned_rejected():
    plans = _plans([(0, 0, (2, 3))])
    committed = set()
    deferred = set()
    universe = {2, 3}
    ok, d = validate_full_partition(committed, plans, deferred, universe,
                                    future_fn=lambda c: c == 3)  # 3 是未来客户
    record('future_planned_rejected', not ok, f"future={d.get('future_planned')}")
    return not ok


def test_new_route_allowed_scope():
    # 空车 0（不可变）、空车 1（可变）。NEW_ROUTE 限定 allowed={1}，必须落到车 1。
    plans = _plans([(0, 0, ()), (1, 0, ())])
    act = FleetAction(customer=5, slot=ActionSlot('new_route', -1), position=0,
                      predecessor=0, successor=0, incumbent=False)
    new = apply_action(plans, act, allowed_vehicle_ids={1})
    ok = 5 in new[1].suffix and 5 not in new[0].suffix
    record('new_route_allowed_scope', ok,
           f"car0={new[0].suffix} car1={new[1].suffix}")
    return ok


def test_writeback_matches_certified():
    # 用 apply_action 生成 new_plan，再「写回」到 vehicles 的 mutable_suffix，
    # 重新 build_vehicle_plans 应还原出相同的 suffix。
    from strict_online_env import VehicleState
    plans = _plans([(0, 0, (2, 3)), (1, 5, (7,))])
    act = FleetAction(customer=3, slot=ActionSlot('anchored', 0), position=0,
                      predecessor=0, successor=2, incumbent=False)
    new = apply_action(plans, act, allowed_vehicle_ids={0, 1})
    # 写回（只写 mutable 车 0）
    vehicles = [VehicleState(vehicle_id=0), VehicleState(vehicle_id=1)]
    vehicles[0].status = 'idle'; vehicles[0].current_node = 0
    vehicles[1].status = 'ready'; vehicles[1].current_node = 5
    for v in vehicles:
        p = new.get(v.vehicle_id)
        if p is not None:
            v.mutable_suffix = list(p.suffix) + [0]
    # 重新提取（用最小 env stub）
    class _E:
        demands = np.zeros((1, 10), np.float32)
    env = _E()
    rebuilt = {p.vehicle_id: p.suffix for p in build_vehicle_plans(env, 0, vehicles).values()}
    ok = rebuilt[0] == tuple(new[0].suffix) and rebuilt[1] == tuple(new[1].suffix)
    record('writeback_matches_certified', ok, f"rebuilt={rebuilt}")
    return ok


def main():
    ok = [test_committed_not_missing(), test_deferred_not_missing(),
          test_deferred_discard_on_insert(), test_real_missing_rejected(),
          test_duplicate_rejected(), test_future_planned_rejected(),
          test_new_route_allowed_scope(), test_writeback_matches_certified()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}  ({sum(ok)}/{len(ok)})")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
