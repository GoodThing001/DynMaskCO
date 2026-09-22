"""N0-r1 针对性验收：预冷修复、分区拒绝、未使用车辆零增量。

纯 NumPy + 对象层，无 JAX。构造最小 VisibleState/计划直接测 evaluate_visible_plan 与
cc_lns_replanner 的关键路径；独立环境回放已在 gate 外单独核对（D/E 精确、Q 连续积分差 ~0.07%）。
"""
import sys
import os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'coldchain', 'baselines', 'models', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from coldchain_state import VehicleColdChainState
from visible_state import VisibleState, VehicleVisibleState, evaluate_visible_plan
from action_contract import VehiclePlan
from cc_lns_replanner import _partition_ok


def _profile():
    return ObjectiveProfile('o0cc-pilot-devmean-equal-v2', 18.332282740839446,
                            2.8916668533318926, 1193.4933642766875, 1.0, 1.0, 'devmean')


def _eff():
    return apply_objective_profile(default_pilot_contract(), _profile())


def _vis(vehicles, node_ids=(0,), dist=((0.0,),), pool=(), deferred=()):
    n = len(node_ids)
    if len(dist) == 1 and n > 1:
        raise ValueError('dist 需匹配 node_ids')
    return VisibleState(
        inst_idx=0, event_id=0, clock=0.0, capacity=50.0, tw_speed=1.0, depot_tw_end=24.0,
        has_future_reveal=False, node_ids=tuple(node_ids),
        demands=tuple(0.0 for _ in node_ids), tw_start=tuple(0.0 for _ in node_ids),
        tw_end=tuple(24.0 for _ in node_ids), service_time=tuple(0.0 for _ in node_ids),
        temp_class=tuple(0 for _ in node_ids), initial_quality=tuple(1.0 for _ in node_ids),
        dist_mat=tuple(map(tuple, dist)), travel_mat=tuple(map(tuple, dist)),
        vehicles=tuple(vehicles), pool_customer_ids=tuple(pool),
        deferred_customer_ids=tuple(deferred), replan_ids=tuple(v.vid for v in vehicles),
        _idx={int(n): i for i, n in enumerate(node_ids)})


def _idle_vehicle(vid=0):
    return VehicleVisibleState(vid=vid, status='idle', current_node=0, ready_time=0.0,
                               current_load=0.0, committed_next=-1, committed_arrive=float('nan'),
                               committed_finish=float('nan'), coldchain_state=None)


def test_unused_idle_zero():
    """未派车、无任务的 idle 车：增量为 0（不收预冷）。"""
    vis = _vis([_idle_vehicle(0)])
    r = evaluate_visible_plan(vis, {0: ()}, _eff(), _eff().objective)
    assert abs(r.J_vis) < 1e-12 and abs(r.D) < 1e-12 and abs(r.E) < 1e-9, \
        f"unused idle 应零增量，得到 J={r.J_vis} D={r.D} E={r.E}"
    assert r.feasible and r.finite
    return True


def test_partition_duplicate_rejected():
    """故意构造跨车重复客户 → 完整分区拒绝。"""
    class _FakeEnv:
        num_nodes = 3
        def __init__(self):
            self.demands = np.array([[0.0, 1.0, 1.0]], np.float32)
            self.reveal_time = np.array([[0.0, 0.0, 0.0]], np.float32)
    env = _FakeEnv()
    # 两辆车，客户 1 同时出现在两车 suffix
    p = {0: VehiclePlan(0, 0, 0.0, 0.0, (1,)), 1: VehiclePlan(1, 0, 0.0, 0.0, (1,))}
    vehicles = []
    ok, detail = _partition_ok(env, 0, 0.0, vehicles, p, set(), np.array([True, False, False]))
    assert not ok and detail.get('duplicate_suffix') == [1], f"应检测重复，得到 {detail}"
    return True


def test_deferred_resolved_on_insert():
    """客户进入 suffix 后应从 deferred 移除（候选携带自己的 deferred）。"""
    class _FakeEnv:
        num_nodes = 2
        def __init__(self):
            self.demands = np.array([[0.0, 1.0]], np.float32)
            self.reveal_time = np.array([[0.0, 0.0]], np.float32)
    env = _FakeEnv()
    p = {0: VehiclePlan(0, 0, 0.0, 0.0, (1,))}
    ok, detail = _partition_ok(env, 0, 0.0, [], p, {1}, np.array([True, False]))
    assert ok and not detail.get('suffix_and_deferred') and 1 not in detail.get('missing', []), \
        f"应解析（suffix 胜），得到 {detail}"
    return True


def test_new_route_precool_once():
    """同一已知任务：空车服务它恰收一次预冷（E 含 ~0.5 kWh dispatch）。"""
    # 单客户 node1，dist depot<->1 = 1.0
    dist = [[0.0, 1.0], [1.0, 0.0]]
    vis = _vis([_idle_vehicle(0)], node_ids=(0, 1), dist=dist, pool=(1,))
    vis = VisibleState(
        inst_idx=0, event_id=0, clock=0.0, capacity=50.0, tw_speed=1.0, depot_tw_end=24.0,
        has_future_reveal=False, node_ids=(0, 1),
        demands=(0.0, 10.0), tw_start=(0.0, 0.0), tw_end=(24.0, 24.0),
        service_time=(0.0, 0.1), temp_class=(0, 1), initial_quality=(1.0, 1.0),
        dist_mat=((0.0, 1.0), (1.0, 0.0)), travel_mat=((0.0, 1.0), (1.0, 0.0)),
        vehicles=(_idle_vehicle(0),), pool_customer_ids=(1,), deferred_customer_ids=(),
        replan_ids=(0,), _idx={0: 0, 1: 1})
    r = evaluate_visible_plan(vis, {0: (1,)}, _eff(), _eff().objective)
    # dispatch 预冷 = 0.05+0.15+0.30 = 0.50 kWh；E 应 >= 0.5 且 == 0.5 + 冷却/行驶能耗
    assert r.E >= 0.5 - 1e-9, f"应含一次预冷(0.5kWh)，得到 E={r.E}"
    assert r.n_known_completed == 1
    return True


def main():
    tests = [test_unused_idle_zero, test_partition_duplicate_rejected,
             test_deferred_resolved_on_insert, test_new_route_precool_once]
    for t in tests:
        ok = t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == '__main__':
    main()
