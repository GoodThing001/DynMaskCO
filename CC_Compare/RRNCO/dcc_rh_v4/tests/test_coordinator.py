"""test_coordinator.py — 确定性多车协调器（贪心/竞争/open-new-vehicle/deferred/fallback/非连续 ID）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinator import coordinate
from subproblem import build_subproblem
from testutil import make_view


class FixedProvider:
    def __init__(self, orderings):
        self.orderings = orderings

    def order(self, subproblem):
        return self.orderings[subproblem.vehicle_id]


def _orderings(view, provider):
    return {int(v.vehicle_id): tuple(provider.order(build_subproblem(view, v)))
            for v in view.vehicles
            if int(v.vehicle_id) in set(int(x) for x in view.replan_ids)}


def _view(coords, demands, tw_start, tw_end, service, capacity, depot_tw_end,
          pool, vehicle_specs, replan_ids, has_future=True):
    return make_view(coords, demands, tw_start, tw_end, service, vehicle_specs,
                     replan_ids, capacity, depot_tw_end, pool, has_future)


def test_greedy_single_vehicle():
    coords = [(0, 0), (1, 0), (2, 0)]
    view = _view(coords, [0, 3, 4], [0, 0, 0], [100, 100, 100], [0, 0, 0],
                 10.0, 100.0, [1, 2], [(1, 0, 0.0, 0.0, 'ready', ())], [1])
    r = coordinate(view, _orderings(view, FixedProvider({1: (1, 2)})))
    assert r.fallback_reason is None, r.fallback_reason
    assert r.suffixes[1] == (1, 2, 0), r.suffixes
    print('  单车贪心分配通过')


def test_open_new_vehicle_preference():
    """两辆空载 depot 闲置车 + 2 客户（容量允许一车装下）→ 优先续用第一辆。"""
    coords = [(0, 0), (5, 0), (6, 0)]
    view = _view(coords, [0, 3, 4], [0, 0, 0], [100, 100, 100], [0, 0, 0],
                 10.0, 100.0, [1, 2],
                 [(1, 0, 0.0, 0.0, 'ready', ()), (2, 0, 0.0, 0.0, 'ready', ())],
                 [1, 2])
    r = coordinate(view, _orderings(view, FixedProvider({1: (1, 2), 2: (1, 2)})))
    assert set(r.suffixes[1]) == {1, 2, 0}, r.suffixes     # 都给了车 1
    assert r.suffixes[2] in ((), (0,)), r.suffixes
    print('  open-new-vehicle 优先（续用已启动车辆）通过')


def test_cross_vehicle_contention():
    coords = [(0, 0), (5, 0), (10, 0), (20, 0)]   # 3 = 车 B anchor
    view = _view(coords, [0, 3, 4, 0], [0, 0, 0, 0], [100, 100, 100, 100],
                 [0, 0, 0, 0], 10.0, 100.0, [1, 2],
                 [(1, 0, 0.0, 0.0, 'ready', ()), (2, 3, 0.0, 0.0, 'ready', ())],
                 [1, 2])
    r = coordinate(view, _orderings(view, FixedProvider({1: (1, 2), 2: (1, 2)})))
    all_cust = sorted(c for s in r.suffixes.values() for c in s if c != 0)
    assert all_cust == [1, 2], '客户跨车唯一且全覆盖'
    print('  跨车竞争通过')


def test_non_contiguous_vehicle_ids():
    coords = [(0, 0), (5, 0), (6, 0)]
    view = _view(coords, [0, 3, 4], [0, 0, 0], [100, 100, 100], [0, 0, 0],
                 10.0, 100.0, [1, 2],
                 [(10, 0, 0.0, 0.0, 'ready', ()), (27, 0, 0.0, 0.0, 'ready', ())],
                 [10, 27])
    r = coordinate(view, _orderings(view, FixedProvider({10: (1, 2), 27: (2, 1)})))
    assert set(r.suffixes.keys()) == {10, 27}
    # open-new-vehicle 优先：2 客户都续用到先启动的车 10；车 27 空载 depot 闲置
    assert r.suffixes[10] == (1, 2, 0), r.suffixes
    assert r.suffixes[27] in ((), (0,)), r.suffixes
    print('  非连续车辆 ID + open-new-vehicle 通过')


def test_deferred_infeasible():
    coords = [(0, 0), (1, 0), (20, 0)]
    view = _view(coords, [0, 3, 4], [0, 0, 0], [100, 100, 3], [0, 0, 0],
                 10.0, 100.0, [1, 2], [(1, 0, 0.0, 0.0, 'ready', ())], [1])
    r = coordinate(view, _orderings(view, FixedProvider({1: (2, 1)})))
    assert r.fallback_reason is None
    assert 2 in r.deferred, r.deferred
    print('  暂时不可行 → deferred（非 fallback）通过')


def test_illegal_ordering_fallback():
    coords = [(0, 0), (1, 0), (2, 0)]
    view = _view(coords, [0, 3, 4], [0, 0, 0], [100, 100, 100], [0, 0, 0],
                 10.0, 100.0, [1, 2], [(1, 0, 0.0, 0.0, 'ready', ())], [1])
    r = coordinate(view, _orderings(view, FixedProvider({1: (1,)})))  # 缺失 2
    assert r.fallback_reason is not None
    assert r.suffixes[1] == (1, 2, 0), r.suffixes   # EDD 兜底
    print('  非法排序 → EDD fallback 通过')


def test_empty_pool_and_anchor_only():
    coords = [(0, 0), (1, 0), (20, 0)]
    view = _view(coords, [0, 3, 0], [0, 0, 0], [100, 100, 100], [0, 0, 0],
                 10.0, 100.0, [], [(1, 2, 0.0, 5.0, 'ready', ())], [1])
    r = coordinate(view, _orderings(view, FixedProvider({1: ()})))
    assert r.fallback_reason is None
    assert r.suffixes[1] in ((), (0,)), r.suffixes
    print('  空 pool / anchor-only 通过')


def main():
    test_greedy_single_vehicle()
    test_open_new_vehicle_preference()
    test_cross_vehicle_contention()
    test_non_contiguous_vehicle_ids()
    test_deferred_infeasible()
    test_illegal_ordering_fallback()
    test_empty_pool_and_anchor_only()
    print('PASS test_coordinator')


if __name__ == '__main__':
    main()
