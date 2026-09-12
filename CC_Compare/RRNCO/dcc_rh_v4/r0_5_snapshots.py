"""R0.5 三类人工快照（Stage A.1 断言测试，MockPreferenceProvider，不跑真实模型）。

每类快照构造 mock view → 计算 orderings → 协调 → 断言预期 suffix/deferred。
同时打印供人工核对（打印是演示，断言是验收）。
"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DCC_VRP)

from coordinator import coordinate
from preference import MockPreferenceProvider
from subproblem import build_subproblem
from testutil import make_view


def _orderings(view, provider):
    return {int(v.vehicle_id): tuple(provider.order(build_subproblem(view, v)))
            for v in view.vehicles
            if int(v.vehicle_id) in set(int(x) for x in view.replan_ids)}


def _show_and_assert(name, view, provider, expected):
    r = coordinate(view, _orderings(view, provider))
    print(f'\n===== {name} =====')
    for v in view.vehicles:
        sp = build_subproblem(view, v)
        print(f'  vehicle {v.vehicle_id}: anchor={v.anchor_node_id} '
              f'sub_nodes={sp.sub_nodes} hash={sp.canonical_hash()[:12]}')
    print(f'  suffixes = {r.suffixes}  deferred = {r.deferred} '
          f'fallback = {r.fallback_reason}')
    assert r.suffixes == expected['suffixes'], (r.suffixes, expected['suffixes'])
    if 'deferred' in expected:
        assert r.deferred == expected['deferred'], r.deferred
    assert r.fallback_reason == expected.get('fallback', None)


def snapshot_1_initial_partial_reveal():
    # depot + 可见客户 1,2,3；未来客户 4,5 不在 pool。容量 10：1+2 可装，3 超载 defer
    coords = [(0, 0), (1, 0), (2, 0), (3, 0), (40, 0), (41, 0)]
    view = make_view(coords, [0, 3, 4, 5, 3, 4], [0, 0, 0, 0, 0, 0],
                     [100, 100, 100, 100, 100, 100], [0, 0, 0, 0, 0, 0],
                     [(1, 0, 0.0, 0.0, 'ready', ())], [1],
                     10.0, 100.0, [1, 2, 3], True)
    _show_and_assert('快照1 初始部分揭示', view, MockPreferenceProvider('edd'),
                     {'suffixes': {1: (1, 2, 0)}, 'deferred': (3,)})


def snapshot_2_mid_trip_anchor_load():
    # 车 1 位于客户 2（anchor=2），ready_time=3，load=5；pool={1,3}；容量 10：
    # 客户 1 需求 3 → 5+3=8 可行；客户 3 需求 5 → 8+5=13 超载 defer
    coords = [(0, 0), (5, 0), (2, 0), (8, 0)]
    view = make_view(coords, [0, 3, 4, 5], [0, 0, 0, 0], [100, 100, 100, 100],
                     [0, 0, 0, 0], [(1, 2, 3.0, 5.0, 'ready', ())], [1],
                     10.0, 100.0, [1, 3], True)
    _show_and_assert('快照2 行程中间 anchor+load', view,
                     MockPreferenceProvider('nearest'),
                     {'suffixes': {1: (1, 0)}, 'deferred': (3,)})


def snapshot_3_multi_vehicle_frozen_prefix():
    # 车 1 committed 到客户 3（冻结前缀，不在 replan）；车 2 在 depot 可重规划；
    # pool = {2,4}（客户 3 冻结，不在 pool）
    coords = [(0, 0), (2, 0), (5, 0), (8, 0), (10, 0)]
    view = make_view(coords, [0, 3, 4, 5, 6], [0, 0, 0, 0, 0],
                     [100, 100, 100, 100, 100], [0, 0, 0, 0, 0],
                     [(1, 3, 5.0, 5.0, 'committed', (3,)),
                      (2, 0, 0.0, 0.0, 'ready', ())],
                     [2], 10.0, 100.0, [2, 4], True)
    _show_and_assert('快照3 多车混合+冻结前缀', view, MockPreferenceProvider('edd'),
                     {'suffixes': {2: (2, 4, 0)}})


if __name__ == '__main__':
    snapshot_1_initial_partial_reveal()
    snapshot_2_mid_trip_anchor_load()
    snapshot_3_multi_vehicle_frozen_prefix()
    print('\n三快照断言验收通过（真实模型验证见 Stage B）')
