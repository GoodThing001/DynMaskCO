"""test_pickup_certificate.py — 独立 pickup 认证（TW/容量/返仓 + WAIT/CLOSE）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pickup_certificate import (VehicleState, certify_append, append_state,
                                certify_suffix, wait_or_close)
from testutil import make_view, make_vehicle


def _view(coords, demands, tw_start, tw_end, service, capacity, depot_tw_end,
          pool, has_future=True, anchor=0, load=0.0, ready_time=0.0):
    view = make_view(coords, demands, tw_start, tw_end, service,
                     [(1, anchor, ready_time, load, 'ready', ())],
                     [1], capacity, depot_tw_end, pool, has_future)
    return view, view.vehicles[0]


def test_capacity_boundary():
    view, v = _view([(0, 0), (1, 0), (2, 0)], [0, 5, 6], [0, 0, 0],
                    [100, 100, 100], [0, 0, 0], 10.0, 100.0, [1, 2])
    st = VehicleState(0, 0.0, 0.0, ())
    ok, _ = certify_append(view, st, 1)
    assert ok
    st2 = append_state(view, st, 1)
    assert st2.load == 5.0
    ok2, reason = certify_append(view, st2, 2)   # 5 + 6 = 11 > 10
    assert not ok2 and 'capacity' in reason, reason
    print('  容量精确边界（=容量可行、>容量拒绝）通过')


def test_tw_boundary():
    view, v = _view([(0, 0), (5, 0)], [0, 3], [0, 0], [10, 10], [0, 0],
                    10.0, 100.0, [1])
    st = VehicleState(0, 0.0, 0.0, ())
    ok, reason = certify_append(view, st, 1)   # arrive=5 <= tw_end 10 → 可行
    assert ok, reason
    # tw_end 收紧到 4 → arrive 5 > 4 拒绝
    view2, v2 = _view([(0, 0), (5, 0)], [0, 3], [0, 0], [4, 4], [0, 0],
                      10.0, 100.0, [1])
    ok2, reason2 = certify_append(view2, VehicleState(0, 0.0, 0.0, ()), 1)
    assert not ok2 and 'tw' in reason2, reason2
    print('  TW 精确边界通过')


def test_return_boundary():
    view, v = _view([(0, 0), (5, 0)], [0, 3], [0, 0], [100, 100], [0, 0],
                    10.0, 5.0, [1])   # depot_tw_end=5，服务后返仓=10 > 5
    st = VehicleState(0, 0.0, 0.0, ())
    ok, reason = certify_append(view, st, 1)
    assert not ok and 'return' in reason, reason
    print('  返仓精确边界通过')


def test_suffix_semantics():
    view, v = _view([(0, 0), (1, 0), (2, 0)], [0, 3, 4], [0, 0, 0],
                    [100, 100, 100], [0, 0, 0], 10.0, 100.0, [1, 2])
    # WAIT
    assert certify_suffix(view, v, ()) == (True, None)
    # CLOSE
    assert certify_suffix(view, v, (0,)) == (True, None)
    # 完整路线 c1,c2,0
    ok, reason = certify_suffix(view, v, (1, 2, 0))
    assert ok, reason
    # 内部 depot
    ok2, r2 = certify_suffix(view, v, (1, 0, 2, 0))
    assert not ok2 and '内部' in r2, r2
    # 非 depot 结尾
    ok3, r3 = certify_suffix(view, v, (1, 2))
    assert not ok3 and '未以' in r3, r3
    # 外部客户
    ok4, r4 = certify_suffix(view, v, (1, 9, 0))
    assert not ok4, r4
    print('  suffix WAIT/CLOSE/完整/内部 depot/非 depot 结尾/外部 全区分')


def test_wait_or_close():
    # 有未来 + 可等待 → WAIT ()
    view, v = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [100, 100], [0, 0],
                    10.0, 100.0, [1], has_future=True)
    assert wait_or_close(view, v) == ()
    # 无未来 → CLOSE (0,)
    view2, v2 = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [100, 100], [0, 0],
                      10.0, 100.0, [1], has_future=False)
    assert wait_or_close(view2, v2) == (0,)
    print('  wait_or_close WAIT/CLOSE 通过')


def test_exact_boundaries():
    """精确边界：=阈值可行，=阈值+ε 拒绝（容量/TW/返仓三处）。"""
    # 容量：5 + 5 == 10 可行；5 + 5.0001 > 10 拒绝
    view, v = _view([(0, 0), (1, 0), (2, 0)], [0, 5, 5.0001], [0, 0, 0],
                    [100, 100, 100], [0, 0, 0], 10.0, 100.0, [1, 2])
    st = append_state(view, VehicleState(0, 0.0, 0.0, ()), 1)   # load=5
    ok, _ = certify_append(view, st, 2)      # 5 + 5.0001 = 10.0001 > 10
    assert not ok
    # 容量恰好 =10：customer 2 demand 5
    view2, v2 = _view([(0, 0), (1, 0), (2, 0)], [0, 5, 5], [0, 0, 0],
                      [100, 100, 100], [0, 0, 0], 10.0, 100.0, [1, 2])
    st2 = append_state(view2, VehicleState(0, 0.0, 0.0, ()), 1)
    ok2, _ = certify_append(view2, st2, 2)   # 5 + 5 = 10 == capacity 可行
    assert ok2
    # TW：arrive==tw_end 可行，arrive>tw_end+ε 拒绝
    view3, v3 = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [1.0, 1.0], [0, 0],
                      10.0, 100.0, [1])
    ok3, _ = certify_append(view3, VehicleState(0, 0.0, 0.0, ()), 1)  # arrive 1 == tw_end 1
    assert ok3
    view4, v4 = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [0.999, 0.999], [0, 0],
                      10.0, 100.0, [1])
    ok4, _ = certify_append(view4, VehicleState(0, 0.0, 0.0, ()), 1)  # arrive 1 > 0.999
    assert not ok4
    # 返仓：ret==depot_tw_end 可行，ret>depot_tw_end 拒绝
    view5, v5 = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [100, 100], [0, 0],
                      10.0, 2.0, [1])   # ret = 1 + 0 + 1 = 2 == depot_tw_end
    ok5, _ = certify_append(view5, VehicleState(0, 0.0, 0.0, ()), 1)
    assert ok5
    view6, v6 = _view([(0, 0), (1, 0)], [0, 3], [0, 0], [100, 100], [0, 0],
                      10.0, 1.999, [1])
    ok6, _ = certify_append(view6, VehicleState(0, 0.0, 0.0, ()), 1)
    assert not ok6
    print('  精确边界（=阈值可行 / +ε 拒绝）容量·TW·返仓 通过')


def main():
    test_capacity_boundary()
    test_tw_boundary()
    test_return_boundary()
    test_exact_boundaries()
    test_suffix_semantics()
    test_wait_or_close()
    print('PASS test_pickup_certificate')


if __name__ == '__main__':
    main()
