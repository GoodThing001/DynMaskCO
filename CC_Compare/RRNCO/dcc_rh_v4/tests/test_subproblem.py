"""test_subproblem.py — 可见子问题构造 + canonical hash（未来扰动不变性）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subproblem import build_subproblem
from testutil import make_view


def _base_view(coords, pool, anchor=0, extra_coords=()):
    """pool 为可见客户；extra_coords 追加为「未来客户」（不在 pool）。"""
    all_coords = list(coords) + list(extra_coords)
    n = len(all_coords)
    return make_view(all_coords, [0.0] * n, [0.0] * n, [100.0] * n, [0.0] * n,
                     [(1, anchor, 0.0, 0.0, 'ready', ())],
                     [1], 50.0, 100.0, pool, has_future_reveal=True)


def test_subproblem_nodes_and_hash():
    view = _base_view([(0, 0), (1, 1), (2, 0)], pool=[1, 2])
    v = view.vehicles[0]
    sp = build_subproblem(view, v)
    assert sp.sub_nodes == (0, 1, 2), sp.sub_nodes
    assert sp.canonical_hash() == sp.canonical_hash()
    print('  subproblem 节点/哈希确定性通过')


def test_future_perturbation_invariance():
    """改变未来客户（坐标/数量）不影响子问题 canonical hash。"""
    base = _base_view([(0, 0), (1, 1), (2, 0)], pool=[1, 2],
                      extra_coords=[(10, 10), (10, -10)])
    h0 = build_subproblem(base, base.vehicles[0]).canonical_hash()

    v1 = _base_view([(0, 0), (1, 1), (2, 0)], pool=[1, 2],
                    extra_coords=[(9, 9), (8, -8), (7, 0)])
    h1 = build_subproblem(v1, v1.vehicles[0]).canonical_hash()

    assert h0 == h1, '未来客户变化导致 canonical hash 改变（泄漏）'
    print('  未来扰动不变性通过')


def test_visible_pool_change_affects_hash():
    v0 = _base_view([(0, 0), (1, 1), (2, 0)], pool=[1, 2])
    h0 = build_subproblem(v0, v0.vehicles[0]).canonical_hash()
    v1 = _base_view([(0, 0), (1, 1), (2, 0)], pool=[1])
    h1 = build_subproblem(v1, v1.vehicles[0]).canonical_hash()
    assert h0 != h1
    print('  可见 pool 变化改变 hash 通过')


def main():
    test_subproblem_nodes_and_hash()
    test_future_perturbation_invariance()
    test_visible_pool_change_affects_hash()
    print('PASS test_subproblem')


if __name__ == '__main__':
    main()
