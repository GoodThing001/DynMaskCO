"""test_preference.py — 偏好定义（首次出现顺序）+ 校验 + mock provider + 无 view 通道。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preference import (extract_first_occurrence, validate_ordering,
                        MockPreferenceProvider)
from subproblem import build_subproblem
from testutil import make_view


def _sp(pool, anchor=0):
    coords = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    n = len(coords)
    view = make_view(coords, [0.0] * n, [0.0] * n, [100.0] * n, [0.0] * n,
                     [(1, anchor, 0.0, 0.0, 'ready', ())],
                     [1], 50.0, 100.0, pool, True)
    return build_subproblem(view, view.vehicles[0])


def test_extract_first_occurrence():
    seq = [0, 1, 2, 0, 3, 1, 4]
    assert extract_first_occurrence(seq, depot=0, anchor=None) == (1, 2, 3, 4)
    assert extract_first_occurrence([5, 1, 5, 2], depot=0, anchor=5) == (1, 2)
    print('  extract_first_occurrence 通过')


def test_validate_ordering():
    pool = [1, 2, 3]
    assert validate_ordering([1, 2, 3], pool) == []
    assert any('外部' in p for p in validate_ordering([1, 2, 4], pool))
    assert any('缺失' in p for p in validate_ordering([1, 2], pool))
    assert any('重复' in p for p in validate_ordering([1, 2, 2], pool))
    print('  validate_ordering 外部/缺失/重复 全拒绝')


def test_mock_provider_determinism():
    sp = _sp([1, 2, 3])
    edd = MockPreferenceProvider('edd')
    assert edd.order(sp) == edd.order(sp)
    sh = MockPreferenceProvider('shuffle', seed=7)
    assert sh.order(sp) == sh.order(sp)
    assert set(edd.order(sp)) == {1, 2, 3}
    print('  MockPreferenceProvider 确定性 + 完整覆盖 pool')


def test_subproblem_has_no_view_channel():
    """P0-2：provider 只能拿到自包含子问题，无法访问完整 view。"""
    coords = [(0, 0), (1, 1), (2, 0), (3, 1), (40, 0)]   # 4 = 隐藏节点
    n = len(coords)
    view = make_view(coords, [0.0] * n, [0.0] * n, [100.0] * n, [0.0] * n,
                     [(1, 0, 0.0, 0.0, 'ready', ())],
                     [1], 50.0, 100.0, [1, 2], True)
    sp = build_subproblem(view, view.vehicles[0])
    assert not hasattr(sp, 'view'), 'SubProblem 仍暴露 view 引用'
    assert sp.sub_nodes == (0, 1, 2), sp.sub_nodes
    # 隐藏节点 4 不在子问题内
    try:
        sp.node_index(4)
        raise AssertionError('隐藏节点不应可寻址')
    except ValueError:
        pass
    print('  SubProblem 无 view 通道（结构隔离）')


def main():
    test_extract_first_occurrence()
    test_validate_ordering()
    test_mock_provider_determinism()
    test_subproblem_has_no_view_channel()
    print('PASS test_preference')


if __name__ == '__main__':
    main()
