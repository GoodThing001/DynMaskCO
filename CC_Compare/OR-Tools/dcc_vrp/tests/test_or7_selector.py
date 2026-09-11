"""OR7 预算选择器测试：preflight 冻结字段、选择规则、bootstrap、批次校验。"""
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np
import or7_selector as sel
from parity_compare import admission_check

CELLS = sel.CELLS


def _per_cell_uniform(d, n=32):
    return {c: [float(d)] * n for c in CELLS}


def _modified_protocol(**overrides):
    proto = json.load(open(sel.PROTOCOL_PATH, encoding='utf-8'))
    for path, val in overrides.items():
        node = proto
        keys = path.split('.')
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return proto


def _fake_instance(dist, i, fallback=0):
    return {'decision_hash': ('0' * 63) + str(i),
            'outcome': {'complete': True, 'distance_cost': dist},
            'stats': {'fallback_triggered_events': fallback, 'n_solver_calls': 5}}


def _write_two_runs(tmp, dist=20.0, fallback=0, drop_cell=None, n=2):
    for run in ('a', 'b'):
        rd = os.path.join(tmp, run)
        for cell in CELLS:
            if cell == drop_cell:
                continue
            d = os.path.join(rd, cell, 'instances')
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                with open(os.path.join(d, f'inst_{i}.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(_fake_instance(dist, i, fallback), f)
    return os.path.join(tmp, 'a'), os.path.join(tmp, 'b')


# --------------------------------------------------------------------------
# preflight：冻结字段漂移拒绝
# --------------------------------------------------------------------------
def test_preflight_rejects_json_drift():
    cases = [
        ('budget_candidates.solution_limit', [999]),
        ('budget_candidates.time_limit_s', 1.0),
        ('budget_candidates.solver_seed', 77),
        ('budget_candidates.threads', 2),
        ('budget_candidates.first_solution_strategy', 'RANDOM'),
        ('budget_candidates.local_search_metaheuristic', 'TABU'),
        ('gate_order', ['distance']),
        ('bootstrap.seed', 77),
        ('bootstrap.n_resamples', 1),
        ('selection_rule.threshold.mean_relative_distance', 0.9),
        ('selection_rule.threshold.bootstrap_ci_upper', 0.9),
        ('data.cells', ['r1_02']),
    ]
    for path, val in cases:
        proto = _modified_protocol(**{path: val})
        try:
            sel.preflight(protocol=proto)
            raise AssertionError(f'{path} 漂移未被拒绝')
        except RuntimeError:
            pass
    # 未篡改的冻结协议必须通过（动态 hash 由真实文件保证）
    sel.preflight()
    print('  preflight 拒绝全部冻结字段漂移 + 原协议通过')


# --------------------------------------------------------------------------
# 选择规则
# --------------------------------------------------------------------------
def test_select_smaller_equivalent():
    results = [
        {'solution_limit': 10, 'eligible': True,
         'per_cell': _per_cell_uniform(20.06)},
        {'solution_limit': 30, 'eligible': True,
         'per_cell': _per_cell_uniform(20.0)},
        {'solution_limit': 100, 'eligible': True,
         'per_cell': _per_cell_uniform(20.02)},
    ]
    out = sel.select_budget(results)
    assert out['selected_budget'] == 10, out
    assert out['upgrade'] is False
    print('  +0.3% 更小预算等价 → 选择最小预算')


def test_select_smaller_not_equivalent():
    results = [
        {'solution_limit': 10, 'eligible': True,
         'per_cell': _per_cell_uniform(20.3)},
        {'solution_limit': 30, 'eligible': True,
         'per_cell': _per_cell_uniform(20.0)},
        {'solution_limit': 100, 'eligible': True,
         'per_cell': _per_cell_uniform(20.02)},
    ]
    out = sel.select_budget(results)
    assert out['selected_budget'] == 30, out
    print('  +1.5% 更小预算不等价 → 选择参考预算')


def test_monotonic_improvement_upgrades():
    results = [
        {'solution_limit': 10, 'eligible': True,
         'per_cell': _per_cell_uniform(103.0)},
        {'solution_limit': 30, 'eligible': True,
         'per_cell': _per_cell_uniform(101.5)},
        {'solution_limit': 100, 'eligible': True,
         'per_cell': _per_cell_uniform(100.0)},
    ]
    out = sel.select_budget(results)
    assert out['selected_budget'] is None, out
    assert out['upgrade'] is True, out
    assert out['verdict'] == 'UPGRADE_TO_300', out
    print('  10→30→100 持续改善 → 触发追加 300')


# --------------------------------------------------------------------------
# bootstrap 同索引
# --------------------------------------------------------------------------
def test_bootstrap_single_index_per_resample():
    n, n_resamples = 5, 7
    indices = sel._grouped_resample_indices(n, n_resamples, 0)
    assert len(indices) == n_resamples, '每轮应只生成一组索引'
    assert all(len(idx) == n for idx in indices), '索引长度应为实例数'
    print('  bootstrap 每轮一组索引（九个 cell 共用同一 seed 索引）')


# --------------------------------------------------------------------------
# 批次校验（缺 cell / 伪造 gate / 重复预算 / 相同 run_id）
# --------------------------------------------------------------------------
def test_validate_instance_pairs_rejects_missing_cell():
    tmp = tempfile.mkdtemp()
    try:
        a, b = _write_two_runs(tmp, drop_cell=CELLS[-1])
        try:
            sel._validate_instance_pairs(a, b, 2)
            raise AssertionError('缺 cell 未被拒绝')
        except RuntimeError as e:
            assert '缺实例' in str(e)
        print('  缺 cell 被拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_instance_pairs_marks_ineligible_on_fallback():
    tmp = tempfile.mkdtemp()
    try:
        a, b = _write_two_runs(tmp, fallback=1)
        eligible, per_cell = sel._validate_instance_pairs(a, b, 2)
        assert eligible is False
        assert set(per_cell.keys()) == set(CELLS)
        print('  伪造 gate（fallback>0）→ eligible=False（不接受手填 gate_pass）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_instance_pairs_rejects_decision_parity_mismatch():
    tmp = tempfile.mkdtemp()
    try:
        a, b = _write_two_runs(tmp)
        # 篡改 run_b 一个实例的 decision_hash
        p = os.path.join(b, CELLS[0], 'instances', 'inst_0.json')
        r = json.load(open(p, encoding='utf-8'))
        r['decision_hash'] = 'f' * 64
        json.dump(r, open(p, 'w', encoding='utf-8'))
        try:
            sel._validate_instance_pairs(a, b, 2)
            raise AssertionError('decision parity 不一致未被拒绝')
        except RuntimeError as e:
            assert 'parity' in str(e)
        print('  decision parity 不一致被拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parse_batches_rejects_duplicate():
    spec = {'instances_per_cell': 2, 'budgets': [
        {'solution_limit': 10, 'run_a': 'a', 'run_b': 'b'},
        {'solution_limit': 10, 'run_a': 'c', 'run_b': 'd'},
    ]}
    try:
        sel.parse_batches(spec)
        raise AssertionError('重复预算未被拒绝')
    except RuntimeError as e:
        assert '重复预算' in str(e)
    print('  重复预算被拒绝')


def test_admission_same_run_id_rejected():
    assert admission_check('p', 'p', 'r1', 'r1')  # 非空 → 拒绝
    assert admission_check('p', 'p', 'r1', 'r2') == []
    print('  相同 run_id 被 admission 拒绝')


def main():
    test_preflight_rejects_json_drift()
    test_select_smaller_equivalent()
    test_select_smaller_not_equivalent()
    test_monotonic_improvement_upgrades()
    test_bootstrap_single_index_per_resample()
    test_validate_instance_pairs_rejects_missing_cell()
    test_validate_instance_pairs_marks_ineligible_on_fallback()
    test_validate_instance_pairs_rejects_decision_parity_mismatch()
    test_parse_batches_rejects_duplicate()
    test_admission_same_run_id_rejected()
    print('PASS test_or7_selector')


if __name__ == '__main__':
    main()
