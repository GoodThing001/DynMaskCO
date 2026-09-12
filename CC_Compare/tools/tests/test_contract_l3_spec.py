"""test_contract_l3_spec.py — L3 严格判等规范的失败测试（先失败后实现）。

覆盖：
  - decision_hash 不同必须失败；
  - actions / execution_trace 任一不同必须失败；
  - outcome.distance_cost 等数值不同必须失败；
  - hard_vector / audit / 非计时 stats 任一不同必须失败；
  - 仅 runtime_s / run_id / code_hash / artifact_hash 不同仍判一致；
  - 缺字段不得默认为相等（MISSING_FIELD）；
  - 9 cell 集合缺失、重复必须失败。

用法（cc_ortools env）：
    python -m pytest -q CC_Compare/tools/tests/test_contract_l3_spec.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract_l3_spec import compare_behavior, validate_cell_set


def _record(**overrides):
    base = {
        'method_name': 'pyvrp-rh-d',
        'method_revision': '1',
        'adapter_revision': '0',
        'checkpoint_hash': 'none',
        'code_hash': 'code_old',
        'data_hash': 'data',
        'profile_hash': 'profile',
        'instance_id': 0,
        'instance_seed': 123,
        'scene_instance_id': 'scene_0',
        'objective': 'coldchain',
        'runtime_budget': None,
        'random_seed': 0,
        'schema_version': 'cc-compare-record-v1',
        'events': [{'event_id': 0, 'clock': 0.0}],
        'actions': [{'action_type': 'COMMIT', 'vehicle_id': 1, 'customer_id': None}],
        'execution_trace': {'traces': [{'vehicle_id': 1, 'nodes': [0]}]},
        'outcome': {'distance_cost': 12.34, 'distance_km': 12.34,
                    'quality_loss': 0.0, 'energy_kwh': 0.0,
                    'coldchain_cost': 50.0, 'complete': True},
        'hard_vector': {'service_ok': True, 'capacity_ok': True},
        'audit': {'ownership_violations': 0, 'terminal_unresolved': 0},
        'protocol': {'error': None, 'checks': []},
        'stats': {'n_events': 1, 'n_solver_calls': 2, 'n_waits': 0,
                  'n_fast_path': 0, 'fallback_triggered_events': 0},
        'runtime_s': 0.123,
        'decision_hash': 'd1',
        'artifact_hash': 'a1',
    }
    base.update(overrides)
    return base


def test_identical_is_unchanged():
    v, d = compare_behavior(_record(), _record())
    assert v == 'BEHAVIOR_UNCHANGED', (v, d)


def test_decision_hash_diff_fails():
    v, _ = compare_behavior(_record(), _record(decision_hash='d2'))
    assert v == 'BEHAVIOR_CHANGED', v


def test_actions_diff_fails():
    a = _record()
    b = _record(actions=[{'action_type': 'COMMIT', 'vehicle_id': 2, 'customer_id': 1}])
    v, _ = compare_behavior(a, b)
    assert v == 'BEHAVIOR_CHANGED', v


def test_execution_trace_diff_fails():
    a = _record()
    b = _record(execution_trace={'traces': [{'vehicle_id': 1, 'nodes': [5]}]})
    v, _ = compare_behavior(a, b)
    assert v == 'BEHAVIOR_CHANGED', v


def test_outcome_numeric_diff_fails():
    for field in ('distance_cost', 'distance_km', 'quality_loss',
                  'energy_kwh', 'coldchain_cost'):
        a = _record()
        out = dict(a['outcome'])
        out[field] = out[field] + 0.001
        v, _ = compare_behavior(a, _record(outcome=out))
        assert v == 'BEHAVIOR_CHANGED', (field, v)


def test_hard_vector_diff_fails():
    a = _record()
    b = _record(hard_vector={'service_ok': False, 'capacity_ok': True})
    v, _ = compare_behavior(a, b)
    assert v == 'BEHAVIOR_CHANGED', v


def test_audit_diff_fails():
    a = _record()
    b = _record(audit={'ownership_violations': 1, 'terminal_unresolved': 0})
    v, _ = compare_behavior(a, b)
    assert v == 'BEHAVIOR_CHANGED', v


def test_non_timing_stats_diff_fails():
    a = _record()
    b = _record(stats={'n_events': 2, 'n_solver_calls': 2, 'n_waits': 0,
                       'n_fast_path': 0, 'fallback_triggered_events': 0})
    v, _ = compare_behavior(a, b)
    assert v == 'BEHAVIOR_CHANGED', v


def test_identity_and_timing_only_diff_is_unchanged():
    a = _record()
    b = _record(code_hash='code_new', artifact_hash='a2', runtime_s=0.999)
    v, d = compare_behavior(a, b)
    assert v == 'BEHAVIOR_UNCHANGED', (v, d)


def test_missing_field_fails():
    a = _record()
    b = _record()
    del b['outcome']
    v, d = compare_behavior(a, b)
    assert v == 'MISSING_FIELD', (v, d)


def test_cell_set_valid():
    cells = [{'type': t, 'edod': e}
             for t in ('R1', 'C1', 'RC1') for e in (0.2, 0.5, 0.8)]
    ok, problems = validate_cell_set(cells)
    assert ok, problems


def test_cell_set_missing_fails():
    cells = [{'type': t, 'edod': e}
             for t in ('R1', 'C1', 'RC1') for e in (0.2, 0.5)]  # 缺 0.8
    ok, problems = validate_cell_set(cells)
    assert not ok, problems
    assert any('缺失' in p for p in problems)


def test_cell_set_duplicate_fails():
    cells = [{'type': 'R1', 'edod': 0.2}, {'type': 'R1', 'edod': 0.2}]
    ok, problems = validate_cell_set(cells)
    assert not ok, problems
    assert any('重复' in p for p in problems)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'  [PASS] {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  [FAIL] {fn.__name__}: {e}')
    print(f'ALL: {"PASS" if failed == 0 else f"{failed} FAIL"}')
    sys.exit(1 if failed else 0)
