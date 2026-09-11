"""B1 Gate 测试 1：记录 schema 完整 + 损坏/缺字段/NaN/Inf/交叉校验拒绝。

覆盖 B1.1 新增校验：audit 层一致性、action 分层、plan_diff certificate、
model_input 子集、clock 单调、event 可解析性等。
"""
import copy
import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import helpers
from strict_online_runner import run_instance
from record_validation import validate_instance_record
import baseline_contract as bc
from coldchain_contract import default_pilot_profile


def _valid_record():
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    return run_instance(ds, capacity=50, num_vehicles=4,
                        adapter_factory=helpers.GreedyEDDAdapter, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=0, data_sha256='test', instance_seed=0)


def test_schema_and_validation():
    rec = _valid_record()
    # 1) 有效记录通过校验
    err, checks = validate_instance_record(rec, 'coldchain')
    assert err is None, f'有效记录被拒绝: {err}'

    # 2) identity 缺字段被拒绝
    for k in bc.IDENTITY_FIELDS:
        bad = copy.deepcopy(rec)
        del bad[k]
        err, _ = validate_instance_record(bad, 'coldchain')
        assert err is not None, f'缺 identity 字段 {k} 未被拒绝'

    # 3) schema_version 不匹配被拒绝
    bad = copy.deepcopy(rec)
    bad['schema_version'] = 'wrong-version'
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None and 'schema_version' in err

    # 4) NaN/Inf 被拒绝
    bad = copy.deepcopy(rec)
    bad['events'][0]['clock'] = float('nan')
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'NaN clock 未被拒绝'
    bad = copy.deepcopy(rec)
    bad['events'][0]['model_runtime_s'] = float('inf')
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'Inf model_runtime_s 未被拒绝'
    bad = copy.deepcopy(rec)
    bad['outcome']['coldchain_cost'] = float('nan')
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'NaN coldchain_cost 未被拒绝'

    # 5) hard_vector 自报不一致被拒绝
    bad = copy.deepcopy(rec)
    bad['hard_vector'] = dict(bad['hard_vector'])
    bad['hard_vector']['complete'] = not bad['hard_vector']['complete']
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'hard_vector 与 outcome 不一致未被拒绝'

    # 6) action_type / action_layer 非法被拒绝
    bad = copy.deepcopy(rec)
    bad['actions'][0]['action_type'] = 'HACK'
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, '非法 action_type 未被拒绝'
    bad = copy.deepcopy(rec)
    bad['actions'][0]['action_layer'] = 'other'
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, '非法 action_layer 未被拒绝'

    # 7) 轨迹 NaN 被拒绝
    bad = copy.deepcopy(rec)
    bad['execution_trace']['vehicles'][0]['services'][0]['arrival_time'] = float('nan')
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'trace NaN 未被拒绝'

    # 8) model_input_customers 超出 visible 被拒绝
    bad = copy.deepcopy(rec)
    bad['events'][0]['model_input_customers'] = [999]
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'model_input 超出 visible 未被拒绝'

    # 9) clock 非单调被拒绝
    bad = copy.deepcopy(rec)
    bad['events'][1]['clock'] = bad['events'][0]['clock'] - 1.0
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'clock 非单调未被拒绝'

    # 10) audit 与 outcome 不一致被拒绝
    bad = copy.deepcopy(rec)
    bad['audit'] = dict(bad['audit'])
    bad['audit']['ownership_violations'] = 5
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'audit/outcome 不一致未被拒绝'

    # 11) plan_diff 动作 writeback 必须 True / certificate 必须可行
    diff_actions = [a for a in rec['actions'] if a['action_layer'] == 'plan_diff']
    assert diff_actions, '记录缺少 plan_diff 动作'
    assert all(a['writeback_ok'] for a in diff_actions)
    for a in diff_actions:
        if a['action_type'] in ('INSERT', 'NEW_ROUTE'):
            assert a['certificate']['feasible'] is True
            assert a['certificate']['certificate_scope'] == 'final_vehicle_suffix'
    bad = copy.deepcopy(rec)
    for a in bad['actions']:
        if a['action_layer'] == 'plan_diff':
            a['writeback_ok'] = False
            break
    err, _ = validate_instance_record(bad, 'coldchain')
    assert err is not None, 'plan_diff writeback_ok=False 未被拒绝'

    # 12) outcome 审计归属字段
    assert rec['outcome']['repair_applicable'] is False
    assert rec['outcome']['audit_source'] == 'common_runner'
    assert rec['audit']['ownership_violations'] == 0
    assert rec['audit']['terminal_unresolved'] == 0

    # 13) 双 hash 存在且可 JSON 序列化
    assert rec['decision_hash'] and rec['artifact_hash']
    json.dumps(rec, default=str)

    # 14) 关键终局字段齐全
    for k in ('complete', 'n_unserved', 'n_duplicate', 'tw_feasible',
              'capacity_feasible', 'depot_return_feasible', 'distance_cost',
              'distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost',
              'temperature_hard_feasible', 'all_orders_picked',
              'all_cargo_delivered_to_depot', 'terminal_manifests_empty',
              'trace_accounting_consistent', 'distance_accounting_consistent',
              'ownership_violations', 'terminal_unresolved'):
        assert k in rec['outcome'], f'outcome 缺字段 {k}'
    print('  schema/validation/cross-check 全部断言通过')


def main():
    test_schema_and_validation()
    print('PASS test_baseline_contract')


if __name__ == '__main__':
    main()
