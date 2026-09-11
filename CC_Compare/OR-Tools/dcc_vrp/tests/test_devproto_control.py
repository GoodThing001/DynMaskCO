"""OR5/OR6 纯控制链测试（不运行真实求解；用伪造记录测守卫/校验/发布/汇总层）。

覆盖四个 P0 + P1 项 + 控制负例：
  P0-1  effective_config 绑定实际预算；protocol_id 确定性 + run_id 唯一；
        无效续跑记录立即退出不覆盖（resume_decision）
  P0-2  续跑校验身份（code/seed/scene/checkpoint/artifact hash）；checkpoint 精确匹配；
        verify_aggregate 复验实际 pre-run（篡改拒绝）
  P0-3  seed/scene 由 DEV_MANIFEST 提供；发布单元完整绑定（manifest/实例文件/cell marker）；
        落盘校验先跑完整 common 合同 + trace replay
  P0-4  aggregator 校验 canonical.COMPLETE；每 cell 前后身份漂移停止
  P1    solver status 白名单、stats 从 events 重算、失败发布保留 BUILDING/FAILED、
        parity 批次准入（protocol 同 / run 异）

注意：fake_record 生成**完整 common 合同合法记录**（通过 record_validation +
trace_replay_check），因此 validate_instance_record 在 fake 记录上会先跑完整公共合同。
"""
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')))

import baseline_contract as bc
from ortools_instance_validation import validate_instance_record
from ortools_cell_summary import publish_cell, sha256_file
from aggregate_devproto import aggregate, verify_aggregate
from run_devproto import (guard_output_dir, build_effective_config,
                          resume_decision, verify_identity_still_intact, PROFILE)
from protocol_identity import build_protocol_id
from parity_compare import admission_check

DATA_SHA = 'd' * 64
PROFILE_HASH = 'p' * 64
CODE_HASH = 'c' * 64
CONTROL_HASH = 'cl' * 32
ANALYSIS_HASH = 'al' * 32
CHECKPOINT_PREFIX = 'ortools=9.11.4210_native='
FAKE_NATIVE = '0123456789abcdef' * 4       # 64 hex
FULL_CHECKPOINT = CHECKPOINT_PREFIX + FAKE_NATIVE
RUN_ID = 'run_id_0001'
PROTOCOL_ID = 'protocol_id_0001'


def fake_record(iid, complete=True, fallback_events=0, solver_status='OPTIMAL',
                scene=None, seed=None, data_sha=DATA_SHA,
                profile_hash=PROFILE_HASH, code_hash=CODE_HASH,
                checkpoint=FULL_CHECKPOINT):
    """构造**完整 common 合同合法**实例记录（decision/artifact hash 由内容实算）。"""
    n_events = 2
    events = []
    for e in range(n_events):
        ev = {
            'event_id': e,
            'clock': float(e),
            'revealed_customer_ids': [],
            'visible_customer_ids': [1, 2, 3],
            'served_customer_ids': [],
            'visible_ids_hash': 'a' * 32,
            'served_mask_hash': 'b' * 32,
            'state_hash': 'c' * 32,
            'replan_vehicle_ids': [0],
            'vehicles': [{
                'vehicle_id': 0, 'status': 'ready', 'anchor_node': 0,
                'ready_time': float(e), 'load': 0.0, 'committed_next': None,
                'mutable_suffix_before': [],
            }],
            'model_input_customers': [1, 2, 3],
            'model_runtime_s': 0.01,
            'budget_exceeded': False,
            'fallback_triggered': (e < fallback_events),
        }
        if solver_status is not None:
            ev['solve_meta'] = {'solver_status': solver_status}
        events.append(ev)
    n_unserved = 0 if complete else 1
    outcome = {
        'complete': complete,
        'n_unserved': n_unserved,
        'n_duplicate': 0,
        'tw_feasible': True,
        'capacity_feasible': True,
        'depot_return_feasible': True,
        'temperature_hard_feasible': True,
        'all_orders_picked': True,
        'all_cargo_delivered_to_depot': True,
        'terminal_manifests_empty': True,
        'trace_accounting_consistent': True,
        'distance_accounting_consistent': True,
        'distance_cost': 20.0, 'distance_km': 20.0,
        'quality_loss': 1.0, 'energy_kwh': 5.0, 'coldchain_cost': 2.0,
        'thermal_violation_count': 0, 'thermal_violation_duration_h': 0.0,
        'ownership_violations': 0, 'terminal_unresolved': 0,
        'repair_applicable': False, 'audit_source': 'common_runner',
    }
    hard_vector = {
        'complete': bool(complete),
        'n_unserved_zero': (n_unserved == 0),
        'n_duplicate_zero': True,
        'tw_feasible': True, 'capacity_feasible': True,
        'depot_return_feasible': True, 'temperature_hard_feasible': True,
        'all_orders_picked': True, 'all_cargo_delivered_to_depot': True,
        'terminal_manifests_empty': True,
        'trace_accounting_consistent': True,
        'distance_accounting_consistent': True,
    }
    rec = {
        'schema_version': bc.RECORD_SCHEMA_VERSION,
        'method_name': 'ortools-rh-d',
        'method_revision': '1',
        'adapter_revision': '1',
        'checkpoint_hash': checkpoint,
        'code_hash': code_hash,
        'data_hash': data_sha,
        'profile_hash': profile_hash,
        'instance_id': iid,
        'instance_seed': seed if seed is not None else 1000 + iid,
        'scene_instance_id': scene if scene is not None else f'scene_{iid}',
        'objective': 'coldchain',
        'runtime_budget': None,
        'random_seed': 0,
        'events': events,
        'actions': [],
        'execution_trace': {
            'vehicles': [{
                'vehicle_id': 0, 'used': True,
                'dispatch_time': 0.0, 'return_depart': 1.0,
                'return_arrival': 2.0,
                'dispatch_preconditioning_energy_kwh': 0.0,
                'return_segment_distance_km': 20.0,
                'return_segment_quality_loss': 1.0,
                'return_segment_energy_kwh': 5.0,
                'return_segment_thermal_violation_count': 0,
                'return_segment_thermal_violation_duration_h': 0.0,
                'services': [], 'unload_records': [],
                'final': {
                    'closed': True, 'cargo_manifest': [],
                    'total_load': 0.0, 'cumulative_distance_km': 20.0,
                    'cumulative_energy_kwh': 5.0,
                    'thermal_violation_count': 0,
                    'thermal_violation_duration_h': 0.0,
                },
            }],
        },
        'outcome': outcome,
        'hard_vector': hard_vector,
        'audit': {
            'ownership_violations': 0, 'terminal_unresolved': 0,
            'per_event': [], 'violations_by_type': {},
            'terminal_unresolved_customers': [], 'n_events_with_deferred': 1,
        },
        'stats': {
            'n_events': n_events, 'n_solver_calls': n_events, 'n_fast_path': 0,
            'n_penalty_warnings': 0,
            'fallback_triggered_events': fallback_events,
            'n_waits': 0, 'n_customer_anchor_waits': 0, 'n_loaded_waits': 0,
        },
        'runtime_s': 1.0,
    }
    rec['decision_hash'] = bc.decision_hash(rec)
    rec['artifact_hash'] = bc.artifact_hash(rec)
    return rec


def expected_for(iid, scene=None, seed=None, data_sha=DATA_SHA,
                 profile_hash=PROFILE_HASH, code_hash=CODE_HASH,
                 checkpoint=FULL_CHECKPOINT):
    return {
        'instance_id': iid,
        'scene_instance_id': scene if scene is not None else f'scene_{iid}',
        'instance_seed': seed if seed is not None else 1000 + iid,
        'data_sha256': data_sha,
        'profile_hash': profile_hash,
        'code_hash': code_hash,
        'checkpoint_hash': checkpoint,
    }


def _cell_dir(tmp, name='r1_02'):
    d = os.path.join(tmp, name)
    os.makedirs(os.path.join(d, 'instances'))
    return d


def _write_instances(cell_dir, recs):
    for i, rec in enumerate(recs):
        with open(os.path.join(cell_dir, 'instances', f'inst_{i}.json'),
                  'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False)


def _expected_map(recs):
    return {i: {'instance_seed': rec['instance_seed'],
                'scene_instance_id': rec['scene_instance_id']}
            for i, rec in enumerate(recs)}


def _publish(cell_dir, name, recs):
    publish_cell(cell_dir, name, tuple(range(len(recs))), _expected_map(recs),
                 DATA_SHA, PROFILE_HASH, CODE_HASH, FULL_CHECKPOINT, RUN_ID,
                 CONTROL_HASH)


def _aggregate(tmp, names, n=2):
    return aggregate(tmp, names, n, os.path.join(tmp, 'pre_run_manifest.json'),
                     expected_code_hash=CODE_HASH,
                     expected_profile_hash=PROFILE_HASH,
                     expected_data_sha256={x: DATA_SHA for x in names},
                     expected_checkpoint_hash=FULL_CHECKPOINT,
                     control_hash=CONTROL_HASH, analysis_hash=ANALYSIS_HASH,
                     protocol_id=PROTOCOL_ID, run_id=RUN_ID)


# --------------------------------------------------------------------------
# P0-1：effective_config / protocol_id / resume
# --------------------------------------------------------------------------
def test_effective_config_binds_actual_budget():
    cfg = build_effective_config(40, 3)
    assert cfg['solution_limit'] == 40, cfg
    assert cfg['instances_per_cell'] == 3, cfg
    assert cfg['instance_set'] == [0, 1, 2], cfg
    assert cfg['time_limit_s'] == 30.0
    assert cfg['solver_objective'] == 'distance'
    assert cfg['evaluation_objective'] == 'coldchain_v2'
    assert cfg['capacity'] == 50 and cfg['num_vehicles'] == 25
    assert cfg['solver_seed'] == 0 and cfg['threads'] == 1
    print('  P0-1 effective_config 绑定实际预算 + solver/eval objective 拆分')


def test_protocol_id_deterministic():
    cfg = build_effective_config(30, 2)
    p1 = build_protocol_id(CODE_HASH, CONTROL_HASH, ANALYSIS_HASH, 'm' * 64,
                           PROFILE_HASH, cfg)
    p2 = build_protocol_id(CODE_HASH, CONTROL_HASH, ANALYSIS_HASH, 'm' * 64,
                           PROFILE_HASH, cfg)
    assert p1 == p2
    p3 = build_protocol_id(CODE_HASH, CONTROL_HASH, ANALYSIS_HASH, 'm' * 64,
                           PROFILE_HASH, build_effective_config(40, 2))
    assert p1 != p3
    # OR7 启动项 1：control/analysis 代码身份变化必须改变 protocol_id
    p4 = build_protocol_id(CODE_HASH, 'x' * 64, ANALYSIS_HASH, 'm' * 64,
                           PROFILE_HASH, cfg)
    assert p1 != p4
    p5 = build_protocol_id(CODE_HASH, CONTROL_HASH, 'y' * 64, 'm' * 64,
                           PROFILE_HASH, cfg)
    assert p1 != p5
    print('  P0-1 protocol_id 确定性 + 绑定 compute/control/analysis')


def test_resume_invalid_record_refuses_overwrite():
    """P0-1：已有但校验失败 → 拒绝覆盖且原文件不变；合法 → reuse；缺失 → compute。"""
    tmp = tempfile.mkdtemp()
    try:
        out_path = os.path.join(tmp, 'inst_0.json')
        json.dump(fake_record(0, code_hash='x' * 64),
                  open(out_path, 'w', encoding='utf-8'))
        before = open(out_path, 'rb').read()
        try:
            resume_decision(out_path, True, expected_for(0))
            raise AssertionError('无效续跑记录未被拒绝')
        except RuntimeError as e:
            assert '拒绝覆盖' in str(e), e
        assert open(out_path, 'rb').read() == before, '原文件被改动'
        json.dump(fake_record(0), open(out_path, 'w', encoding='utf-8'))
        assert resume_decision(out_path, True, expected_for(0)) == 'reuse'
        assert resume_decision(os.path.join(tmp, 'missing.json'), True,
                               expected_for(0)) == 'compute'
        print('  P0-1 无效续跑记录拒绝覆盖且原文件不变')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# P0-2 + P0-3：身份校验（外源 expected）
# --------------------------------------------------------------------------
def test_identity_drift_rejected():
    rec = fake_record(0)
    for key, bad in (('code_hash', 'x' * 64),
                     ('data_sha', 'x' * 64),
                     ('profile_hash', 'x' * 64),
                     ('seed', 9999),
                     ('scene', 'tampered_scene')):
        kw = {key: bad}
        rec2 = fake_record(0, **kw)
        problems = validate_instance_record(rec2, expected_for(0))
        assert problems, f'{key} 漂移未被拒绝: {problems}'
    print('  P0-2/P0-3 code/data/profile/seed/scene 漂移全部拒绝')


def test_checkpoint_native_mismatch_rejected():
    """P1-3：checkpoint 完整 64 位 native hash，精确相等。"""
    rec = fake_record(0)
    rec['checkpoint_hash'] = CHECKPOINT_PREFIX + 'f' * 64
    rec['decision_hash'] = bc.decision_hash(rec)
    rec['artifact_hash'] = bc.artifact_hash(rec)
    problems = validate_instance_record(rec, expected_for(0))
    assert any('checkpoint_hash' in p for p in problems), problems
    print('  P1-3 checkpoint 完整 native hash 不同被拒绝')


def test_outcome_tamper_with_stale_artifact_hash():
    rec = fake_record(0)
    rec['outcome']['distance_cost'] = 9999.0
    problems = validate_instance_record(rec, expected_for(0))
    assert any('artifact_hash' in p for p in problems), problems
    print('  P0-2 篡改 outcome + 陈旧 artifact_hash 被重算对账拒绝')


def test_stats_self_report_mismatch_rejected():
    rec = fake_record(0)
    rec['stats']['n_solver_calls'] = 3  # 实际 2
    problems = validate_instance_record(rec, expected_for(0))
    assert any('n_solver_calls' in p for p in problems), problems
    print('  P1 stats 自报与 events 重算不一致被拒绝')


def test_solver_status_whitelist_rejected():
    for bad in ('TIME_LIMIT', 'NO_SOLUTION', 'SOLVER_EXCEPTION', 'SOLVED',
                'BUILD_ERROR'):
        rec = fake_record(0, solver_status=bad)
        problems = validate_instance_record(rec, expected_for(0))
        assert any(bad in p for p in problems), f'{bad}: {problems}'
    print('  P1 失败/未知 solver_status 全部拒绝')


def test_mid_event_fallback_rejected():
    rec = fake_record(0, complete=True, fallback_events=1)
    problems = validate_instance_record(rec, expected_for(0))
    assert any('fallback' in p for p in problems), problems
    print('  P1 中间事件 fallback（终局 complete）仍判协议失败')


# --------------------------------------------------------------------------
# P0-3：落盘校验先跑完整 common 合同 + trace replay
# --------------------------------------------------------------------------
def test_common_contract_valid_record_passes():
    problems = validate_instance_record(fake_record(0), expected_for(0))
    assert not problems, problems
    print('  P0-3 完整 common 合同 + trace replay 对完整记录通过')


def test_common_contract_structural_defect_rejected():
    """仅 common 合同可见的结构缺陷（非法 action_type），同步重算 hash 也逃不过。"""
    rec = fake_record(0)
    rec['actions'] = [{'event_id': 0, 'vehicle_id': 0, 'action_type': 'BOGUS',
                       'action_layer': 'execution', 'suffix_before': [],
                       'suffix_after': []}]
    rec['decision_hash'] = bc.decision_hash(rec)
    rec['artifact_hash'] = bc.artifact_hash(rec)
    problems = validate_instance_record(rec, expected_for(0))
    assert any('action_type' in p for p in problems), problems
    print('  P0-3 非法 action_type 被 common 合同拒绝（OR 特有检查不可见）')


# --------------------------------------------------------------------------
# 守卫层（续跑身份）
# --------------------------------------------------------------------------
def test_success_path_and_clean_resume():
    tmp = tempfile.mkdtemp()
    try:
        pre_run = {'config': build_effective_config(30, 2),
                   'compute_hash': CODE_HASH,
                   'dev_manifest_sha256': 'm' * 64,
                   'objective_profile_hash': PROFILE_HASH,
                   'ortools_environment': {'ortools_version': '9.11.4210'}}
        guard_output_dir(tmp, resume=False, pre_run=pre_run)
        guard_output_dir(tmp, resume=True, pre_run=pre_run)
        names = [f'{t}_{ed}' for t in ('r1', 'c1', 'rc1')
                 for ed in ('02', '05', '08')]
        for name in names:
            d = _cell_dir(tmp, name)
            recs = [fake_record(0), fake_record(1)]
            _write_instances(d, recs)
            _publish(d, name, recs)
        agg, _ = _aggregate(tmp, names)
        assert agg['verdict'] == 'PROTOCOL_PASS', agg
        print('  九 cell 成功路径 + 干净续跑（守卫层）通过')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_layer_identity_drift_rejected():
    """P1（OR7）：control/analysis/protocol_id 漂移 → 续跑拒绝。"""
    tmp = tempfile.mkdtemp()
    try:
        base = {'config': build_effective_config(30, 2),
                'compute_hash': CODE_HASH,
                'control_hash': CONTROL_HASH,
                'analysis_hash': ANALYSIS_HASH,
                'protocol_id': PROTOCOL_ID,
                'dev_manifest_sha256': 'm' * 64,
                'objective_profile_hash': PROFILE_HASH,
                'ortools_environment': {'ortools_version': '9.11.4210',
                                        'native_extension_sha256': FAKE_NATIVE}}
        guard_output_dir(tmp, resume=False, pre_run=base)
        for key in ('control_hash', 'analysis_hash', 'protocol_id'):
            drifted = copy.deepcopy(base)
            drifted[key] = 'x' * 64
            try:
                guard_output_dir(tmp, resume=True, pre_run=drifted)
                raise AssertionError(f'{key} 漂移未被拒绝')
            except RuntimeError as e:
                assert '漂移' in str(e), f'{key}: {e}'
        print('  续跑漂移（control/analysis/protocol_id）全部拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nonempty_dir_rejected():
    tmp = tempfile.mkdtemp()
    try:
        open(os.path.join(tmp, 'stray.txt'), 'w').close()
        try:
            guard_output_dir(tmp, resume=False, pre_run={'config': {}})
            raise AssertionError('非空目录未被拒绝')
        except RuntimeError as e:
            assert '拒绝接管' in str(e)
        print('  非空目录无 manifest 被拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_drift_rejected():
    tmp = tempfile.mkdtemp()
    try:
        base = {'config': build_effective_config(30, 2),
                'compute_hash': CODE_HASH,
                'dev_manifest_sha256': 'm' * 64,
                'objective_profile_hash': PROFILE_HASH,
                'ortools_environment': {'ortools_version': '9.11.4210',
                                        'native_extension_sha256': FAKE_NATIVE}}
        guard_output_dir(tmp, resume=False, pre_run=base)
        for label, mutate in (
                ('instances_per_cell', lambda b: b.__setitem__(
                    'config', build_effective_config(30, 3))),
                ('solution_limit', lambda b: b.__setitem__(
                    'config', build_effective_config(40, 2))),
                ('compute_hash', lambda b: b.__setitem__('compute_hash', 'x' * 64)),
                ('dev_manifest_sha256', lambda b: b.__setitem__(
                    'dev_manifest_sha256', 'x' * 64)),
                ('objective_profile_hash', lambda b: b.__setitem__(
                    'objective_profile_hash', 'x' * 64)),
                ('ortools_environment', lambda b: b.__setitem__(
                    'ortools_environment',
                    {'ortools_version': '9.11.4210',
                     'native_extension_sha256': 'y' * 64})),
        ):
            drifted = copy.deepcopy(base)
            mutate(drifted)
            try:
                guard_output_dir(tmp, resume=True, pre_run=drifted)
                raise AssertionError(f'{label} 漂移未被拒绝')
            except RuntimeError as e:
                assert '漂移' in str(e), f'{label}: {e}'
        print('  续跑漂移（config/compute/data/profile/environment）全部拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 发布层（实例集合 + 篡改）
# --------------------------------------------------------------------------
def test_null_truncated_non_dict_rejected():
    assert validate_instance_record(None, expected_for(0)), 'null 未被拒绝'
    assert validate_instance_record({'instance_id': 0}, expected_for(0)), \
        '截断未被拒绝'
    assert validate_instance_record('not-a-dict', expected_for(0)), \
        '非 dict 未被拒绝'
    print('  null/截断/非 dict 实例全部拒绝')


def test_missing_extra_instances_rejected():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0)]
        _write_instances(d, recs)
        try:
            publish_cell(d, 'r1_02', (0, 1), _expected_map(recs), DATA_SHA,
                         PROFILE_HASH, CODE_HASH, FULL_CHECKPOINT, RUN_ID)
            raise AssertionError('缺失实例未被拒绝')
        except RuntimeError as e:
            assert '实例集合不精确' in str(e)
        recs = [fake_record(0), fake_record(1), fake_record(2)]
        _write_instances(d, recs)
        try:
            publish_cell(d, 'r1_02', (0, 1), _expected_map(recs), DATA_SHA,
                         PROFILE_HASH, CODE_HASH, FULL_CHECKPOINT, RUN_ID)
            raise AssertionError('多出实例未被拒绝')
        except RuntimeError as e:
            assert '实例集合不精确' in str(e)
        print('  缺失/多出实例全部拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_publish_rejects_seed_scene_from_record():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0, seed=7777, scene='self_proved_scene')]
        _write_instances(d, recs)
        exp = {0: {'instance_seed': 1000, 'scene_instance_id': 'scene_0'}}
        try:
            publish_cell(d, 'r1_02', (0,), exp, DATA_SHA, PROFILE_HASH, CODE_HASH,
                         FULL_CHECKPOINT, RUN_ID)
            raise AssertionError('自证 seed/scene 未被拒绝')
        except RuntimeError as e:
            assert 'instance_seed' in str(e) or 'scene_instance_id' in str(e), e
        print('  P0-3 seed/scene 期望值来自 manifest，记录自证被拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tampered_summary_detected():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        with open(os.path.join(d, 'canonical.COMPLETE'), encoding='utf-8') as f:
            canonical = json.load(f)
        good_sha = canonical['summary_sha256']
        with open(os.path.join(d, 'summary.json'), 'r+', encoding='utf-8') as f:
            content = json.load(f)
            content['n_complete'] = 0
            f.seek(0)
            json.dump(content, f, ensure_ascii=False)
            f.truncate()
        assert sha256_file(os.path.join(d, 'summary.json')) != good_sha
        print('  篡改 summary → canonical 绑定 hash 可检出')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# P0-4：aggregator 校验 canonical.COMPLETE
# --------------------------------------------------------------------------
def test_aggregate_rejects_missing_marker():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        from ortools_cell_summary import rebuild_cell_summary, atomic_write_json
        s = rebuild_cell_summary(d, 'r1_02', (0, 1), _expected_map(recs),
                                 DATA_SHA, PROFILE_HASH, CODE_HASH, FULL_CHECKPOINT)
        atomic_write_json(os.path.join(d, 'summary.json'), s)
        atomic_write_json(os.path.join(d, 'manifest.json'), {
            'cell': 'r1_02', 'run_id': RUN_ID, 'n_instances': 2,
            'data_sha256': DATA_SHA, 'profile_hash': PROFILE_HASH,
            'code_hash': CODE_HASH, 'checkpoint_hash': FULL_CHECKPOINT,
            'published_at': 'now'})
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('缺 canonical.COMPLETE 未被拒绝')
        except RuntimeError as e:
            assert 'canonical.COMPLETE' in str(e)
        print('  P0-4 缺 canonical.COMPLETE 被 aggregator 拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aggregate_rejects_building_marker():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        from ortools_cell_summary import atomic_write_json
        atomic_write_json(os.path.join(d, 'BUILDING'), {'started_at': 'now'})
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('BUILDING 标记未被拒绝')
        except RuntimeError as e:
            assert 'BUILDING' in str(e)
        print('  P0-4 BUILDING 标记共存被 aggregator 拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aggregate_rejects_tampered_marker():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        with open(os.path.join(d, 'canonical.COMPLETE'), 'r+', encoding='utf-8') as f:
            c = json.load(f)
            c['summary_sha256'] = 'f' * 64
            f.seek(0)
            json.dump(c, f, ensure_ascii=False)
            f.truncate()
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('篡改 marker 未被拒绝')
        except RuntimeError as e:
            assert 'summary_sha256' in str(e)
        print('  P0-4 篡改 canonical.summary_sha256 被 aggregator 拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aggregate_rejects_manifest_tamper():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        with open(os.path.join(d, 'manifest.json'), 'r+', encoding='utf-8') as f:
            m = json.load(f)
            m['code_hash'] = 'x' * 64
            f.seek(0)
            json.dump(m, f, ensure_ascii=False)
            f.truncate()
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('manifest 篡改未被拒绝')
        except RuntimeError as e:
            assert 'manifest' in str(e) or 'code_hash' in str(e), e
        print('  P0-3 manifest 篡改被 aggregator 拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aggregate_rejects_instance_file_tamper():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        p = os.path.join(d, 'instances', 'inst_0.json')
        with open(p, 'r+', encoding='utf-8') as f:
            r = json.load(f)
            r['outcome']['distance_cost'] = 999999.0
            f.seek(0)
            json.dump(r, f, ensure_ascii=False)
            f.truncate()
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('实例文件篡改未被拒绝')
        except RuntimeError as e:
            assert 'instance_file_sha256' in str(e), e
        print('  P0-3 实例文件字节篡改被拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# P0-2：顶层复验 pre-run / P1-1：失败发布 / P1-2：parity 准入
# --------------------------------------------------------------------------
def test_verify_aggregate_rejects_pre_run_tamper():
    tmp = tempfile.mkdtemp()
    try:
        from ortools_cell_summary import atomic_write_json
        atomic_write_json(os.path.join(tmp, 'pre_run_manifest.json'),
                          {'run_id': 'r1'})
        atomic_write_json(os.path.join(tmp, 'canonical.COMPLETE'),
                          {'pre_run_manifest_sha256': 'f' * 64})
        try:
            verify_aggregate(tmp)
            raise AssertionError('pre-run 篡改未被拒绝')
        except RuntimeError as e:
            assert 'pre_run_manifest_sha256' in str(e), e
        print('  P0-2 顶层复验拒绝 pre-run 篡改')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_failed_publish_leaves_building():
    tmp = tempfile.mkdtemp()
    try:
        d = _cell_dir(tmp, 'r1_02')
        recs = [fake_record(0), fake_record(1)]
        _write_instances(d, recs)
        _publish(d, 'r1_02', recs)
        assert not os.path.exists(os.path.join(d, 'BUILDING'))
        # 破坏一个实例 → 重新发布失败
        with open(os.path.join(d, 'instances', 'inst_0.json'), 'w',
                  encoding='utf-8') as f:
            json.dump('not-a-dict', f)
        try:
            _publish(d, 'r1_02', recs)
            raise AssertionError('破坏实例后重新发布未失败')
        except RuntimeError:
            pass
        assert os.path.exists(os.path.join(d, 'BUILDING'))
        b = json.load(open(os.path.join(d, 'BUILDING'), encoding='utf-8'))
        assert b.get('status') == 'FAILED'
        try:
            _aggregate(tmp, ['r1_02'])
            raise AssertionError('BUILDING/FAILED 标记未被 aggregator 拒绝')
        except RuntimeError as e:
            assert 'BUILDING' in str(e), e
        print('  P1-1 失败发布保留 BUILDING/FAILED，聚合拒绝')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parity_admission_check():
    assert admission_check('p', 'p', 'r1', 'r2') == []
    assert any('run_id' in x for x in admission_check('p', 'p', 'r1', 'r1'))
    assert any('protocol_id' in x for x in admission_check('p', 'q', 'r1', 'r2'))
    assert any('缺 protocol_id' in x for x in admission_check(None, 'p', 'r1', 'r2'))
    print('  P1-2 parity 准入：protocol 同 / run 异通过，反之拒绝')


def test_identity_drift_stops():
    """P0-4：运行中 compute 身份漂移必须立即停止。"""
    from strict_online_runner import load_objective_profile, code_hash
    import identity as ortools_identity
    from ortools_adapter import ORToolsRHDAdapter
    _dcc = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..'))
    profile = load_objective_profile(PROFILE)
    env = ortools_identity.check_environment()
    frozen = code_hash(os.path.join(_dcc, 'ortools_adapter.py'),
                       ORToolsRHDAdapter.compute_files())
    verify_identity_still_intact([], profile, env, frozen, 'test-ok')
    try:
        verify_identity_still_intact([], profile, env, 'x' * 64, 'test-bad')
        raise AssertionError('compute 漂移未停止')
    except RuntimeError as e:
        assert 'compute_hash 漂移' in str(e), e
    print('  P0-4 运行中 compute 身份漂移被停止')


def main():
    test_effective_config_binds_actual_budget()
    test_protocol_id_deterministic()
    test_resume_invalid_record_refuses_overwrite()
    test_identity_drift_rejected()
    test_checkpoint_native_mismatch_rejected()
    test_outcome_tamper_with_stale_artifact_hash()
    test_stats_self_report_mismatch_rejected()
    test_solver_status_whitelist_rejected()
    test_mid_event_fallback_rejected()
    test_common_contract_valid_record_passes()
    test_common_contract_structural_defect_rejected()
    test_success_path_and_clean_resume()
    test_nonempty_dir_rejected()
    test_resume_drift_rejected()
    test_resume_layer_identity_drift_rejected()
    test_null_truncated_non_dict_rejected()
    test_missing_extra_instances_rejected()
    test_publish_rejects_seed_scene_from_record()
    test_tampered_summary_detected()
    test_aggregate_rejects_missing_marker()
    test_aggregate_rejects_building_marker()
    test_aggregate_rejects_tampered_marker()
    test_aggregate_rejects_manifest_tamper()
    test_aggregate_rejects_instance_file_tamper()
    test_verify_aggregate_rejects_pre_run_tamper()
    test_failed_publish_leaves_building()
    test_parity_admission_check()
    test_identity_drift_stops()
    print('PASS test_devproto_control')


if __name__ == '__main__':
    main()
