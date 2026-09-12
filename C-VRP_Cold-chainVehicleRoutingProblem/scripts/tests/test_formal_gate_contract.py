"""formal_gate_contract 契约与身份校验测试（纯 stdlib，不跑 oracle）。

覆盖：
  1. 合法 DEV-GATE（旧 schema）/ VAL（新 schema）manifest 通过；
  2. role / split_role 不一致、九 cell 缺失/重复/多余、每 cell 非 128、
     seed 缺失/重复/九 cell 不一致、scene id 重复 → 全部拒绝；
  3. check_disjoint：instance_seeds / scene_instance_ids 重叠被检测，无重叠通过；
  4. _validate_frozen_compute：role / hash / config 任一漂移拒绝；
  5. recompute_profile_hash 自洽。

用法：python scripts/tests/test_formal_gate_contract.py
"""
import json
import os
import sys
import tempfile

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_EVAL = os.path.join(_BASE, 'evaluation')
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import formal_gate_contract as contract
from formal_gate_contract import ContractError, TYPES, EDODS

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _cell(t, e, n=128, seeds=None, sid_prefix='x'):
    seeds = list(seeds) if seeds is not None else list(range(1000, 1000 + n))
    return {'type': t, 'edod': e, 'file': f'dcc_50_{t.lower()}_edod{int(e*10):02d}_x.npz',
            'sha256': 'a' * 64, 'num_instances': n, 'seed': 123456,
            'scene_instance_ids': [f'{sid_prefix}__{i}' for i in range(n)],
            'instance_seeds': seeds}


def _manifest(role='val', n=128, seeds=None, cells=None, new_schema=True, sid_prefix='x'):
    if seeds is None:
        seeds = list(range(1000, 1000 + n))
    if cells is None:
        cells = [_cell(t, e, n, seeds, sid_prefix=sid_prefix) for t in TYPES for e in EDODS]
    m = {'split_role': role, 'problem_size': 50, 'capacity': 50, 'cells': cells}
    if new_schema:
        m['schema'] = 'o0cc-split-manifest-v1'
        m['root_seed'] = 123456
        m['instances_per_cell'] = n
        m['generator_identity'] = {'generator': 'x', 'sha256': '0' * 64}
        m['contract_identity'] = {'contract_hash': 'c' * 64}
        m['profile_identity'] = {'profile_hash': 'p' * 64}
    else:
        m['seed'] = 123456
        m['num_instances_per_cell'] = n
    return m


def _expect_ok(manifest, role):
    try:
        contract.validate_manifest(manifest, role)
        return True, None
    except ContractError as e:
        return False, str(e)


def _expect_fail(manifest, role):
    try:
        contract.validate_manifest(manifest, role)
        return False, '未拒绝'
    except ContractError:
        return True, None


def test_valid_manifests():
    ok = True
    ok &= _expect_ok(_manifest('dev_gate', new_schema=False), 'dev_gate')[0]
    ok &= _expect_ok(_manifest('val', new_schema=True), 'val')[0]
    record('valid_manifests', ok)
    return ok


def test_role_split_role_mismatch():
    ok = _expect_fail(_manifest('dev_gate'), 'val')[0]      # val role + dev_gate split
    ok &= _expect_fail(_manifest('val'), 'dev_gate')[0]     # dev_gate role + val split
    ok &= _expect_fail(_manifest('dev_gate'), 'regression')[0]  # 非正式角色
    record('role_split_role_mismatch', ok)
    return ok


def test_cell_count():
    cells = [_cell(t, e) for t in TYPES for e in EDODS]
    ok = _expect_fail(_manifest(cells=cells[:8]), 'val')[0]           # 缺一个
    ok &= _expect_fail(_manifest(cells=cells + cells[:1]), 'val')[0]  # 多余
    dup = list(cells); dup[1] = cells[0]
    ok &= _expect_fail(_manifest(cells=dup), 'val')[0]                # 重复 cell
    record('cell_count', ok)
    return ok


def test_instances_per_cell():
    ok = _expect_fail(_manifest(n=127), 'val')[0]
    ok &= _expect_fail(_manifest(n=129), 'val')[0]
    record('instances_per_cell', ok)
    return ok


def test_seed_integrity():
    # seed 不足
    m = _manifest(); m['cells'][0]['instance_seeds'] = m['cells'][0]['instance_seeds'][:-1]
    ok = _expect_fail(m, 'val')[0]
    # seed 重复
    m = _manifest(); s = list(m['cells'][0]['instance_seeds']); s[5] = s[0]
    for c in m['cells']:
        c['instance_seeds'] = list(s)
    ok &= _expect_fail(m, 'val')[0]
    # 九 cell 不一致
    m = _manifest(); m['cells'][3]['instance_seeds'] = list(range(2000, 2128))
    ok &= _expect_fail(m, 'val')[0]
    # scene id 重复
    m = _manifest(); c0 = m['cells'][0]
    c0['scene_instance_ids'] = ['same'] * 128
    ok &= _expect_fail(m, 'val')[0]
    record('seed_integrity', ok)
    return ok


def test_disjoint():
    tmp = tempfile.mkdtemp(prefix='contract_disjoint_')
    try:
        this = _manifest('val', seeds=list(range(1000, 1128)), sid_prefix='v')
        # 与 this 共享 seeds → 应检测 seed 重叠（scene id 用不同前缀，避免误判）
        overlap_manifest = _manifest('dev_gate', seeds=list(range(1050, 1178)), sid_prefix='d')
        p = os.path.join(tmp, 'overlap.json')
        json.dump(overlap_manifest, open(p, 'w'))
        report = contract.check_disjoint(this, [p])
        ok = report[0]['seed_overlap'] != []
        # 完全不重叠（seeds 与 scene id 都不同）→ 无重叠
        clean_manifest = _manifest('dev_gate', seeds=list(range(2000, 2128)), sid_prefix='d')
        p2 = os.path.join(tmp, 'clean.json')
        json.dump(clean_manifest, open(p2, 'w'))
        report2 = contract.check_disjoint(this, [p2])
        ok &= report2[0]['seed_overlap'] == [] and report2[0]['scene_id_overlap'] == []
        record('disjoint', ok, f'overlap={len(report[0]["seed_overlap"])}')
        return ok
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_frozen_compute_drift():
    cells = [_cell(t, e) for t in TYPES for e in EDODS]
    base = contract._frozen_compute(cells, 128, 'c' * 64, {}, 'p' * 64, role='val')
    ok = True
    # role 漂移
    r = dict(base); r['role'] = 'dev_gate'
    try:
        contract._validate_frozen_compute(base, r); ok = False
    except ContractError:
        pass
    # compute hash 漂移
    r = dict(base); r['compute_sha256'] = '0' * 64
    try:
        contract._validate_frozen_compute(base, r); ok = False
    except ContractError:
        pass
    # config 漂移
    r = dict(base); r['config'] = dict(base['config']); r['config']['baseline'] = 'WRONG'
    try:
        contract._validate_frozen_compute(base, r); ok = False
    except ContractError:
        pass
    # 完全一致 → 通过
    try:
        contract._validate_frozen_compute(base, dict(base))
    except ContractError:
        ok = False
    record('frozen_compute_drift', ok)
    return ok


def test_profile_hash_self_consistent():
    tmp = tempfile.mkdtemp(prefix='contract_profile_')
    try:
        payload = {'name': 't', 'distance_scale': 18.0, 'quality_scale': 2.9,
                   'energy_scale': 1193.0, 'lambda_quality': 1.0, 'lambda_energy': 1.0,
                   'scale_source': 'pilot', 'dev_statistics': None}
        import hashlib
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True).encode()).hexdigest()
        payload['profile_hash'] = h
        p = os.path.join(tmp, 'profile.json')
        json.dump(payload, open(p, 'w'))
        ok = contract.recompute_profile_hash(p) == h
        record('profile_hash_self_consistent', ok)
        return ok
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_requires_schema_and_root_seed():
    # 缺 schema
    m = _manifest('val', new_schema=True); del m['schema']
    ok = _expect_fail(m, 'val')[0]
    # 缺 root_seed（回退旧 seed 不允许）
    m = _manifest('val', new_schema=True); del m['root_seed']; m['seed'] = 123
    ok &= _expect_fail(m, 'val')[0]
    # 缺 identity 字段
    m = _manifest('val', new_schema=True); del m['contract_identity']
    ok &= _expect_fail(m, 'val')[0]
    record('val_requires_schema_and_root_seed', ok)
    return ok


def test_frozen_compute_roleless_old_val():
    """blocking item 1：roleless 旧 plan 只允许 dev_gate 续跑，不得用作 VAL。"""
    cells = [_cell(t, e) for t in TYPES for e in EDODS]
    base = contract._frozen_compute(cells, 128, 'c' * 64, {}, 'p' * 64, role='val')
    old = dict(base); old.pop('role')
    try:
        contract._validate_frozen_compute(old, base)
        ok = False
    except ContractError:
        ok = True
    # roleless 旧 plan + dev_gate new → 允许（旧 DEV-GATE v4 无 role 字段）
    dg = contract._frozen_compute(cells, 128, 'c' * 64, {}, 'p' * 64, role='dev_gate')
    old_dg = dict(dg); old_dg.pop('role')
    try:
        contract._validate_frozen_compute(old_dg, dg)
        ok &= True
    except ContractError:
        ok &= False
    record('frozen_compute_roleless_old_val', ok)
    return ok


def _registered_manifest(role, seed_start):
    seeds = list(range(seed_start, seed_start + 128))
    return {'split_role': role, 'root_seed': seed_start, 'instances_per_cell': 128,
            'cells': [{'type': 'R1', 'edod': 0.2, 'file': 'x.npz', 'sha256': '0' * 64,
                       'instance_seeds': seeds,
                       'scene_instance_ids': [f'{role}__{i}' for i in range(128)]}]}


def _write_registry(tmp, manifests):
    splits = []
    for role, m in manifests:
        p = os.path.join(tmp, f'{role}.json')
        json.dump(m, open(p, 'w'))
        splits.append({'split_id': f'reg-{role}', 'split_role': role,
                       'manifest': p, 'manifest_sha256': contract.sha256_file(p)})
    reg_path = os.path.join(tmp, 'registry.json')
    json.dump({'schema': contract.SPLIT_REGISTRY_SCHEMA, 'splits': splits},
              open(reg_path, 'w'))
    return reg_path


def test_split_registry():
    tmp = tempfile.mkdtemp(prefix='contract_reg_')
    try:
        this = _manifest('val', seeds=list(range(1000, 1128)), sid_prefix='v')
        manifests = [('train', _registered_manifest('train', 3000)),
                     ('dev_cal', _registered_manifest('dev_cal', 4000)),
                     ('dev_proto', _registered_manifest('dev_proto', 5000)),
                     ('dev_gate', _registered_manifest('dev_gate', 2000))]
        reg_path = _write_registry(tmp, manifests)
        report, _ = contract.check_registry_disjoint(this, reg_path)
        ok = all(e['seed_overlap'] == [] and e['scene_id_overlap'] == []
                 for e in report)
        # 只登记 dev_gate（未覆盖 train/dev_cal/dev_proto）→ 拒绝
        reg_partial = os.path.join(tmp, 'reg_partial.json')
        json.dump({'schema': contract.SPLIT_REGISTRY_SCHEMA,
                   'splits': [{'split_id': 'g', 'split_role': 'dev_gate',
                               'manifest': manifests[-1][1] and
                               os.path.join(tmp, 'dev_gate.json'),
                               'manifest_sha256': contract.sha256_file(
                                   os.path.join(tmp, 'dev_gate.json'))}]},
                  open(reg_partial, 'w'))
        try:
            contract.check_registry_disjoint(this, reg_partial); ok = ok and False
        except ContractError:
            pass
        # 缺 manifest_sha256 → load_split_registry 拒绝
        reg_nohash = os.path.join(tmp, 'reg_nohash.json')
        json.dump({'schema': contract.SPLIT_REGISTRY_SCHEMA,
                   'splits': [{'split_id': 'x', 'split_role': 'dev_gate',
                               'manifest': 'y.json'}]}, open(reg_nohash, 'w'))
        try:
            contract.load_split_registry(reg_nohash); ok = ok and False
        except ContractError:
            pass
        record('split_registry', ok)
        return ok
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_identity_drift():
    """VAL 冻结身份（manifest hash / root_seed / scene / contract / registry）漂移必须拒绝。"""
    cells = [_cell(t, e) for t in TYPES for e in EDODS]
    m = _manifest('val')
    vi1 = contract.build_val_identity(m, 'm' * 64, ['d' * 64])
    a = contract._frozen_compute(cells, 128, 'c' * 64, {}, 'p' * 64, role='val', val_identity=vi1)
    ok = True
    for k in ('manifest_sha256', 'root_seed', 'scene_ids_sha256', 'contract_hash'):
        vi2 = dict(vi1); vi2[k] = '0' * 64 if k != 'root_seed' else 999999
        b = dict(a); b['val_identity'] = vi2
        try:
            contract._validate_frozen_compute(a, b); ok = False
        except ContractError:
            pass
    # 完全一致 → 通过
    try:
        contract._validate_frozen_compute(a, dict(a))
    except ContractError:
        ok = False
    record('val_identity_drift', ok)
    return ok


def test_verify_active_code():
    # 错误 seal hash → 拒绝
    bad = {'compute_sha256': '0' * 64, 'control_sha256': '0' * 64, 'analysis_sha256': '0' * 64}
    try:
        contract.verify_active_code(bad)
        ok = False
    except ContractError:
        ok = True
    # 正确 seal hash → 通过
    good = {'compute_sha256': contract._version(contract.COMPUTE_FILES)[0],
            'control_sha256': contract._version(contract.CONTROL_FILES)[0],
            'analysis_sha256': contract._version(contract.ANALYSIS_FILES)[0]}
    try:
        contract.verify_active_code(good)
    except ContractError:
        ok = False
    record('verify_active_code', ok)
    return ok


def test_statistical_plan():
    plan = {'schema': 'o0cc-statistical-plan-v1',
            'primary_endpoint': 'nine_cell_equal_weight_paired_delta_jcc',
            'n_boot': 10000, 'boot_seed': 42,
            'cell_weight': 'equal', 'grouping': 'internal_seed_group',
            'confidence_level': 0.95, 'ci_method': 'percentile',
            'alternative': 'less_than_zero',
            'go_rule': 'mean_lt_zero_and_upper_ci_lt_zero',
            'local_endpoint_role': 'auxiliary', 'report_dqe_components': True,
            'service_gate_required': True, 'fail_closed_on_error': True,
            'nominal_is_primary': True, 'sensitivity_non_gating': True,
            'single_cell_non_gating': True}
    ok = contract.validate_statistical_plan(plan, contract.FROZEN_CONFIG) is None
    for key, val in (('n_boot', 1), ('boot_seed', 1), ('cell_weight', 'weighted'),
                     ('grouping', 'instance'), ('primary_endpoint', 'x'),
                     ('confidence_level', 0.90), ('ci_method', 'normal'),
                     ('alternative', 'greater'), ('go_rule', 'any'),
                     ('local_endpoint_role', 'primary'), ('report_dqe_components', False),
                     ('service_gate_required', False), ('fail_closed_on_error', False),
                     ('nominal_is_primary', False), ('sensitivity_non_gating', False),
                     ('single_cell_non_gating', False)):
        bad = dict(plan); bad[key] = val
        try:
            contract.validate_statistical_plan(bad, contract.FROZEN_CONFIG); ok = False
        except ContractError:
            pass
    record('statistical_plan', ok)
    return ok


def test_contract_identity_drift():
    """显式 contract 的冻结身份（raw/effective hash + 文件 hash）漂移必须拒绝。"""
    cells = [_cell(t, e) for t in TYPES for e in EDODS]
    ci = {'contract_hash': 'c' * 64, 'contract_file_sha256': 'f' * 64,
          'profile_file_sha256': 'p' * 64, 'effective_contract_hash': 'e' * 64}
    a = contract._frozen_compute(cells, 128, 'c' * 64, {}, 'p' * 64, role='dev_gate',
                                 contract_identity=ci)
    ok = True
    for k in ci:
        ci2 = dict(ci); ci2[k] = '0' * 64
        b = dict(a); b['contract_identity'] = ci2
        try:
            contract._validate_frozen_compute(a, b); ok = False
        except ContractError:
            pass
    # 完全一致 → 通过
    try:
        contract._validate_frozen_compute(a, dict(a))
    except ContractError:
        ok = False
    record('contract_identity_drift', ok)
    return ok


def main():
    ok = [test_valid_manifests(), test_role_split_role_mismatch(), test_cell_count(),
          test_instances_per_cell(), test_seed_integrity(), test_disjoint(),
          test_val_requires_schema_and_root_seed(), test_frozen_compute_roleless_old_val(),
          test_split_registry(), test_val_identity_drift(), test_verify_active_code(),
          test_statistical_plan(), test_contract_identity_drift(), test_frozen_compute_drift(),
          test_profile_hash_self_consistent()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
