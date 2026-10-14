"""O0-CC VAL 启动链端到端测试（freezer → archive → run_o0cc_val，不跑 oracle rollout）。

在临时目录生成 9 份伪 NPZ + VAL_MANIFEST（统一 schema + identity）+ profile + 完整
split registry + 显式 pilot contract + 预注册统计方案，调真实 `freeze_source_archive.py`
封存成自包含 archive，再调真实 `run_o0cc_val.py --archive`（预写 9×128 合成实例，避免跑
oracle），验证：
  - archive happy path（`val_summary.json` 产出，total=1152 / seed_groups=128 / mode=val）；
  - archive 搬到另一目录后仍可运行；
  - 缺 / 错 `--expected-bundle-sha256` 拒绝；archive 缺 NPZ 拒绝；
  - registry 缺预注册 split / 缺 hash 拒绝；
  - VAL 缺 contract / 缺统计方案 / contract-profile 三方不一致 / 统计方案与配置不一致 拒绝；
  - `run_formal_gate.py --role val` 拒绝（非 archive 不能产出正式 VAL verdict）。

用法：python scripts/tests/test_val_control_chain.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
_EVAL = os.path.join(_CVRPTW, 'scripts', 'evaluation')
_COLD = os.path.join(_CVRPTW, 'scripts', 'coldchain')
for p in (_EVAL, _BASE, _COLD):
    if p not in sys.path:
        sys.path.insert(0, p)

import formal_gate_contract as fg
from coldchain_contract import (write_pilot_contract, load_coldchain_contract,
                                ObjectiveProfile, apply_objective_profile)

FREEZER = os.path.join(_EVAL, 'freeze_source_archive.py')
RUN_VAL = os.path.join(_EVAL, 'run_o0cc_val.py')
RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _outcome(cost, contract_hash=None):
    return {
        'complete': True, 'n_unserved': 0, 'n_duplicate': 0,
        'tw_feasible': True, 'capacity_feasible': True, 'depot_return_feasible': True,
        'distance_cost': cost, 'distance_km': cost, 'quality_loss': 1.0,
        'energy_kwh': 5.0, 'coldchain_cost': cost,
        'temperature_hard_feasible': True, 'all_orders_picked': True,
        'all_cargo_delivered_to_depot': True, 'terminal_manifests_empty': True,
        'trace_accounting_consistent': True, 'distance_accounting_consistent': True,
        'ownership_violations': 0, 'terminal_unresolved': 0,
        'num_unsalable': 0, 'thermal_violation_count': 0, 'thermal_violation_duration_h': 0.0,
        'contract_hash': contract_hash,
    }


def _instance(idx, runner_hash, seed, contract_hash=None):
    b = 3.2 + (seed % 5) * 0.01
    o = b - 0.05
    ed = -0.05
    return {
        'inst_idx': idx, 'baseline': _outcome(b, contract_hash),
        'oracle': _outcome(o, contract_hash),
        'local': {'event_deltas': [ed],
                  'records': [{'event_id': 0, 'n_customers': 2, 'n_customer_deltas': 2,
                               'mean_customer_delta': ed, 'keep_cost': b}],
                  'base_cost': b},
        'log': [], 'runtime': 1.0, 'code_sha256': runner_hash, 'error': None,
    }


def _registered_manifest(role, seed_start):
    seeds = list(range(seed_start, seed_start + 128))
    return {'split_role': role, 'root_seed': seed_start, 'instances_per_cell': 128,
            'cells': [{'type': 'R1', 'edod': 0.2, 'file': 'x.npz', 'sha256': '0' * 64,
                       'instance_seeds': seeds,
                       'scene_instance_ids': [f'{role}__{i}' for i in range(128)]}]}


def _write_registry(tmp, offsets):
    splits = []
    for role, start in offsets.items():
        m = _registered_manifest(role, start)
        p = os.path.join(tmp, 'reg', f'{role}.json')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(m, open(p, 'w'))
        splits.append({'split_id': f'reg-{role}', 'split_role': role,
                       'manifest': p, 'manifest_sha256': fg.sha256_file(p)})
    reg_path = os.path.join(tmp, 'split_registry.json')
    json.dump({'schema': fg.SPLIT_REGISTRY_SCHEMA, 'splits': splits},
              open(reg_path, 'w'))
    return reg_path


def _pilot_contract(tmp):
    p = os.path.join(tmp, 'contract.json')
    write_pilot_contract(p)
    return p


def _statistical_plan(tmp, n_boot=10000, boot_seed=42):
    plan = {'schema': 'o0cc-statistical-plan-v1',
            'primary_endpoint': 'nine_cell_equal_weight_paired_delta_jcc',
            'n_boot': n_boot, 'boot_seed': boot_seed,
            'cell_weight': 'equal', 'grouping': 'internal_seed_group',
            'confidence_level': 0.95, 'ci_method': 'percentile',
            'alternative': 'less_than_zero',
            'go_rule': 'mean_lt_zero_and_upper_ci_lt_zero',
            'local_endpoint_role': 'auxiliary', 'report_dqe_components': True,
            'service_gate_required': True, 'fail_closed_on_error': True,
            'nominal_is_primary': True, 'sensitivity_non_gating': True,
            'single_cell_non_gating': True}
    p = os.path.join(tmp, 'statistical_analysis_plan.json')
    json.dump(plan, open(p, 'w'))
    return p


def _setup(tmp, val_seeds=None, registry_offsets=None):
    """生成 VAL 数据 + manifest + profile + registry + contract + 统计方案。

    contract 三方身份一致（manifest.contract_identity == profile.provenance == loaded hash）。
    返回 (manifest, profile, registry, contract, statistical_plan)。
    """
    if val_seeds is None:
        val_seeds = list(range(1000, 1128))
    if registry_offsets is None:
        registry_offsets = {'train': 3000, 'dev_cal': 4000, 'dev_proto': 5000, 'dev_gate': 2000}

    contract_path = _pilot_contract(tmp)
    contract_hash = load_coldchain_contract(contract_path).contract_hash

    data_dir = os.path.join(tmp, 'data'); os.makedirs(data_dir)
    cells = []
    for t in ('R1', 'C1', 'RC1'):
        for e in (0.2, 0.5, 0.8):
            fname = f'dcc_50_{t.lower()}_edod{int(e*10):02d}_val.npz'
            npz = os.path.join(data_dir, fname)
            with open(npz, 'wb') as f:
                f.write(os.urandom(64))
            cells.append({'type': t, 'edod': e, 'file': fname,
                          'sha256': fg.sha256_file(npz), 'num_instances': 128,
                          'seed': 123456, 'scene_instance_ids': list(range(128)),
                          'instance_seeds': list(val_seeds)})

    prof_path = os.path.join(tmp, 'objective_profile.json')
    payload = {'name': 'test-profile', 'distance_scale': 18.0, 'quality_scale': 2.9,
               'energy_scale': 1193.0, 'lambda_quality': 1.0, 'lambda_energy': 1.0,
               'scale_source': 'pilot', 'dev_statistics': None,
               'provenance': {'contract_hash': contract_hash}}
    json.dump(payload, open(prof_path, 'w'))
    payload['profile_hash'] = fg.recompute_profile_hash(prof_path)
    json.dump(payload, open(prof_path, 'w'))
    profile_hash = payload['profile_hash']

    manifest = {'schema': 'o0cc-split-manifest-v1', 'split_role': 'val',
                'root_seed': 123456, 'problem_size': 50, 'capacity': 50,
                'instances_per_cell': 128, 'cells': cells,
                'generator_identity': {'generator': 'test', 'sha256': '0' * 64},
                'contract_identity': {'contract_hash': contract_hash},
                'profile_identity': {'profile_hash': profile_hash},
                'disjoint_from': []}
    manifest_path = os.path.join(data_dir, 'VAL_MANIFEST.json')
    json.dump(manifest, open(manifest_path, 'w'))

    registry = _write_registry(tmp, registry_offsets)
    stat_plan = _statistical_plan(tmp)
    return manifest_path, prof_path, registry, contract_path, stat_plan


def _freeze(manifest, profile, registry, contract, stat_plan, archive):
    cmd = [sys.executable, FREEZER, '--role', 'val', '--manifest', manifest,
           '--profile', profile, '--split-registry', registry,
           '--contract', contract, '--statistical-plan', stat_plan, '--out', archive]
    return subprocess.run(cmd, capture_output=True, text=True)


def _prepopulate(archive, out_dir):
    manifest_path = os.path.join(archive, 'inputs', 'manifest.json')
    profile_path = os.path.join(archive, 'inputs', 'objective_profile.json')
    registry_path = os.path.join(archive, 'inputs', 'split_registry.json')
    manifest_obj = fg.load_json(manifest_path)
    cells, n = fg.validate_manifest(manifest_obj, 'val')
    profile_hash = fg.load_json(profile_path)['profile_hash']
    compute_sha256, compute_files = fg._version(fg.COMPUTE_FILES)
    control_sha256, _ = fg._version(fg.CONTROL_FILES)
    analysis_sha256, _ = fg._version(fg.ANALYSIS_FILES)
    runner_hash = compute_files['run_action_oracle.py']
    data = {c['file']: c['sha256'] for c in cells}
    reg = fg.load_split_registry(registry_path)
    disjoint_sha256s = sorted(s['manifest_sha256'] for s in reg['splits'])
    val_identity = fg.build_val_identity(manifest_obj, fg.sha256_file(manifest_path),
                                               disjoint_sha256s,
                                               fg.sha256_file(registry_path))
    val_identity['statistical_plan_sha256'] = fg.sha256_file(
        os.path.join(archive, 'inputs', 'statistical_analysis_plan.json'))
    val_identity['contract_file_sha256'] = fg.sha256_file(
        os.path.join(archive, 'inputs', 'contract.json'))
    # contract_identity（与 driver 一致）：raw/effective hash + 文件 hash
    profile_obj = fg.load_json(profile_path)
    loaded_contract = load_coldchain_contract(os.path.join(archive, 'inputs', 'contract.json'))
    prof = ObjectiveProfile(
        name=profile_obj['name'], distance_scale=float(profile_obj['distance_scale']),
        quality_scale=float(profile_obj['quality_scale']),
        energy_scale=float(profile_obj['energy_scale']),
        lambda_quality=float(profile_obj['lambda_quality']),
        lambda_energy=float(profile_obj['lambda_energy']),
        scale_source=profile_obj.get('scale_source', 'pilot'),
        dev_statistics=profile_obj.get('dev_statistics'))
    contract_identity = {
        'contract_hash': loaded_contract.contract_hash,
        'contract_file_sha256': fg.sha256_file(os.path.join(archive, 'inputs', 'contract.json')),
        'profile_file_sha256': fg.sha256_file(profile_path),
        'effective_contract_hash': apply_objective_profile(loaded_contract, prof).contract_hash,
    }
    frozen = fg._frozen_compute(cells, n, compute_sha256, data, profile_hash,
                                role='val', val_identity=val_identity,
                                contract_identity=contract_identity)
    for c in cells:
        tag = fg.cell_tag(c)
        inst_dir = os.path.join(out_dir, tag, 'instances')
        os.makedirs(inst_dir, exist_ok=True)
        for i in range(n):
            json.dump(_instance(i, runner_hash, c['instance_seeds'][i],
                                contract_identity['effective_contract_hash']),
                      open(os.path.join(inst_dir, f'inst_{i}.json'), 'w'))
    pre = {'run': {'run_id': 'test-archive-run', 'timestamp_utc': 'x', 'command': 'test',
                   'data_dir': os.path.join(archive, 'inputs'), 'profile': profile_path,
                   'workers': 1, 'out': out_dir, 'instances_per_cell': n,
                   'total_instances': n * 9},
           **frozen,
           'compute_files': compute_files, 'control_sha256': control_sha256,
           'control_files': {}, 'analysis_sha256': analysis_sha256, 'analysis_files': {},
           'env': {}}
    json.dump(pre, open(os.path.join(out_dir, 'pre_run_manifest.json'), 'w'))


def _run_val(archive, bundle, out_dir, expected=None):
    cmd = [sys.executable, RUN_VAL, '--archive', archive,
           '--expected-bundle-sha256', expected if expected is not None else bundle,
           '--workers', '1', '--out', out_dir]
    return subprocess.run(cmd, capture_output=True, text=True)


def _bundle(archive):
    return json.load(open(os.path.join(archive, 'manifests', '归档封存清单.json')))['bundle_sha256']


def test_archive_happy_path():
    tmp = tempfile.mkdtemp(prefix='val_archive_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        r = _freeze(manifest, profile, registry, contract, stat_plan, archive)
        if r.returncode != 0:
            record('val_archive_happy_path', False, f'freeze fail {r.returncode} {r.stderr[:200]}')
            return False
        out_dir = os.path.join(tmp, 'out')
        _prepopulate(archive, out_dir)
        r = _run_val(archive, _bundle(archive), out_dir)
        summ = os.path.join(out_dir, 'val_summary.json')
        s = json.load(open(summ)) if os.path.exists(summ) else {}
        ok = r.returncode == 0 and s.get('total_instances') == 1152 \
             and s.get('n_seed_groups') == 128 and s.get('mode') == 'val'
        record('val_archive_happy_path', ok,
               f"exit={r.returncode} mode={s.get('mode')} total={s.get('total_instances')}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_moved():
    tmp = tempfile.mkdtemp(prefix='val_moved_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        bundle = _bundle(archive)
        moved = os.path.join(tmp, 'moved_archive')
        shutil.move(archive, moved)
        out_dir = os.path.join(tmp, 'out')
        _prepopulate(moved, out_dir)
        r = _run_val(moved, bundle, out_dir)
        ok = r.returncode == 0 and os.path.exists(os.path.join(out_dir, 'val_summary.json'))
        record('val_archive_moved', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_wrong_bundle():
    tmp = tempfile.mkdtemp(prefix='val_bundle_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        out_dir = os.path.join(tmp, 'out')
        r = _run_val(archive, '0' * 64, out_dir)
        ok = r.returncode != 0
        record('val_archive_wrong_bundle', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_missing_bundle_arg():
    tmp = tempfile.mkdtemp(prefix='val_nobundle_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        cmd = [sys.executable, RUN_VAL, '--archive', archive, '--workers', '1',
               '--out', os.path.join(tmp, 'out')]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode != 0
        record('val_archive_missing_bundle_arg', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_missing_npz():
    tmp = tempfile.mkdtemp(prefix='val_nonpz_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        bundle = _bundle(archive)
        npz = os.path.join(archive, 'inputs', 'dcc_50_r1_edod02_val.npz')
        os.chmod(npz, 0o644); os.remove(npz)
        r = _run_val(archive, bundle, os.path.join(tmp, 'out'))
        ok = r.returncode != 0
        record('val_archive_missing_npz', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_registry_missing_split():
    tmp = tempfile.mkdtemp(prefix='val_regmiss_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        registry3 = _write_registry(tmp, {'train': 3000, 'dev_cal': 4000, 'dev_proto': 5000})
        archive = os.path.join(tmp, 'archive')
        r = _freeze(manifest, profile, registry3, contract, stat_plan, archive)
        ok = r.returncode != 0
        record('val_registry_missing_split', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_registry_missing_hash():
    tmp = tempfile.mkdtemp(prefix='val_reghash_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        reg_path = os.path.join(tmp, 'bad_registry.json')
        json.dump({'schema': fg.SPLIT_REGISTRY_SCHEMA,
                   'splits': [{'split_id': 'x', 'split_role': 'dev_gate',
                               'manifest': 'y.json'}]}, open(reg_path, 'w'))
        archive = os.path.join(tmp, 'archive')
        r = _freeze(manifest, profile, reg_path, contract, stat_plan, archive)
        ok = r.returncode != 0
        record('val_registry_missing_hash', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_missing_contract():
    tmp = tempfile.mkdtemp(prefix='val_nocontract_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        cmd = [sys.executable, FREEZER, '--role', 'val', '--manifest', manifest,
               '--profile', profile, '--split-registry', registry,
               '--statistical-plan', stat_plan, '--out', archive]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode != 0
        record('val_missing_contract', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_missing_statistical_plan():
    tmp = tempfile.mkdtemp(prefix='val_noplan_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        cmd = [sys.executable, FREEZER, '--role', 'val', '--manifest', manifest,
               '--profile', profile, '--split-registry', registry,
               '--contract', contract, '--out', archive]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode != 0
        record('val_missing_statistical_plan', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_contract_profile_mismatch():
    tmp = tempfile.mkdtemp(prefix='val_contractmm_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        # 篡改 profile 的 provenance.contract_hash → 三方不一致 → 拒绝
        p = json.load(open(profile))
        p['provenance']['contract_hash'] = '0' * 64
        json.dump(p, open(profile, 'w'))
        archive = os.path.join(tmp, 'archive')
        r = _freeze(manifest, profile, registry, contract, stat_plan, archive)
        ok = r.returncode != 0
        record('val_contract_profile_mismatch', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_statistical_plan_mismatch():
    tmp = tempfile.mkdtemp(prefix='val_planmm_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        bad_plan = _statistical_plan(tmp, n_boot=1)  # 与 FROZEN_CONFIG n_boot=10000 不符
        archive = os.path.join(tmp, 'archive')
        r = _freeze(manifest, profile, registry, contract, bad_plan, archive)
        ok = r.returncode != 0
        record('val_statistical_plan_mismatch', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_formal_gate_role_val_rejected():
    tmp = tempfile.mkdtemp(prefix='val_noarchive_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        cmd = [sys.executable, os.path.join(_EVAL, 'run_formal_gate.py'),
               '--role', 'val', '--manifest', manifest, '--profile', profile,
               '--workers', '1', '--out', os.path.join(tmp, 'out')]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode != 0
        record('val_formal_gate_role_val_rejected', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_seal_tamper():
    """只篡改 seal 顶层 compute_sha256（不在 bundle 内）→ 与源码冻结清单交叉校验拒绝。"""
    tmp = tempfile.mkdtemp(prefix='val_sealtamper_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        bundle = _bundle(archive)
        seal_path = os.path.join(archive, 'manifests', '归档封存清单.json')
        os.chmod(seal_path, 0o644)
        seal = json.load(open(seal_path))
        seal['compute_sha256'] = '0' * 64
        json.dump(seal, open(seal_path, 'w'))
        r = _run_val(archive, bundle, os.path.join(tmp, 'out'))
        ok = r.returncode != 0
        record('val_archive_seal_tamper', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_missing_sealed():
    tmp = tempfile.mkdtemp(prefix='val_nosealed_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        bundle = _bundle(archive)
        sealed = os.path.join(archive, 'ARCHIVE_SEALED')
        os.chmod(sealed, 0o644)
        os.remove(sealed)
        r = _run_val(archive, bundle, os.path.join(tmp, 'out'))
        ok = r.returncode != 0
        record('val_archive_missing_sealed', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_val_contract_hash_tamper():
    """篡改一个实例的 oracle.contract_hash → 续跑验收必须拒绝（effective contract 绑定）。"""
    tmp = tempfile.mkdtemp(prefix='val_chtamper_')
    try:
        manifest, profile, registry, contract, stat_plan = _setup(tmp)
        archive = os.path.join(tmp, 'archive')
        _freeze(manifest, profile, registry, contract, stat_plan, archive)
        out_dir = os.path.join(tmp, 'out')
        _prepopulate(archive, out_dir)
        p = os.path.join(out_dir, 'r1_02', 'instances', 'inst_0.json')
        j = json.load(open(p))
        j['oracle']['contract_hash'] = '0' * 64
        json.dump(j, open(p, 'w'))
        r = _run_val(archive, _bundle(archive), out_dir)
        ok = r.returncode != 0
        record('val_contract_hash_tamper', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ok = [test_archive_happy_path(), test_archive_moved(), test_archive_wrong_bundle(),
          test_archive_missing_bundle_arg(), test_archive_missing_npz(),
          test_registry_missing_split(), test_registry_missing_hash(),
          test_val_missing_contract(), test_val_missing_statistical_plan(),
          test_val_contract_profile_mismatch(), test_val_statistical_plan_mismatch(),
          test_archive_seal_tamper(), test_archive_missing_sealed(),
          test_val_contract_hash_tamper(), test_formal_gate_role_val_rejected()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
