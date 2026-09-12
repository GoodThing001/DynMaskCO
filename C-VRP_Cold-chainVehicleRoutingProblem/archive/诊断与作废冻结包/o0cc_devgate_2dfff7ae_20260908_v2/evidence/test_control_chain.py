"""控制链验收测试（不跑 oracle rollout）。

流程：在临时目录生成 9 份伪 NPZ + DEV_MANIFEST + profile + 9×128 合成合法实例 + pre_run_manifest，
调用真实 `run_dev_gate.py`（resume 路径，不绕过入口），验证：
  - runner 不被调用（所有实例已预置）；
  - 9 份 canonical 全部生成，marker 四产物 hash 正确；
  - 正式聚合返回 0，total_instances=1152、n_seed_groups=128、bootstrap=10000/42。
再验证纯续跑一致 + 一组负例注入在 runner 启动前/聚合前停止。

用法：python scripts/tests/test_control_chain.py
产物：results/o0cc/control_chain_tests.json + control_chain_report.md
"""
import os, sys, json, subprocess, shutil, tempfile, hashlib

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
_EVAL = os.path.join(_CVRPTW, 'scripts', 'evaluation')
for p in (_EVAL, _BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dev_gate as dg

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def _outcome(cost):
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
    }


def _instance(idx, runner_hash, seed):
    b = 3.2 + (seed % 5) * 0.01
    o = b - 0.05  # oracle 略优于 baseline（非退化）
    ed = -0.05
    return {
        'inst_idx': idx, 'baseline': _outcome(b), 'oracle': _outcome(o),
        'local': {'event_deltas': [ed],
                  'records': [{'event_id': 0, 'n_customers': 2, 'n_customer_deltas': 2,
                               'mean_customer_delta': ed, 'keep_cost': b}],
                  'base_cost': b},
        'log': [], 'runtime': 1.0, 'code_sha256': runner_hash, 'error': None,
    }


def _profile(tmp):
    payload = {'name': 'test-profile', 'distance_scale': 18.0, 'quality_scale': 2.9,
               'energy_scale': 1193.0, 'lambda_quality': 1.0, 'lambda_energy': 1.0,
               'scale_source': 'pilot', 'dev_statistics': None}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=True).encode()).hexdigest()
    payload['profile_hash'] = h
    p = os.path.join(tmp, 'objective_profile.json')
    json.dump(payload, open(p, 'w'))
    return p


def _setup(tmp):
    """生成 9 份伪 NPZ + DEV_MANIFEST + 9×128 实例 + pre_run_manifest。返回 (data_dir, out_dir)。"""
    data_dir = os.path.join(tmp, 'data'); os.makedirs(data_dir)
    out_dir = os.path.join(tmp, 'out'); os.makedirs(out_dir)

    cells = []
    seeds = list(range(1000, 1128))  # 128 唯一内部 seed，跨 cell 一致
    for t in ('R1', 'C1', 'RC1'):
        for e in (0.2, 0.5, 0.8):
            fname = f'dcc_50_{t.lower()}_edod{int(e*10):02d}_dev_gate.npz'
            npz = os.path.join(data_dir, fname)
            with open(npz, 'wb') as f:
                f.write(os.urandom(64))
            cells.append({'type': t, 'edod': e, 'file': fname,
                          'sha256': _sha256_file(npz), 'num_instances': 128,
                          'seed': 7782, 'scene_instance_ids': list(range(128)),
                          'instance_seeds': seeds})

    manifest = {'split_role': 'dev_gate', 'seed': 7782, 'problem_size': 50,
                'num_instances_per_cell': 128, 'capacity': 50, 'cells': cells,
                'generator': 'test', 'generator_sha256': '0' * 64}
    json.dump(manifest, open(os.path.join(data_dir, 'DEV_MANIFEST.json'), 'w'))

    # 计算身份
    compute_sha256, compute_files = dg._version(dg.COMPUTE_FILES)
    control_sha256, _ = dg._version(dg.CONTROL_FILES)
    analysis_sha256, _ = dg._version(dg.ANALYSIS_FILES)
    runner_hash = compute_files['run_action_oracle.py']
    data = {c['file']: c['sha256'] for c in cells}
    prof_path = _profile(tmp)
    prof = json.load(open(prof_path))
    profile_hash = prof['profile_hash']
    frozen = dg._frozen_compute(cells, 128, compute_sha256, data, profile_hash)

    # 生成 9×128 实例
    for c in cells:
        tag = f"{c['type'].lower()}_{int(c['edod']*10):02d}"
        inst_dir = os.path.join(out_dir, tag, 'instances')
        os.makedirs(inst_dir)
        for i in range(128):
            json.dump(_instance(i, runner_hash, c['instance_seeds'][i]),
                      open(os.path.join(inst_dir, f'inst_{i}.json'), 'w'))

    # pre_run_manifest
    pre = {'run': {'run_id': 'test-run-0001', 'timestamp_utc': '2026-09-08T00:00:00Z',
                   'command': 'test', 'data_dir': data_dir, 'profile': 'test',
                   'workers': 1, 'out': out_dir, 'instances_per_cell': 128,
                   'total_instances': 1152},
           **frozen,
           'compute_files': compute_files, 'control_sha256': control_sha256,
           'control_files': {}, 'analysis_sha256': analysis_sha256, 'analysis_files': {},
           'env': {}}
    json.dump(pre, open(os.path.join(out_dir, 'pre_run_manifest.json'), 'w'))
    return data_dir, out_dir, cells, prof_path, runner_hash, compute_sha256, frozen


def _run_driver(data_dir, profile, out_dir):
    return subprocess.run([sys.executable, os.path.join(_EVAL, 'run_dev_gate.py'),
                           '--data-dir', data_dir, '--profile', profile,
                           '--workers', '1', '--out', out_dir],
                          capture_output=True, text=True)


def test_happy_path():
    tmp = tempfile.mkdtemp(prefix='ctrl_chain_')
    try:
        data_dir, out_dir, cells, prof, runner_hash, compute_sha256, frozen = _setup(tmp)
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode == 0
        # 聚合产物
        summ = os.path.join(out_dir, 'dev_gate_summary.json')
        s = json.load(open(summ)) if os.path.exists(summ) else {}
        ok = ok and s.get('total_instances') == 1152 and s.get('n_seed_groups') == 128 \
             and s.get('bootstrap') == {'n_boot': 10000, 'seed': 42}
        # 9 份 canonical marker
        n_marker = 0
        for c in cells:
            tag = f"{c['type'].lower()}_{int(c['edod']*10):02d}"
            mp = os.path.join(out_dir, tag, 'canonical.COMPLETE')
            if os.path.exists(mp):
                m = json.load(open(mp))
                if m.get('status') == 'COMPLETE' and m.get('compute_sha256') == compute_sha256 \
                   and m.get('run_id') == 'test-run-0001':
                    n_marker += 1
        ok = ok and n_marker == 9
        # runner 未被调用（无 attempt START / 无 cell log）
        attempts = os.path.join(out_dir, 'attempts.jsonl')
        no_runner = (not os.path.exists(attempts)) or 'START' not in open(attempts).read()
        ok = ok and no_runner
        record('control_chain_happy_path', ok,
               f"exit={r.returncode} total={s.get('total_instances')} "
               f"seed_groups={s.get('n_seed_groups')} markers={n_marker} runner_called={not no_runner}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume():
    tmp = tempfile.mkdtemp(prefix='ctrl_resume_')
    try:
        data_dir, out_dir, cells, prof, *_ = _setup(tmp)
        r1 = _run_driver(data_dir, prof, out_dir)
        r2 = _run_driver(data_dir, prof, out_dir)
        s1 = json.load(open(os.path.join(out_dir, 'dev_gate_summary.json')))
        ok = r1.returncode == 0 and r2.returncode == 0 and \
             s1['nine_cell_equal_weight_mean_delta'] is not None
        record('control_chain_resume', ok, f"first={r1.returncode} resume={r2.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nonempty_no_manifest():
    tmp = tempfile.mkdtemp(prefix='ctrl_empty_')
    try:
        data_dir, out_dir, _, prof, *_ = _setup(tmp)
        os.remove(os.path.join(out_dir, 'pre_run_manifest.json'))  # 目录非空但无冻结记录
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode != 0
        record('control_chain_nonempty_no_manifest', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_cfg_')
    try:
        data_dir, out_dir, _, prof, *_ = _setup(tmp)
        pre = json.load(open(os.path.join(out_dir, 'pre_run_manifest.json')))
        pre['config']['baseline'] = 'WRONG'
        json.dump(pre, open(os.path.join(out_dir, 'pre_run_manifest.json'), 'w'))
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode != 0
        record('control_chain_config_drift', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_null_instance():
    tmp = tempfile.mkdtemp(prefix='ctrl_null_')
    try:
        data_dir, out_dir, cells, prof, *_ = _setup(tmp)
        tag = f"{cells[0]['type'].lower()}_{int(cells[0]['edod']*10):02d}"
        open(os.path.join(out_dir, tag, 'instances', 'inst_0.json'), 'w').write('null')
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode != 0
        record('control_chain_null_instance', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_aggregator(data_dir, out_dir, prof):
    cells = json.load(open(os.path.join(data_dir, 'DEV_MANIFEST.json')))['cells']
    results = [os.path.join(out_dir, f"{c['type'].lower()}_{int(c['edod']*10):02d}") for c in cells]
    return subprocess.run([sys.executable, os.path.join(_EVAL, 'aggregate_cross_cell.py'),
                           '--results'] + results +
                          ['--manifest', os.path.join(data_dir, 'DEV_MANIFEST.json'),
                           '--mode', 'dev_gate',
                           '--frozen-plan', os.path.join(out_dir, 'pre_run_manifest.json'),
                           '--profile', prof,
                           '--out', os.path.join(out_dir, 'agg_out.json')],
                          capture_output=True, text=True)


def _drift(tmp, mutate, name):
    data_dir, out_dir, cells, prof, *_ = _setup(tmp)
    mutate(data_dir, out_dir, cells)
    r = _run_driver(data_dir, prof, out_dir)
    ok = r.returncode != 0
    record(name, ok, f"exit={r.returncode}")
    return ok


def test_compute_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_cmp_')
    try:
        def mut(d, o, c):
            pre = json.load(open(os.path.join(o, 'pre_run_manifest.json')))
            pre['compute_sha256'] = '0' * 64
            json.dump(pre, open(os.path.join(o, 'pre_run_manifest.json'), 'w'))
        return _drift(tmp, mut, 'control_chain_compute_drift')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_npz_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_npz_')
    try:
        def mut(d, o, c):
            npz = os.path.join(d, c[0]['file'])
            open(npz, 'wb').write(os.urandom(64))  # 内容变化 → hash 不符
        return _drift(tmp, mut, 'control_chain_npz_drift')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_profile_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_prof_')
    try:
        def mut(d, o, c):
            p = os.path.join(tmp, 'objective_profile.json')
            j = json.load(open(p)); j['distance_scale'] = 999.0  # 改内容不改 profile_hash
            json.dump(j, open(p, 'w'))
        data_dir, out_dir, _, prof, *_ = _setup(tmp)
        mut(data_dir, out_dir, None)
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode != 0
        record('control_chain_profile_drift', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cell_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_cell_')
    try:
        def mut(d, o, c):
            pre = json.load(open(os.path.join(o, 'pre_run_manifest.json')))
            pre['cells'][0]['instance_seeds'] = list(range(2000, 2128))
            json.dump(pre, open(os.path.join(o, 'pre_run_manifest.json'), 'w'))
        return _drift(tmp, mut, 'control_chain_cell_drift')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bootstrap_drift():
    tmp = tempfile.mkdtemp(prefix='ctrl_boot_')
    try:
        def mut(d, o, c):
            pre = json.load(open(os.path.join(o, 'pre_run_manifest.json')))
            pre['config']['n_boot'] = 1
            json.dump(pre, open(os.path.join(o, 'pre_run_manifest.json'), 'w'))
        return _drift(tmp, mut, 'control_chain_bootstrap_drift')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_extra_instance():
    tmp = tempfile.mkdtemp(prefix='ctrl_extra_')
    try:
        data_dir, out_dir, cells, prof, *_ = _setup(tmp)
        tag = f"{cells[0]['type'].lower()}_{int(cells[0]['edod']*10):02d}"
        json.dump(_instance(128, 'x' * 64, 128),
                  open(os.path.join(out_dir, tag, 'instances', 'inst_128.json'), 'w'))
        r = _run_driver(data_dir, prof, out_dir)
        ok = r.returncode != 0
        record('control_chain_extra_instance', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_marker_tamper():
    tmp = tempfile.mkdtemp(prefix='ctrl_marker_')
    try:
        data_dir, out_dir, cells, prof, *_ = _setup(tmp)
        r0 = _run_driver(data_dir, prof, out_dir)
        if r0.returncode != 0:
            record('control_chain_marker_tamper', False, f'setup fail {r0.returncode}')
            return False
        tag = f"{cells[0]['type'].lower()}_{int(cells[0]['edod']*10):02d}"
        mp = os.path.join(out_dir, tag, 'canonical.COMPLETE')
        m = json.load(open(mp)); m['artifact_hashes']['summary.json'] = '0' * 64
        json.dump(m, open(mp, 'w'))
        r = _run_aggregator(data_dir, out_dir, prof)
        ok = r.returncode != 0
        record('control_chain_marker_tamper', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_mix():
    tmp = tempfile.mkdtemp(prefix='ctrl_runmix_')
    try:
        data_dir, out_dir, cells, prof, *_ = _setup(tmp)
        r0 = _run_driver(data_dir, prof, out_dir)
        if r0.returncode != 0:
            record('control_chain_run_mix', False, f'setup fail {r0.returncode}')
            return False
        tag = f"{cells[0]['type'].lower()}_{int(cells[0]['edod']*10):02d}"
        mp = os.path.join(out_dir, tag, 'canonical.COMPLETE')
        m = json.load(open(mp)); m['run_id'] = 'other-run'
        json.dump(m, open(mp, 'w'))
        r = _run_aggregator(data_dir, out_dir, prof)
        ok = r.returncode != 0
        record('control_chain_run_mix', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ok = []
    ok.append(test_happy_path())
    ok.append(test_resume())
    ok.append(test_nonempty_no_manifest())
    ok.append(test_config_drift())
    ok.append(test_null_instance())
    ok.append(test_compute_drift())
    ok.append(test_npz_drift())
    ok.append(test_profile_drift())
    ok.append(test_cell_drift())
    ok.append(test_bootstrap_drift())
    ok.append(test_extra_instance())
    ok.append(test_marker_tamper())
    ok.append(test_run_mix())
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'control_chain_tests.json'), 'w') as f:
        json.dump({'all_pass': all(ok), 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
