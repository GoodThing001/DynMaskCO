"""规范 cell 汇总重建（控制/分析层，不在「计算版本」冻结清单内）。

断点续跑时，runner 每次只覆盖本次运行的实例子集汇总；本模块在 cell 全部实例落盘后，
从全部 inst_*.json 原始记录重建 canonical summary / per_instance.csv / manifest.json /
oracle_log.jsonl，保证汇总、coverage 与运行证据代表完整 cell，而非「最后一次补算」的子集。

汇总公式与 run_action_oracle.py 的 summary 段保持一致（复用 service_first.paired_bootstrap_ci
与 hard_gate.service_ok），因此「部分补算后完整重建」与「一次跑完」应产出同一份汇总。
"""
import csv, json, os, datetime, hashlib
import numpy as np

from hard_gate import service_ok
from service_first import paired_bootstrap_ci
from instance_validation import validate_instance_record, _cost

_COMPONENT_FIELDS = ['distance_cost', 'quality_loss', 'energy_kwh', 'num_unsalable',
                     'thermal_violation_count', 'thermal_violation_duration_h']


def _atomic_write_text(path, text):
    tmp = path + f'.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _full_service_ok(o, objective):
    return (service_ok(o, objective)
            and int(o.get('terminal_unresolved', 0)) == 0
            and int(o.get('ownership_violations', 0)) == 0)


def rebuild_cell_summary(cell_dir, *, objective, role, profile_manifest,
                         data_sha256, expected_code_sha256=None, workers=None,
                         launch_mode='canonical', require_local=False,
                         expected_instance_ids=None, compute_sha256=None, run_id=None,
                         expected_contract_hash=None):
    inst_dir = os.path.join(cell_dir, 'instances')
    if not os.path.isdir(inst_dir):
        raise RuntimeError(f'{cell_dir}/instances 不存在')

    files = sorted(f for f in os.listdir(inst_dir)
                   if f.startswith('inst_') and f.endswith('.json'))
    ids = []
    for f in files:
        stem = f[len('inst_'):-len('.json')]
        if not stem.isdigit():
            raise RuntimeError(f'非预期实例文件名：{f}')
        ids.append(int(stem))
    ids = sorted(ids)
    if expected_instance_ids is not None:
        expected = sorted(int(x) for x in expected_instance_ids)
        if ids != expected:
            raise RuntimeError(f'实例 ID 集合不符：实际 {ids[:12]}... 预期 {expected[:12]}...')
    elif ids != list(range(len(ids))):
        raise RuntimeError(f'实例 ID 非连续前缀：{ids[:10]}...')
    n = len(ids)
    if n == 0:
        raise RuntimeError(f'{cell_dir} 无实例')

    results = []
    for i in ids:
        p = os.path.join(inst_dir, f'inst_{i}.json')
        with open(p) as f:
            rec = json.load(f)
        state, err = validate_instance_record(rec, i, objective, require_local=require_local,
                                              expected_code_sha256=expected_code_sha256,
                                              expected_contract_hash=expected_contract_hash)
        if state != 'valid':
            raise RuntimeError(f'{p} 状态 {state}：{err}')
        results.append(rec)

    base = [r['baseline'] for r in results]
    orac = [r['oracle'] for r in results]
    runtimes = [r.get('runtime') for r in results]
    local = [r.get('local') for r in results]

    base_service = all(_full_service_ok(o, objective) for o in base)
    orac_service = all(_full_service_ok(o, objective) for o in orac)
    paired_idx = [i for i in range(n)
                  if _full_service_ok(base[i], objective) and _full_service_ok(orac[i], objective)]
    deltas = [_cost(orac[i], objective) - _cost(base[i], objective) for i in paired_idx]
    ci_lo, ci_hi = paired_bootstrap_ci(deltas)
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    base_mean = float(np.mean([_cost(base[i], objective) for i in paired_idx])) if paired_idx else float('nan')
    orac_mean = float(np.mean([_cost(orac[i], objective) for i in paired_idx])) if paired_idx else float('nan')

    if role == 'regression':
        verdict = 'PROTOCOL_PASS' if (base_service and orac_service) else 'SERVICE_GATE_FAIL'
    elif not (base_service and orac_service):
        verdict = 'SERVICE_GATE_FAIL'
    elif mean_delta < 0 and ci_hi < 0:
        verdict = 'GO'
    elif mean_delta < 0:
        verdict = 'EVIDENCE_INSUFFICIENT'
    else:
        verdict = 'NO_HEADROOM'

    log = [rec for r in results for rec in r['log']]
    log.sort(key=lambda rec: (rec['inst_idx'], rec.get('seq', 0)))

    field = 'distance_cost' if objective == 'distance' else 'coldchain_cost'
    component_fields = _COMPONENT_FIELDS if objective == 'coldchain' else []

    # 重建开始：先撤销旧完成标记（原子替换为 BUILDING），避免旧 marker 与新旧混合产物并存
    _atomic_write_text(os.path.join(cell_dir, 'canonical.COMPLETE'),
                       json.dumps({'status': 'BUILDING'}))

    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ['instance_id', f'baseline_{field}', f'oracle_{field}', 'delta',
              'baseline_complete', 'oracle_complete', 'runtime']
    for cf in component_fields:
        header += [f'baseline_{cf}', f'oracle_{cf}']
    w.writerow(header)
    for i in range(n):
        row = [results[i]['inst_idx'], f"{_cost(base[i], objective):.4f}",
               f"{_cost(orac[i], objective):.4f}",
               f"{_cost(orac[i], objective) - _cost(base[i], objective):+.4f}",
               int(base[i]['complete']), int(orac[i]['complete']),
               f"{runtimes[i]:.2f}"]
        for cf in component_fields:
            row += [f"{base[i][cf]:.4f}", f"{orac[i][cf]:.4f}"]
        w.writerow(row)
    _atomic_write_text(os.path.join(cell_dir, 'per_instance.csv'), buf.getvalue())

    local_inst_deltas = []
    n_local_no_eligible = 0
    local_collected = False
    for loc in local:
        if isinstance(loc, dict):
            local_collected = True
            ed = loc.get('event_deltas') or []
            if ed:
                local_inst_deltas.append(float(np.mean(ed)))
            else:
                n_local_no_eligible += 1

    summary = {
        'objective': objective,
        'role': role,
        'instance_ids': list(range(n)),
        'objective_profile': profile_manifest,
        'n': n,
        'workers': workers,
        'launch_mode': launch_mode,
        'baseline_service_ok': bool(base_service),
        'oracle_service_ok': bool(orac_service),
        'paired_n': len(paired_idx),
        'baseline_mean': base_mean,
        'oracle_mean': orac_mean,
        'mean_delta': mean_delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'verdict': verdict,
        'n_oracle_decision_records': len(log),
        'coverage': {
            'n_candidates': int(sum(r.get('num_candidates', 0) for r in log)),
            'n_feasible': int(sum(r.get('num_feasible', 0) for r in log)),
            'n_rolled_out': int(sum(r.get('num_rolled_out', 0) for r in log)),
            'n_hard_pass': int(sum(r.get('num_hard_pass', 0) for r in log)),
        },
        'n_accepted_actions': int(sum(1 for r in log
                                      if not str(r.get('selected_action', '')).startswith('__'))),
        'n_keep': int(sum(1 for r in log if r.get('selected_action') == '__KEEP__')),
        'n_defer': int(sum(1 for r in log if r.get('selected_action') == '__DEFER__')),
        'first_plan_change_seq': next(
            (r['seq'] for r in log
             if not str(r.get('selected_action', '')).startswith('__')), None),
        'n_failed': 0,
        'per_instance_runtime_mean': float(np.mean(runtimes)) if runtimes else None,
        'per_instance_runtime': [round(t, 2) for t in runtimes],
    }
    if local_collected:
        ld = np.array(local_inst_deltas)
        summary['local_mean_delta'] = float(np.mean(ld)) if len(ld) else float('nan')
        summary['local_ci'] = list(paired_bootstrap_ci(ld))
        summary['local_n_instances'] = int(len(ld))
        summary['local_n_no_eligible'] = int(n_local_no_eligible)
    for cf in component_fields:
        cf_deltas = [orac[i][cf] - base[i][cf] for i in paired_idx]
        cf_lo, cf_hi = paired_bootstrap_ci(cf_deltas)
        summary[f'{cf}_mean_delta'] = float(np.mean(cf_deltas)) if cf_deltas else float('nan')
        summary[f'{cf}_ci'] = [cf_lo, cf_hi]

    _atomic_write_text(os.path.join(cell_dir, 'summary.json'),
                       json.dumps(summary, indent=2))
    # oracle_log.jsonl 总是显式发布（即便空），避免旧 log 残留
    _atomic_write_text(os.path.join(cell_dir, 'oracle_log.jsonl'),
                       ''.join(json.dumps(rec) + '\n' for rec in log))

    manifest = {
        'objective': objective,
        'workers': workers,
        'launch_mode': launch_mode,
        'code_sha256': expected_code_sha256,
        'compute_sha256': compute_sha256,
        'run_id': run_id,
        'data_sha256': data_sha256,
        'profile_hash': (profile_manifest.get('profile_hash') if profile_manifest else None),
        'n_failed': 0,
        'n_instances': n,
        'per_instance_runtime': [round(t, 2) for t in runtimes],
    }
    _atomic_write_text(os.path.join(cell_dir, 'manifest.json'),
                       json.dumps(manifest, indent=2))

    # 完成标记：绑定四个产物 hash + 实例集合 + 身份，聚合器据此核验发布完整性
    artifact_files = ['summary.json', 'manifest.json', 'per_instance.csv', 'oracle_log.jsonl']
    artifact_hashes = {f: _sha256_file(os.path.join(cell_dir, f)) for f in artifact_files}
    _atomic_write_text(os.path.join(cell_dir, 'canonical.COMPLETE'),
                       json.dumps({
                           'status': 'COMPLETE',
                           'n': n,
                           'instance_ids': list(range(n)),
                           'runner_sha256': expected_code_sha256,
                           'compute_sha256': compute_sha256,
                           'run_id': run_id,
                           'data_sha256': data_sha256,
                           'profile_hash': (profile_manifest.get('profile_hash')
                                            if profile_manifest else None),
                           'artifact_hashes': artifact_hashes,
                           'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                       }))

    return summary
