"""OR6 确定性 parity 比较器：先独立验收两批，再对比行为，证明确定性。

P1-2 批次准入（在比较前执行）：
  1. 对两批分别运行 verify_aggregate（结构性 + pre-run 绑定复验）；
  2. 对每个实例按各自 pre-run + DEV_MANIFEST 外源身份重跑完整
     validate_instance_record（common 合同 + trace replay + OR 特有）；
  3. 要求 protocol_id_a == protocol_id_b 且 run_id_a != run_id_b。

比较口径（每 cell × 每实例）：
  - decision_hash（行为确定性，必须相等）；
  - artifact_hash（复现身份，排除计时后必须相等）；
  - outcome D/Q/E/J + complete/n_unserved/n_duplicate；
  - hard_vector / audit / stats；
  - actions 完整动作序列（JSON 规范化逐字节对账）。

parity_summary.json 绑定：comparator_source_sha256、两批 pre-run / top canonical /
实例文件 sha256、每实例对账结果。

用法：
    python parity_compare.py --run-a <dirA> --run-b <dirB> \
        --out parity_summary.json
"""
import argparse
import hashlib
import json
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from aggregate_devproto import verify_aggregate, CELLS, DEV_MANIFEST
from ortools_instance_validation import validate_instance_record, CHECKPOINT_PREFIX

OUTCOME_FIELDS = ('distance_cost', 'distance_km', 'quality_loss', 'energy_kwh',
                  'coldchain_cost', 'complete', 'n_unserved', 'n_duplicate')
HARD_VECTOR_KEYS = ('complete', 'n_unserved_zero', 'n_duplicate_zero',
                    'tw_feasible', 'capacity_feasible',
                    'depot_return_feasible', 'temperature_hard_feasible',
                    'all_orders_picked', 'all_cargo_delivered_to_depot',
                    'terminal_manifests_empty', 'trace_accounting_consistent',
                    'distance_accounting_consistent')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _canon_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _compare_records(ra, rb):
    problems = []
    if ra['decision_hash'] != rb['decision_hash']:
        problems.append('decision_hash 不一致')
    if ra['artifact_hash'] != rb['artifact_hash']:
        problems.append('artifact_hash 不一致')
    for k in OUTCOME_FIELDS:
        if k in ra['outcome'] or k in rb['outcome']:
            if ra['outcome'].get(k) != rb['outcome'].get(k):
                problems.append(f'outcome.{k} 不一致')
    hv_a, hv_b = ra.get('hard_vector'), rb.get('hard_vector')
    for k in HARD_VECTOR_KEYS:
        if (hv_a or {}).get(k) != (hv_b or {}).get(k):
            problems.append(f'hard_vector.{k} 不一致')
    au_a, au_b = ra.get('audit'), rb.get('audit')
    for k in ('ownership_violations', 'terminal_unresolved'):
        if (au_a or {}).get(k) != (au_b or {}).get(k):
            problems.append(f'audit.{k} 不一致')
    st_a, st_b = ra.get('stats'), rb.get('stats')
    for k in ('n_solver_calls', 'n_fast_path', 'fallback_triggered_events'):
        if (st_a or {}).get(k) != (st_b or {}).get(k):
            problems.append(f'stats.{k} 不一致')
    if _canon_json(ra.get('actions')) != _canon_json(rb.get('actions')):
        problems.append('actions 动作序列不一致')
    return len(problems) == 0, problems


def admission_check(protocol_a, protocol_b, run_id_a, run_id_b):
    """P1-2：parity 批次准入——protocol_id 必须相同，run_id 必须不同。"""
    problems = []
    if not protocol_a or not protocol_b:
        problems.append('缺 protocol_id（旧格式批次）')
    elif protocol_a != protocol_b:
        problems.append(f'protocol_id 不一致: {protocol_a} vs {protocol_b}')
    if run_id_a == run_id_b:
        problems.append(f'run_id 相同（非独立批次）: {run_id_a}')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-a', required=True)
    ap.add_argument('--run-b', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    comparator_sha = sha256_file(os.path.abspath(__file__))

    # ---- P1-2 批次准入：先独立验收两批 ----
    verify_aggregate(args.run_a)
    verify_aggregate(args.run_b)

    pre_a = json.load(open(os.path.join(args.run_a, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    pre_b = json.load(open(os.path.join(args.run_b, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    protocol_a = pre_a.get('protocol_id')
    protocol_b = pre_b.get('protocol_id')
    run_id_a = pre_a.get('run_id')
    run_id_b = pre_b.get('run_id')
    admission = admission_check(protocol_a, protocol_b, run_id_a, run_id_b)
    if admission:
        raise RuntimeError('; '.join(admission))

    dev_manifest = json.load(open(DEV_MANIFEST, encoding='utf-8'))
    cells_by_name = {
        f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}": c
        for c in dev_manifest['cells']}
    instances = tuple(range(int(pre_a['config']['instances_per_cell'])))
    env_a = pre_a.get('ortools_environment') or {}
    env_b = pre_b.get('ortools_environment') or {}
    native_a = env_a.get('native_extension_sha256')
    native_b = env_b.get('native_extension_sha256')

    per_instance = []
    all_match = True
    for cell in CELLS:
        c = cells_by_name[cell]
        for inst in instances:
            pa = os.path.join(args.run_a, cell, 'instances', f'inst_{inst}.json')
            pb = os.path.join(args.run_b, cell, 'instances', f'inst_{inst}.json')
            if not os.path.exists(pa) or not os.path.exists(pb):
                raise RuntimeError(f'实例文件缺失: {cell} inst_{inst}')
            ra = json.load(open(pa, encoding='utf-8'))
            rb = json.load(open(pb, encoding='utf-8'))

            def _expected(pre, native):
                return {
                    'instance_id': inst,
                    'scene_instance_id': str(c['scene_instance_ids'][inst]),
                    'instance_seed': int(c['instance_seeds'][inst]),
                    'data_sha256': c['sha256'],
                    'profile_hash': pre['objective_profile_hash'],
                    'code_hash': pre['compute_hash'],
                    'checkpoint_hash': (CHECKPOINT_PREFIX + native) if native else None,
                }
            problems_a = validate_instance_record(ra, _expected(pre_a, native_a))
            problems_b = validate_instance_record(rb, _expected(pre_b, native_b))
            admission_problems = (['run_a: ' + p for p in problems_a]
                                  + ['run_b: ' + p for p in problems_b])

            match, problems = _compare_records(ra, rb)
            all_match &= match and not admission_problems
            per_instance.append({
                'cell': cell, 'instance_id': inst,
                'scene_instance_id': ra.get('scene_instance_id'),
                'admission_ok': not admission_problems,
                'admission_problems': admission_problems,
                'match': match, 'problems': problems,
                'decision_hash': ra['decision_hash'],
                'artifact_hash': ra['artifact_hash'],
                'file_sha256_a': sha256_file(pa),
                'file_sha256_b': sha256_file(pb),
                'outcome': {k: ra['outcome'].get(k) for k in OUTCOME_FIELDS},
                'n_actions': len(ra.get('actions', [])),
            })

    summary = {
        'schema_version': 'cc-compare-parity-summary-v1',
        'method': 'OR-Tools-RH-D',
        'comparator_source_sha256': comparator_sha,
        'cells': CELLS,
        'instances_per_cell': len(instances),
        'n_instances': len(per_instance),
        'all_match': all_match,
        'verdict': 'PARITY_ALL_MATCH' if all_match else 'PARITY_MISMATCH',
        'identity': {
            'run_a': os.path.basename(os.path.normpath(args.run_a)),
            'run_b': os.path.basename(os.path.normpath(args.run_b)),
            'protocol_id_a': protocol_a,
            'protocol_id_b': protocol_b,
            'run_id_a': run_id_a,
            'run_id_b': run_id_b,
            'compute_hash_a': pre_a.get('compute_hash'),
            'compute_hash_b': pre_b.get('compute_hash'),
            'native_extension_sha256_a': native_a,
            'native_extension_sha256_b': native_b,
            'pre_run_manifest_sha256_a': sha256_file(
                os.path.join(args.run_a, 'pre_run_manifest.json')),
            'pre_run_manifest_sha256_b': sha256_file(
                os.path.join(args.run_b, 'pre_run_manifest.json')),
            'top_canonical_sha256_a': sha256_file(
                os.path.join(args.run_a, 'canonical.COMPLETE')),
            'top_canonical_sha256_b': sha256_file(
                os.path.join(args.run_b, 'canonical.COMPLETE')),
        },
        'per_instance': per_instance,
    }
    tmp = args.out + f'.tmp_{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out)
    print(f'parity_summary written: {args.out}')
    print(f'  comparator_source_sha256 = {comparator_sha}')
    print(f'  verdict = {summary["verdict"]} '
          f'({sum(1 for p in per_instance if p["match"])}/{len(per_instance)} match)')


if __name__ == '__main__':
    main()
