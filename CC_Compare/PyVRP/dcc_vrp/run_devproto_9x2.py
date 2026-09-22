"""DEV-PROTO 9×2 证据级回归入口（B2 跨场景行为/证据 Gate）。

证据链要求：
  - 从 DEV_MANIFEST.json 读取九 cell 精确身份（npz sha256 / instance_seeds /
    scene_instance_ids），运行前验证 9 个 NPZ 字节 hash；
  - 真实 instance seed + scene_instance_id 进入记录 identity；
  - 数据身份三重：npz_sha256（文件字节，必须与 manifest 一致）/
    dataset_content_sha256（解压数组组合） / dev_manifest_sha256；
  - 每 cell 独立目录 + 完整实例 record（原子写）；
  - 输出目录必须为空（拒绝接管非空目录）；
  - 汇总从落盘完整记录重新校验（不信运行时内存结果）；
  - 完成后原子发布 canonical.COMPLETE（绑定全部 artifact hash）。

用法：
    python run_devproto_9x2.py --iterations 300 --out results/devproto_9x2_iter300
"""
import argparse
import hashlib
import json
import os
import sys
import time

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import (run_instance, load_objective_profile,
                                   dataset_hash)
from record_validation import validate_instance_record
from pyvrp_adapter import PyVRPRHDAdapter
import identity as pyvrp_identity

DATA_DIR = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
            r'\data\baseline\50_node\dev_proto')
PROFILE = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
           r'\results\o0cc\scale_v2\objective_profile.json')
ADAPTER_MODULE = os.path.join(_DCC_VRP, 'pyvrp_adapter.py')
INSTANCES = ()  # 由 CLI --instances-per-cell 决定
SEED = 0


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path, obj):
    tmp = path + f'.tmp_{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=300)
    ap.add_argument('--instances-per-cell', type=int, default=2,
                    help='每 cell 实例数（2=9x2 行为 Gate；32=9x32 协议稳定性）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # ---- 输出目录必须为空（拒绝接管）----
    if os.path.exists(args.out):
        if os.listdir(args.out):
            raise RuntimeError(f'输出目录非空，拒绝接管: {args.out}')
    else:
        os.makedirs(args.out)

    # ---- 身份装载与验证 ----
    manifest_path = os.path.join(DATA_DIR, 'DEV_MANIFEST.json')
    dev_manifest = json.load(open(manifest_path, encoding='utf-8'))
    dev_manifest_sha256 = _sha256_file(manifest_path)
    instances = tuple(range(args.instances_per_cell))
    for c in dev_manifest['cells']:
        p = os.path.join(DATA_DIR, c['file'])
        h = _sha256_file(p)
        if h != c['sha256']:
            raise RuntimeError(f'NPZ hash 与 manifest 不一致: {c["file"]}')
    profile = load_objective_profile(PROFILE)
    env_identity = pyvrp_identity.environment_identity(include_freeze=True)

    pre_run = {
        'run_id': os.path.basename(os.path.normpath(args.out)),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {'max_iterations': args.iterations, 'seed': SEED,
                   'instances_per_cell': list(range(args.instances_per_cell)),
                   'objective': 'coldchain',
                   'solver_objective': 'distance（PyVRP 内部）',
                   'evaluation': '统一 C0 evaluator D/Q/E/J_CC'},
        'data': {'dev_manifest_sha256': dev_manifest_sha256,
                 'split_role': dev_manifest['split_role'],
                 'split_seed': dev_manifest['seed'],
                 'cells': [{'file': c['file'], 'npz_sha256': c['sha256'],
                            'type': c['type'], 'edod': c['edod'],
                            'instance_seeds': c['instance_seeds'],
                            'scene_instance_ids': c['scene_instance_ids']}
                           for c in dev_manifest['cells']]},
        'objective_profile': profile.to_manifest(),
        'pyvrp_environment': env_identity,
    }
    _atomic_write_json(os.path.join(args.out, 'pre_run_manifest.json'), pre_run)
    print(f"[pre-run] {len(dev_manifest['cells'])} cells 身份验证通过，"
          f"iterations={args.iterations}", flush=True)

    all_artifact_hashes = []
    t0 = time.time()
    for c in dev_manifest['cells']:
        cell_dir = os.path.join(args.out, f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}")
        os.makedirs(os.path.join(cell_dir, 'instances'))
        ds = dict(np.load(os.path.join(DATA_DIR, c['file'])))
        content_hash = dataset_hash(ds)
        cell_records = []
        for inst in instances:
            rec = run_instance(
                ds, 50, 25,
                adapter_factory=lambda: PyVRPRHDAdapter(
                    max_iterations=args.iterations, seed=SEED),
                inst_idx=inst, objective='coldchain', profile=profile,
                seed=SEED, data_sha256=c['sha256'],
                adapter_module_path=ADAPTER_MODULE,
                instance_seed=c['instance_seeds'][inst],
                scene_instance_id=c['scene_instance_ids'][inst])
            # 运行时校验（完整记录）
            err, _ = validate_instance_record(rec, 'coldchain')
            if err:
                raise RuntimeError(f'{c["file"]} inst {inst} 记录校验失败: {err}')
            _atomic_write_json(os.path.join(cell_dir, 'instances',
                                            f'inst_{inst}.json'), rec)
            cell_records.append(rec)
            print(f'  [{c["type"]}_edod{c["edod"]} inst {inst}] '
                  f'complete={rec["outcome"]["complete"]} '
                  f'viol={rec["audit"]["ownership_violations"]} '
                  f'unresolved={rec["audit"]["terminal_unresolved"]} '
                  f'solve={rec["stats"]["n_solver_calls"]} '
                  f'fast={rec["stats"]["n_fast_path"]} '
                  f'waits={rec["stats"]["n_waits"]} '
                  f'customer_waits={rec["stats"]["n_customer_anchor_waits"]} '
                  f'loaded_waits={rec["stats"]["n_loaded_waits"]} '
                  f'warn={rec["stats"]["n_penalty_warnings"]} '
                  f'dist={rec["outcome"]["distance_cost"]:.3f}', flush=True)
        cell_summary = {
            'cell': f'{c["type"]}_edod{c["edod"]}',
            'npz_sha256': c['sha256'],
            'dataset_content_sha256': content_hash,
            'n_instances': len(cell_records),
            'n_complete': sum(1 for r in cell_records
                              if r['outcome']['complete']),
            'instance_hashes': [{'instance_id': r['instance_id'],
                                 'scene_instance_id': r['scene_instance_id'],
                                 'instance_seed': r['instance_seed'],
                                 'decision_hash': r['decision_hash'],
                                 'artifact_hash': r['artifact_hash']}
                                for r in cell_records],
        }
        _atomic_write_json(os.path.join(cell_dir, 'summary.json'), cell_summary)
        _atomic_write_json(os.path.join(cell_dir, 'manifest.json'), {
            'cell': cell_summary['cell'],
            'max_iterations': args.iterations,
            'code_hash': cell_records[0]['code_hash'],
            'checkpoint_hash': cell_records[0]['checkpoint_hash'],
            'profile_hash': cell_records[0]['profile_hash'],
            'data_sha256': c['sha256'],
        })
        all_artifact_hashes += [h['artifact_hash']
                                for h in cell_summary['instance_hashes']]
        print(f'  [{c["type"]}_edod{c["edod"]}] cell 目录发布完成', flush=True)

    # ---- 从落盘完整记录重新校验（不信内存结果）----
    print('\n[re-validate] 从落盘记录重新校验 18 实例...', flush=True)
    aggregate = {'n': 0, 'n_complete': 0, 'ownership_violations': 0,
                 'terminal_unresolved': 0, 'protocol_errors': 0,
                 'penalty_warnings': 0, 'n_solver_calls': 0, 'n_fast_path': 0,
                 'n_waits': 0, 'n_customer_anchor_waits': 0,
                 'n_loaded_waits': 0, 'n_unplanned_customer_events': 0,
                 'fallback_triggered_events': 0, 'per_instance': []}
    for c in dev_manifest['cells']:
        cell_dir = os.path.join(args.out, f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}")
        for inst in instances:
            with open(os.path.join(cell_dir, 'instances',
                                   f'inst_{inst}.json'), encoding='utf-8') as f:
                rec = json.load(f)
            err, _ = validate_instance_record(rec, 'coldchain')
            if err:
                raise RuntimeError(f'落盘记录校验失败 {c["file"]} inst {inst}: {err}')
            aggregate['n'] += 1
            aggregate['n_complete'] += int(rec['outcome']['complete'])
            aggregate['ownership_violations'] += int(rec['audit']['ownership_violations'])
            aggregate['terminal_unresolved'] += int(rec['audit']['terminal_unresolved'])
            aggregate['protocol_errors'] += int(rec['protocol']['error'] is not None)
            aggregate['penalty_warnings'] += int(rec['stats']['n_penalty_warnings'])
            aggregate['n_solver_calls'] += int(rec['stats']['n_solver_calls'])
            aggregate['n_fast_path'] += int(rec['stats']['n_fast_path'])
            aggregate['n_waits'] += int(rec['stats']['n_waits'])
            aggregate['n_customer_anchor_waits'] += int(rec['stats']['n_customer_anchor_waits'])
            aggregate['n_loaded_waits'] += int(rec['stats']['n_loaded_waits'])
            aggregate['n_unplanned_customer_events'] += int(rec['stats']['n_unplanned_customer_events'])
            aggregate['fallback_triggered_events'] += int(rec['stats']['fallback_triggered_events'])
            aggregate['per_instance'].append({
                'cell': f"{c['type']}_edod{c['edod']}", 'instance_id': inst,
                'complete': rec['outcome']['complete'],
                'distance_cost': rec['outcome']['distance_cost'],
                'coldchain_cost': rec['outcome']['coldchain_cost'],
                'decision_hash': rec['decision_hash'],
                'artifact_hash': rec['artifact_hash'],
                'fallback_triggered_events':
                    rec['stats']['fallback_triggered_events'],
            })
    n = aggregate['n']
    aggregate['native_complete_rate'] = aggregate['n_complete'] / n
    aggregate['fallback_instance_rate'] = (sum(
        1 for r in aggregate['per_instance']
        if r['fallback_triggered_events'] > 0) / n if n else None)
    # fallback_event_rate：fallback 事件数 / 总事件数——从落盘记录重算
    n_events = 0
    for c in dev_manifest['cells']:
        cell_dir = os.path.join(args.out, f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}")
        for inst in instances:
            with open(os.path.join(cell_dir, 'instances',
                                   f'inst_{inst}.json'), encoding='utf-8') as f:
                rec = json.load(f)
            n_events += int(rec['stats']['n_events'])
    aggregate['fallback_event_rate'] = (aggregate['fallback_triggered_events']
                                        / n_events if n_events else None)
    aggregate['hard_vector_all_pass'] = all(
        r['complete'] for r in aggregate['per_instance'])
    aggregate['runtime_total_s'] = round(time.time() - t0, 1)
    aggregate['iterations'] = args.iterations
    _atomic_write_json(os.path.join(args.out, 'aggregate_summary.json'), aggregate)

    # ---- canonical.COMPLETE（绑定全部 artifact hash）----
    canonical = {
        'run_id': os.path.basename(os.path.normpath(args.out)),
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_instances': n,
        'iterations': args.iterations,
        'pre_run_manifest_sha256': _sha256_file(
            os.path.join(args.out, 'pre_run_manifest.json')),
        'aggregate_summary_sha256': _sha256_file(
            os.path.join(args.out, 'aggregate_summary.json')),
        'instance_artifact_hashes': sorted(all_artifact_hashes),
        'dev_manifest_sha256': dev_manifest_sha256,
        'gate': {
            'n_complete': aggregate['n_complete'],
            'ownership_violations': aggregate['ownership_violations'],
            'terminal_unresolved': aggregate['terminal_unresolved'],
            'protocol_errors': aggregate['protocol_errors'],
            'penalty_warnings': aggregate['penalty_warnings'],
        },
    }
    _atomic_write_json(os.path.join(args.out, 'canonical.COMPLETE'), canonical)

    print('\n=== AGGREGATE (from disk) ===', flush=True)
    for k in ('n', 'n_complete', 'ownership_violations', 'terminal_unresolved',
              'protocol_errors', 'penalty_warnings', 'n_solver_calls',
              'n_fast_path', 'n_waits', 'n_customer_anchor_waits',
              'n_loaded_waits', 'n_unplanned_customer_events',
              'native_complete_rate', 'runtime_total_s'):
        print(f'  {k}: {aggregate[k]}', flush=True)


if __name__ == '__main__':
    main()
