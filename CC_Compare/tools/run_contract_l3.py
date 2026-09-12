"""run_contract_l3.py — L3 合同迁移的单变量注入 runner。

每个处理组在独立 Python 进程里用 importlib 加载指定合同文件，校验源码 SHA-256
与 runtime contract hash 后，只替换 strict_online_runner.make_contract（用被选
模块自己的 default_pilot_contract + apply_objective_profile）；数据、profile、
公共 runner、adapter、求解器配置与环境保持不变。

方法配置在 runner 内部按 method 锁定，CLI 不得随意改变：
  - PyVRP  : max_iterations=300, seed=0, capacity=50, vehicles=25
  - OR-Tools: solution_limit=30, time_limit_s=30（安全上限）, capacity=50, vehicles=25

每个 cell 只取 DEV_MANIFEST 登记的 instance 0（instance_seeds[0] +
scene_instance_ids[0]），instance seed/scene/NPZ SHA 一律来自 manifest，不来自输出自证。

输出目录必须为空（拒绝接管/续写）。写完 pre_run_manifest → 9 实例 JSON →
cell summary → aggregate summary → canonical.COMPLETE。

用法：
    python run_contract_l3.py --method pyvrp --contract-source <abs> \
        --expected-contract-source-sha256 <hex> --expected-runtime-contract-hash <hex> \
        --dev-manifest <path> --objective-profile <path> \
        --instances-per-cell 1 --treatment old --out <new dir>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid

_DCC_TOOLS = os.path.dirname(os.path.abspath(__file__))
_CC_COMPARE = os.path.dirname(_DCC_TOOLS)
_COMMON = os.path.join(_CC_COMPARE, 'common')
for p in (_DCC_TOOLS, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import run_instance, dataset_hash
from record_validation import validate_instance_record

# 冻结配置（计划 Task 2 Step 2）：CLI 值不得冲突
METHOD_CONFIG = {
    'pyvrp': {
        'max_iterations': 300, 'seed': 0, 'capacity': 50, 'num_vehicles': 25,
        'adapter_module': os.path.join(_CC_COMPARE, 'PyVRP', 'dcc_vrp',
                                       'pyvrp_adapter.py'),
    },
    'ortools': {
        'solution_limit': 30, 'time_limit_s': 30.0, 'capacity': 50,
        'num_vehicles': 25,
        'adapter_module': os.path.join(_CC_COMPARE, 'OR-Tools', 'dcc_vrp',
                                       'ortools_adapter.py'),
    },
}


def sha256_file(path):
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


def load_contract_module(path, unique_name):
    """用唯一模块名加载指定合同文件，并预写 sys.modules（避免 dataclass 解析问题）。"""
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def inject_contract(sr, contract_mod):
    """只替换 strict_online_runner.make_contract，不改任何文件。"""
    def make_contract_injected(profile):
        if profile is None:
            return None
        return contract_mod.apply_objective_profile(
            contract_mod.default_pilot_contract(), profile)
    sr.make_contract = make_contract_injected


def build_profile(profile_path, contract_mod):
    """用被选合同模块自己的 ObjectiveProfile 从 JSON 构造 profile（并校验 hash）。"""
    data = json.load(open(profile_path, encoding='utf-8'))
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError('拒绝加载 INVALIDATED v1 profile')
    profile = contract_mod.ObjectiveProfile(
        name=data['name'],
        distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']),
        energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']),
        lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'),
        dev_statistics=data.get('dev_statistics'),
    )
    if profile.profile_hash != data.get('profile_hash'):
        raise ValueError(f'profile reload hash 不一致: recomputed={profile.profile_hash} '
                         f'file={data.get("profile_hash")}')
    return profile


def _make_adapter_factory(method):
    if method == 'pyvrp':
        _adapter_dir = os.path.join(_CC_COMPARE, 'PyVRP', 'dcc_vrp')
        if _adapter_dir not in sys.path:
            sys.path.insert(0, _adapter_dir)
        from pyvrp_adapter import PyVRPRHDAdapter
        cfg = METHOD_CONFIG['pyvrp']
        return (lambda: PyVRPRHDAdapter(max_iterations=cfg['max_iterations'],
                                        seed=cfg['seed'])), cfg
    if method == 'ortools':
        _adapter_dir = os.path.join(_CC_COMPARE, 'OR-Tools', 'dcc_vrp')
        if _adapter_dir not in sys.path:
            sys.path.insert(0, _adapter_dir)
        from ortools_adapter import ORToolsRHDAdapter
        cfg = METHOD_CONFIG['ortools']
        return (lambda: ORToolsRHDAdapter(solution_limit=cfg['solution_limit'],
                                          time_limit_s=cfg['time_limit_s'])), cfg
    raise ValueError(f'未知 method: {method}')


def _git_head():
    import subprocess
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return '<err: %s>' % exc
    try:
        dirty = subprocess.run(['git', 'status', '--porcelain'],
                               capture_output=True, text=True,
                               timeout=30).stdout
    except Exception:  # noqa: BLE001
        dirty = ''
    return head, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=('pyvrp', 'ortools'))
    ap.add_argument('--contract-source', required=True)
    ap.add_argument('--expected-contract-source-sha256', required=True)
    ap.add_argument('--expected-runtime-contract-hash', required=True)
    ap.add_argument('--dev-manifest', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--instances-per-cell', type=int, default=1)
    ap.add_argument('--treatment', required=True, choices=('old', 'new'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true',
                    help='只跑第一个 cell 的 instance 0')
    args = ap.parse_args()

    # ---- 输出目录必须为空 ----
    if os.path.exists(args.out):
        if os.listdir(args.out):
            raise RuntimeError(f'输出目录非空，拒绝接管: {args.out}')
    else:
        os.makedirs(args.out)

    # ---- 合同模块加载 + 注入 ----
    unique_name = f'_l3_contract_{args.treatment}_{uuid.uuid4().hex[:8]}'
    contract_mod = load_contract_module(args.contract_source, unique_name)
    src_sha = sha256_file(args.contract_source)
    if src_sha != args.expected_contract_source_sha256:
        raise ValueError(f'合同源码 SHA 不匹配: actual={src_sha} '
                         f'expected={args.expected_contract_source_sha256}')
    pilot = contract_mod.default_pilot_contract()
    runtime_hash = pilot.contract_hash
    if runtime_hash != args.expected_runtime_contract_hash:
        raise ValueError(f'runtime contract hash 不匹配: actual={runtime_hash} '
                         f'expected={args.expected_runtime_contract_hash}')
    # canonical manifest hash（默认 pilot 的 to_manifest 内容）
    manifest_obj = pilot.to_manifest()
    canonical_manifest_hash = hashlib.sha256(
        json.dumps(manifest_obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()

    import strict_online_runner as sr
    inject_contract(sr, contract_mod)

    # ---- profile ----
    profile = build_profile(args.objective_profile, contract_mod)
    profile_sha = sha256_file(args.objective_profile)

    # ---- DEV_MANIFEST + NPZ 校验 ----
    dev_manifest = json.load(open(args.dev_manifest, encoding='utf-8'))
    dev_manifest_sha = sha256_file(args.dev_manifest)
    data_dir = os.path.dirname(args.dev_manifest)
    for c in dev_manifest['cells']:
        npz = os.path.join(data_dir, c['file'])
        if sha256_file(npz) != c['sha256']:
            raise RuntimeError(f'NPZ hash 与 manifest 不一致: {c["file"]}')

    # ---- adapter factory + 冻结配置校验 ----
    adapter_factory, cfg = _make_adapter_factory(args.method)
    # 冻结配置锁定（CLI 无对应参数，这里显式断言内部值）
    assert cfg['capacity'] == 50 and cfg['num_vehicles'] == 25

    # ---- pre_run_manifest ----
    head, dirty = _git_head()
    instances = tuple(range(args.instances_per_cell))
    pre_run = {
        'run_id': uuid.uuid4().hex,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': args.method,
        'method_config': cfg,
        'treatment': args.treatment,
        'contract_source': os.path.abspath(args.contract_source),
        'contract_source_sha256': src_sha,
        'runtime_contract_hash': runtime_hash,
        'canonical_manifest_hash': canonical_manifest_hash,
        'dev_manifest_sha256': dev_manifest_sha,
        'objective_profile_sha256': profile_sha,
        'objective_profile_hash': profile.profile_hash,
        'git_head': head,
        'git_dirty': bool(dirty.strip()),
        'git_dirty_hash': hashlib.sha256(dirty.encode('utf-8')).hexdigest()
                          if dirty.strip() else None,
        'expected_instances': [
            {'cell': f"{c['type']}_edod{c['edod']}", 'type': c['type'],
             'edod': float(c['edod']), 'file': c['file'],
             'npz_sha256': c['sha256'],
             'instance_seeds': [int(x) for x in c['instance_seeds'][:len(instances)]],
             'scene_instance_ids': [str(x) for x in c['scene_instance_ids'][:len(instances)]]}
            for c in dev_manifest['cells']],
    }
    _atomic_write_json(os.path.join(args.out, 'pre_run_manifest.json'), pre_run)
    print(f"[pre-run] method={args.method} treatment={args.treatment} "
          f"contract={src_sha[:16]} runtime={runtime_hash[:12]}", flush=True)

    cells = dev_manifest['cells']
    if args.dry_run:
        cells = cells[:1]
        print('[dry-run] 只跑第一个 cell', flush=True)

    all_artifact_hashes = []
    t0 = time.time()
    for c in cells:
        cell_dir = os.path.join(args.out, f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}")
        os.makedirs(os.path.join(cell_dir, 'instances'), exist_ok=True)
        ds = dict(np.load(os.path.join(data_dir, c['file'])))
        content_hash = dataset_hash(ds)
        cell_records = []
        for inst in instances:
            rec = run_instance(
                ds, 50, 25, adapter_factory=adapter_factory,
                inst_idx=inst, objective='coldchain', profile=profile,
                seed=cfg.get('seed', 0), data_sha256=c['sha256'],
                adapter_module_path=cfg['adapter_module'],
                instance_seed=c['instance_seeds'][inst],
                scene_instance_id=c['scene_instance_ids'][inst])
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
                  f'fallback={rec["stats"]["fallback_triggered_events"]} '
                  f'dist={rec["outcome"]["distance_cost"]:.3f}', flush=True)
        cell_summary = {
            'cell': f"{c['type']}_edod{c['edod']}",
            'npz_sha256': c['sha256'],
            'dataset_content_sha256': content_hash,
            'n_instances': len(cell_records),
            'n_complete': sum(1 for r in cell_records if r['outcome']['complete']),
            'instance_hashes': [
                {'instance_id': r['instance_id'],
                 'scene_instance_id': r['scene_instance_id'],
                 'instance_seed': r['instance_seed'],
                 'decision_hash': r['decision_hash'],
                 'artifact_hash': r['artifact_hash']} for r in cell_records],
        }
        _atomic_write_json(os.path.join(cell_dir, 'summary.json'), cell_summary)
        all_artifact_hashes += [h['artifact_hash']
                                for h in cell_summary['instance_hashes']]
        print(f'  [{c["type"]}_edod{c["edod"]}] cell 发布完成', flush=True)

    # ---- aggregate summary（从落盘重算）----
    aggregate = {'n': 0, 'n_complete': 0, 'ownership_violations': 0,
                 'terminal_unresolved': 0, 'protocol_errors': 0,
                 'fallback_triggered_events': 0, 'per_instance': []}
    for c in cells:
        cell_dir = os.path.join(args.out, f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}")
        for inst in instances:
            with open(os.path.join(cell_dir, 'instances', f'inst_{inst}.json'),
                      encoding='utf-8') as f:
                rec = json.load(f)
            aggregate['n'] += 1
            aggregate['n_complete'] += int(rec['outcome']['complete'])
            aggregate['ownership_violations'] += int(rec['audit']['ownership_violations'])
            aggregate['terminal_unresolved'] += int(rec['audit']['terminal_unresolved'])
            aggregate['protocol_errors'] += int(rec['protocol']['error'] is not None)
            aggregate['fallback_triggered_events'] += int(rec['stats']['fallback_triggered_events'])
            aggregate['per_instance'].append({
                'cell': f"{c['type']}_edod{c['edod']}", 'instance_id': inst,
                'complete': rec['outcome']['complete'],
                'distance_cost': rec['outcome']['distance_cost'],
                'decision_hash': rec['decision_hash'],
                'artifact_hash': rec['artifact_hash']})
    aggregate['hard_vector_all_pass'] = all(
        r['complete'] for r in aggregate['per_instance'])
    aggregate['runtime_total_s'] = round(time.time() - t0, 1)
    _atomic_write_json(os.path.join(args.out, 'aggregate_summary.json'), aggregate)

    canonical = {
        'run_id': pre_run['run_id'],
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': args.method,
        'treatment': args.treatment,
        'n_instances': aggregate['n'],
        'gate': {
            'n_complete': aggregate['n_complete'],
            'ownership_violations': aggregate['ownership_violations'],
            'terminal_unresolved': aggregate['terminal_unresolved'],
            'protocol_errors': aggregate['protocol_errors'],
            'fallback_triggered_events': aggregate['fallback_triggered_events'],
        },
        'pre_run_manifest_sha256': sha256_file(os.path.join(args.out, 'pre_run_manifest.json')),
        'aggregate_summary_sha256': sha256_file(os.path.join(args.out, 'aggregate_summary.json')),
        'instance_artifact_hashes': sorted(all_artifact_hashes),
    }
    _atomic_write_json(os.path.join(args.out, 'canonical.COMPLETE'), canonical)
    print('\n=== AGGREGATE ===', flush=True)
    for k in ('n', 'n_complete', 'ownership_violations', 'terminal_unresolved',
              'protocol_errors', 'fallback_triggered_events', 'runtime_total_s'):
        print(f'  {k}: {aggregate[k]}', flush=True)


if __name__ == '__main__':
    main()
