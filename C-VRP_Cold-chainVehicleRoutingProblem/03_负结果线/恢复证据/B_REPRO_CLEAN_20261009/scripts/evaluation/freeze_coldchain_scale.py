"""DEV-CAL coldchain scale 冻结 v2（阶段 B）。

用 hard-feasible baseline **JF1-H-F（slack_vehicles=1）** 统计 D/Q/E 算术均值，9 个 cell 等权，
冻结具名 profile **o0cc-pilot-devmean-equal-v2**。

与 Stage-A 使用同一 hard vector：任一实例 hard 失败 → 退出，不生成正式 profile（不能悄悄
排除失败实例）。provenance 保存 baseline 名/版本/slack、代码 hash、contract hash、DEV-CAL
manifest + 9 数据 hash、hard Gate 汇总、每 cell D/Q/E 统计与 scale 聚合规则。

发布流程：写临时文件 → reload → 复算 profile hash 一致 → 发布正式 objective_profile.json。
v1 不删除，标 INVALIDATED（reason: generated from non-hard-feasible original JF1-H baseline）。

用法（服务器，DEV-CAL 9×128）：
    python scripts/evaluation/freeze_coldchain_scale.py \
        --data data/baseline/50_node/dev_cal/dcc_50_{r1,c1,rc1}_edod{02,05,08}_devcal.npz \
        --capacity 50 --num_vehicles 25 \
        --name o0cc-pilot-devmean-equal-v2 \
        --out results/o0cc/scale_v2 \
        --v1-profile results/o0cc/devcal_scale/objective_profile.json
"""
import argparse, os, json, sys, hashlib, time, datetime
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from coldchain_contract import (default_pilot_contract, make_devmean_profile,
                               ObjectiveProfile, load_coldchain_contract)
from coldchain_evaluator import evaluate_coldchain_trace


BASELINE_NAME = 'JF1-H-F'
BASELINE_VERSION = 'jf1h-f-v1'
SLACK_VEHICLES = 1
SCALE_SOURCE = 'devmean'


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _hard_vector(out, rp, served_mask):
    """与 Stage-A audit 相同的显式 hard vector。"""
    terminal_unresolved = sum(1 for c in rp.deferred_customers if not served_mask[int(c)])
    return {
        'complete': bool(out.get('complete')),
        'n_unserved_zero': int(out.get('n_unserved', -1)) == 0,
        'n_duplicate_zero': int(out.get('n_duplicate', -1)) == 0,
        'tw_feasible': bool(out.get('tw_feasible')),
        'capacity_feasible': bool(out.get('capacity_feasible')),
        'depot_return_feasible': bool(out.get('depot_return_feasible')),
        'temperature_hard_feasible': bool(out.get('temperature_hard_feasible')),
        'all_orders_picked': bool(out.get('all_orders_picked')),
        'all_cargo_delivered_to_depot': bool(out.get('all_cargo_delivered_to_depot')),
        'terminal_manifests_empty': bool(out.get('terminal_manifests_empty')),
        'trace_accounting_consistent': bool(out.get('trace_accounting_consistent')),
        'distance_accounting_consistent': bool(out.get('distance_accounting_consistent')),
        'terminal_unresolved_zero': int(terminal_unresolved) == 0,
        'ownership_violations_zero': int(rp.repair_stats['ownership_violations']) == 0,
    }


def _load_devcal_manifest(data_paths):
    """定位并加载 DEV_CAL_MANIFEST.json（与数据文件同目录）。"""
    d = os.path.dirname(os.path.abspath(data_paths[0]))
    mpath = os.path.join(d, 'DEV_CAL_MANIFEST.json')
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"找不到 DEV_CAL_MANIFEST.json：{mpath}")
    with open(mpath) as f:
        manifest = json.load(f)
    return mpath, manifest


def _validate_inputs(data_paths, manifest):
    """校验 9 个唯一 cell + 与 manifest 的 file/sha256/128 实例一致。"""
    cells = manifest['cells']
    manifest_by_file = {c['file']: c for c in cells}
    names = [os.path.basename(p) for p in data_paths]
    if len(names) != len(set(names)):
        raise ValueError(f"重复 data 文件：{names}")
    if set(names) != set(manifest_by_file):
        raise ValueError(f"输入 cell 与 manifest 不符：missing={sorted(set(manifest_by_file)-set(names))} "
                         f"extra={sorted(set(names)-set(manifest_by_file))}")
    if len(data_paths) != 9:
        raise ValueError(f"期望 9 个 cell，得到 {len(data_paths)}")
    for p in data_paths:
        name = os.path.basename(p)
        cell = manifest_by_file[name]
        if cell['num_instances'] != 128:
            raise ValueError(f"{name} manifest num_instances={cell['num_instances']}，期望 128")
        actual_hash = _sha256_file(p)
        if actual_hash != cell['sha256']:
            raise ValueError(f"{name} 数据 hash 不符：manifest={cell['sha256'][:12]} "
                             f"actual={actual_hash[:12]}")
        d = np.load(p)
        n = int(d['coords'].shape[0])
        d.close()
        if n != 128:
            raise ValueError(f"{name} 实例数={n}，期望 128")


def _quantiles(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'median': float(np.median(a)),
        'iqr': float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


def _cell_statistics(data_path, capacity, num_vehicles, out_dir, contract=None):
    """用 JF1-H-F 跑一个 cell，收集 D/Q/E + hard Gate；任一实例 hard 失败则抛错。"""
    dataset = dict(np.load(data_path))
    contract = contract or default_pilot_contract()
    rp = JF1HRepairReplanner(slack_vehicles=SLACK_VEHICLES)
    env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                          replanner=rp, coldchain_contract=contract)
    n = dataset['coords'].shape[0]
    cell = os.path.splitext(os.path.basename(data_path))[0]

    dists, quals, energies = [], [], []
    n_hard_fail = 0
    first_fail = None

    inst_dir = os.path.join(out_dir, 'instances')
    os.makedirs(inst_dir, exist_ok=True)

    for i in range(n):
        traces, served_mask = env.run(i)
        out = evaluate_coldchain_trace(traces, {
            'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
            'service_time': env.service_time[i], 'demands': env.demands[i],
            'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
            'initial_quality': env.initial_quality[i], 'speed': env.tw_speed,
        }, contract)
        hv = _hard_vector(out, rp, served_mask)
        inst_pass = all(hv.values())
        if not inst_pass:
            n_hard_fail += 1
            if first_fail is None:
                first_fail = {'instance_index': i, 'hard_vector': hv}
        dists.append(float(out['distance_km']))
        quals.append(float(out['quality_loss']))
        energies.append(float(out['energy_kwh']))
        with open(os.path.join(inst_dir, f'{cell}__{i}.json'), 'w') as f:
            json.dump({
                'scene_instance_id': f'{cell}__{i}', 'cell': cell, 'instance_index': i,
                'hard_pass': inst_pass, 'hard_vector': hv,
                'distance_km': out['distance_km'], 'quality_loss': out['quality_loss'],
                'energy_kwh': out['energy_kwh'],
            }, f, indent=2)

    if n_hard_fail > 0:
        raise RuntimeError(
            f"{cell} hard Gate 失败 {n_hard_fail}/{n}；首例={json.dumps(first_fail)}。"
            f"终止 freeze，不生成 profile。")

    return {
        'dataset': os.path.basename(data_path),
        'dataset_path': data_path,
        'dataset_sha256': _sha256_file(data_path),
        'n_instances': n,
        'hard_pass': '{} / {}'.format(n, n),
        'distance': _quantiles(dists),
        'quality': _quantiles(quals),
        'energy': _quantiles(energies),
    }


def main():
    parser = argparse.ArgumentParser(description='DEV-CAL coldchain scale 冻结 v2（JF1-H-F）')
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--name', default='o0cc-pilot-devmean-equal-v2')
    parser.add_argument('--lambda-quality', type=float, default=1.0)
    parser.add_argument('--lambda-energy', type=float, default=1.0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--v1-profile', default=None,
                        help='v1 objective_profile.json 路径，用于写 INVALIDATED 标记（不删除）')
    parser.add_argument('--coldchain-contract', default=None,
                        help='calibrated contract JSON（缺省回退 pilot）')
    args = parser.parse_args()

    mpath, manifest = _load_devcal_manifest(args.data)
    _validate_inputs(args.data, manifest)
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())

    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    cells = [_cell_statistics(d, args.capacity, args.num_vehicles, args.out, contract)
             for d in args.data]

    # 各场景等权：主 scale = 各场景算术均值的算术平均
    distance_mean = float(np.mean([c['distance']['mean'] for c in cells]))
    quality_mean = float(np.mean([c['quality']['mean'] for c in cells]))
    energy_mean = float(np.mean([c['energy']['mean'] for c in cells]))

    statistics = {
        'baseline': BASELINE_NAME,
        'baseline_version': BASELINE_VERSION,
        'slack_vehicles': SLACK_VEHICLES,
        'split_role': 'devcal',
        'scale_rule': 'cell_equal_weight_arithmetic_mean',
        'num_cells': len(cells),
        'distance_mean': distance_mean,
        'quality_mean': quality_mean,
        'energy_mean': energy_mean,
        'lambda_quality': args.lambda_quality,
        'lambda_energy': args.lambda_energy,
        'normalized_baseline_contribution': {
            'distance': 1.0,
            'quality': args.lambda_quality,
            'energy': args.lambda_energy,
        },
        'cells': cells,
    }
    profile = make_devmean_profile(args.name, statistics,
                                   args.lambda_quality, args.lambda_energy)

    # 验收：scale 有限且 > 0
    for comp, v in (('distance', distance_mean), ('quality', quality_mean),
                    ('energy', energy_mean)):
        if not (np.isfinite(v) and v > 0):
            raise RuntimeError(f"{comp}_scale 非法（须有限且 >0）：{v}")

    # provenance
    _src = os.path.join(_CVRPTW, 'scripts')
    devcal_manifest_hash = _sha256_file(mpath)
    provenance = {
        'baseline': BASELINE_NAME,
        'baseline_version': BASELINE_VERSION,
        'slack_vehicles': SLACK_VEHICLES,
        'capacity': args.capacity,
        'num_vehicles': args.num_vehicles,
        'contract_hash': contract.contract_hash,
        'devcal_manifest_path': mpath,
        'devcal_manifest_sha256': devcal_manifest_hash,
        'code_sha256': {
            'freeze_coldchain_scale': _sha256_file(os.path.join(_src, 'evaluation', 'freeze_coldchain_scale.py')),
            'jf1h_repair': _sha256_file(os.path.join(_src, 'simulation', 'jf1h_repair.py')),
            'action_contract': _sha256_file(os.path.join(_src, 'simulation', 'action_contract.py')),
            'strict_online_env': _sha256_file(os.path.join(_src, 'simulation', 'strict_online_env.py')),
            'joint_fleet': _sha256_file(os.path.join(_src, 'simulation', 'joint_fleet.py')),
            'coldchain_state': _sha256_file(os.path.join(_src, 'coldchain', 'coldchain_state.py')),
            'coldchain_evaluator': _sha256_file(os.path.join(_src, 'evaluation', 'coldchain_evaluator.py')),
            'coldchain_contract': _sha256_file(os.path.join(_src, 'coldchain', 'coldchain_contract.py')),
        },
        'devcal_generator': manifest.get('generator'),
        'devcal_generator_sha256': manifest.get('generator_sha256'),
    }

    manifest_out = profile.to_manifest()
    manifest_out['provenance'] = provenance
    manifest_out['hard_gate'] = {
        'total_instances': sum(c['n_instances'] for c in cells),
        'hard_pass': '{} / {}'.format(sum(c['n_instances'] for c in cells),
                                      sum(c['n_instances'] for c in cells)),
    }

    # 发布流程：临时文件 → reload → 复算 profile hash → 一致后发布正式文件
    tmp_path = os.path.join(args.out, 'objective_profile.tmp.json')
    with open(tmp_path, 'w') as f:
        json.dump(manifest_out, f, indent=2)
    with open(tmp_path) as f:
        reloaded = json.load(f)
    recomputed = ObjectiveProfile(
        name=reloaded['name'],
        distance_scale=reloaded['distance_scale'],
        quality_scale=reloaded['quality_scale'],
        energy_scale=reloaded['energy_scale'],
        lambda_quality=reloaded['lambda_quality'],
        lambda_energy=reloaded['lambda_energy'],
        scale_source=reloaded['scale_source'],
        dev_statistics=reloaded['dev_statistics'],
    )
    if recomputed.profile_hash != reloaded['profile_hash']:
        raise RuntimeError("profile reload hash 不一致：recomputed="
                           f"{recomputed.profile_hash} file={reloaded['profile_hash']}")
    final_path = os.path.join(args.out, 'objective_profile.json')
    os.replace(tmp_path, final_path)

    # 附加产物：scale_statistics.json / service_gate.json / manifest.json
    with open(os.path.join(args.out, 'scale_statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    with open(os.path.join(args.out, 'service_gate.json'), 'w') as f:
        json.dump({'total_instances': sum(c['n_instances'] for c in cells),
                   'hard_pass': '{} / {}'.format(sum(c['n_instances'] for c in cells),
                                                 sum(c['n_instances'] for c in cells)),
                   'all_hard_pass': True,
                   'cells': [{c['dataset']: c['hard_pass']} for c in cells]}, f, indent=2)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump({
            'run': {'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                    'wall_seconds': round(time.time() - started, 2),
                    'baseline': BASELINE_NAME, 'baseline_version': BASELINE_VERSION,
                    'slack_vehicles': SLACK_VEHICLES},
            'contract_hash': contract.contract_hash,
            'devcal_manifest_sha256': devcal_manifest_hash,
            'profile_name': profile.name,
            'profile_hash': profile.profile_hash,
            'provenance': provenance,
        }, f, indent=2)

    # v1 INVALIDATED 标记（不删除 v1）
    if args.v1_profile:
        v1_dir = os.path.dirname(os.path.abspath(args.v1_profile))
        v1_name = os.path.basename(args.v1_profile)
        marker = os.path.join(v1_dir, 'objective_profile.INVALIDATED.json')
        with open(marker, 'w') as f:
            json.dump({
                'profile': v1_name,
                'status': 'INVALIDATED',
                'reason': 'generated from non-hard-feasible original JF1-H baseline',
                'superseded_by': profile.name,
                'invalidated_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',
            }, f, indent=2)

    print(f"=== DEV-CAL coldchain scale 冻结 v2 ===")
    print(f"  profile: {profile.name}")
    print(f"  profile_hash: {profile.profile_hash}")
    print(f"  baseline: {BASELINE_NAME} v{BASELINE_VERSION} slack={SLACK_VEHICLES}")
    print(f"  主 scale（9 cell 等权算术均值）:")
    print(f"    distance_scale = {distance_mean:.6f}")
    print(f"    quality_scale  = {quality_mean:.6f}")
    print(f"    energy_scale   = {energy_mean:.6f}")
    print(f"  λ: quality={args.lambda_quality}  energy={args.lambda_energy}")
    print(f"  归一化后 baseline 贡献 = (1.0, {args.lambda_quality}, {args.lambda_energy})")
    print(f"  hard Gate: 1152/1152 (每 cell 128)")
    print(f"  reload hash exact: True")
    for c in cells:
        print(f"    {c['dataset']}: D={c['distance']['mean']:.3f} "
              f"Q={c['quality']['mean']:.4f} E={c['energy']['mean']:.2f} "
              f"(n={c['n_instances']})")
    print(f"  saved: {args.out}/objective_profile.json (+scale_statistics/service_gate/manifest)")


if __name__ == '__main__':
    main()
