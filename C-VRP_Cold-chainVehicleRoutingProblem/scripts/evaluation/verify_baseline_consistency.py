"""阶段 C 基线一致性证明：当前代码 baseline 与 v2 freeze 逐实例 D/Q/E + 完整 hard vector 对比。

数据/配置来源：
  - manifest.json：contract_hash、provenance（baseline/slack/capacity/num_vehicles）；
  - scale_statistics.json：cells（dataset/dataset_sha256/n_instances）。

校验顺序：**先做廉价的前置校验（数据 hash / 配置 / 实例数），通过后才跑 rollout**，
避免在长时间计算后才报 KeyError 之类。任一 mismatch / 缺失 / 数量不符 / 非有限值 → 非零退出码，
保存 baseline_consistency.json。结论措辞 =「scale 相关指标与可行性一致，v2 继续有效」。

用法（服务器）：
    python scripts/evaluation/verify_baseline_consistency.py \
        --data <9 个 dev_cal npz...> --freeze-dir results/o0cc/scale_v2 \
        --capacity 50 --num_vehicles 25
"""
import argparse, os, sys, json, math, hashlib
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
from jf1h_repair import make_continuation
from coldchain_contract import default_pilot_contract
from coldchain_evaluator import evaluate_coldchain_trace


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _hard_vector(out, rp, served_mask):
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


def _fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--freeze-dir', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    args = ap.parse_args()

    # ---------- 廉价前置校验（rollout 之前） ----------
    names = [os.path.basename(p) for p in args.data]
    if len(names) != len(set(names)):
        _fail(f"duplicate data files: {names}")
    if len(names) != 9:
        _fail(f"expected 9 cells, got {len(names)}")

    manifest_path = os.path.join(args.freeze_dir, 'manifest.json')
    stats_path = os.path.join(args.freeze_dir, 'scale_statistics.json')
    if not os.path.exists(manifest_path):
        _fail(f"freeze manifest not found: {manifest_path}")
    if not os.path.exists(stats_path):
        _fail(f"scale_statistics not found: {stats_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    contract = default_pilot_contract()
    if manifest.get('contract_hash') != contract.contract_hash:
        _fail("contract hash mismatch vs freeze manifest")

    # 配置一致
    prov = manifest.get('provenance', {})
    if prov.get('baseline') != 'JF1-H-F':
        _fail(f"baseline 不符：{prov.get('baseline')}")
    if prov.get('slack_vehicles') != 1:
        _fail(f"slack_vehicles 不符：{prov.get('slack_vehicles')}")
    if float(prov.get('capacity', 0)) != args.capacity:
        _fail(f"capacity 不符：{prov.get('capacity')} vs {args.capacity}")
    if int(prov.get('num_vehicles', 0)) != args.num_vehicles:
        _fail(f"num_vehicles 不符：{prov.get('num_vehicles')} vs {args.num_vehicles}")

    # 按 dataset 身份映射（不按输入顺序）
    cells = stats.get('cells')
    if not cells or len(cells) != 9:
        _fail(f"scale_statistics cells 缺失或数量不符：{len(cells) if cells else 0}")
    cell_by_file = {c['dataset']: c for c in cells}
    for p in args.data:
        name = os.path.basename(p)
        c = cell_by_file.get(name)
        if c is None:
            _fail(f"{name} 不在 scale_statistics cells 中")
        if c.get('n_instances') != 128:
            _fail(f"{name} n_instances={c.get('n_instances')} != 128")
        actual = _sha256_file(p)
        if actual != c.get('dataset_sha256'):
            _fail(f"{name} 数据 hash 不符：manifest={c.get('dataset_sha256','?')[:12]} "
                  f"actual={actual[:12]}")
    total_expected = sum(c['n_instances'] for c in cells)
    if total_expected != 1152:
        _fail(f"总实例数不符：{total_expected} != 1152")

    # ---------- rollout 逐实例对比 ----------
    mismatches = []
    per_component = {k: {'max_abs': 0.0, 'max_rel': 0.0} for k in
                     ('distance_km', 'quality_loss', 'energy_kwh')}
    checked = 0
    total_hard_fail = 0

    for d in args.data:
        cell = os.path.splitext(os.path.basename(d))[0]
        dataset = dict(np.load(d))
        rp = make_continuation()
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=rp, coldchain_contract=contract)
        n = dataset['coords'].shape[0]
        for i in range(n):
            traces, served = env.run(i)
            out = evaluate_coldchain_trace(traces, {
                'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
                'service_time': env.service_time[i], 'demands': env.demands[i],
                'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
                'initial_quality': env.initial_quality[i], 'speed': env.tw_speed}, contract)
            fpath = os.path.join(args.freeze_dir, 'instances', f'{cell}__{i}.json')
            if not os.path.exists(fpath):
                mismatches.append([cell, i, 'freeze instance missing']); continue
            with open(fpath) as f:
                fr = json.load(f)
            checked += 1

            for k in ('distance_km', 'quality_loss', 'energy_kwh'):
                a, b = float(out[k]), float(fr[k])
                if not (math.isfinite(a) and math.isfinite(b)):
                    mismatches.append([cell, i, k, 'non-finite', a, b]); continue
                abs_err = abs(a - b)
                rel_err = abs_err / max(1.0, abs(b))
                per_component[k]['max_abs'] = max(per_component[k]['max_abs'], abs_err)
                per_component[k]['max_rel'] = max(per_component[k]['max_rel'], rel_err)
                if rel_err > 1e-6:
                    mismatches.append([cell, i, k, a, b])

            hv = _hard_vector(out, rp, served)
            fhv = fr.get('hard_vector')
            if fhv is None:
                mismatches.append([cell, i, 'hard_vector missing in freeze']); continue
            if hv != fhv:
                diff = {k: (hv.get(k), fhv.get(k)) for k in hv if hv.get(k) != fhv.get(k)}
                mismatches.append([cell, i, 'hard_vector', diff])
            if not all(hv.values()):
                total_hard_fail += 1

    ok = (len(mismatches) == 0 and total_hard_fail == 0 and checked == 1152)
    if checked != 1152:
        print(f"ERROR: 覆盖实例数 {checked} != 1152（不完整覆盖）")
    print(f"=== 基线一致性（{checked} 实例 vs v2 freeze）===")
    print(f"  合同 hash: {'一致' if manifest.get('contract_hash') == contract.contract_hash else '不一致'}")
    print(f"  配置: baseline=JF1-H-F slack=1 capacity={args.capacity} num_vehicles={args.num_vehicles}")
    for k, v in per_component.items():
        print(f"  {k}: max_abs={v['max_abs']:.3e} max_rel={v['max_rel']:.3e}")
    print(f"  当前 hard Gate fail: {total_hard_fail}")
    print(f"  mismatch 数: {len(mismatches)}")
    print(f"  结论: {'scale 相关指标与可行性一致，v2 继续有效' if ok else '存在 mismatch，需复核'}")

    with open(os.path.join(args.freeze_dir, 'baseline_consistency.json'), 'w') as f:
        json.dump({
            'all_match': ok, 'checked_instances': checked,
            'mismatches': mismatches[:100],
            'per_component_max_error': per_component,
            'current_hard_gate_fail': total_hard_fail,
            'contract_hash_match': manifest.get('contract_hash') == contract.contract_hash,
            'conclusion': ('scale 相关指标与可行性一致，v2 继续有效' if ok else '存在 mismatch，需复核'),
        }, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
