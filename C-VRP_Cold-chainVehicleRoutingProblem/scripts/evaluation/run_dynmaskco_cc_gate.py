"""M1 在线评估入口：JF1-H-F baseline vs DynMaskCO-CC replanner（同一 strict-online 协议）。

step 1（no-op）：DynMaskCOReplanner(model=None) 与 baseline 精确一致，验收在线骨架。
step 2+：加载冻结 encoder + 训练 adapter/masked decoder/utility head 后，替换 model 传参，
在同一实例上比较 M1 与 baseline 的终局 J/D/Q/E、硬约束与动作分叉。

用法（服务器）：
    # no-op 骨架验收
    python scripts/evaluation/run_dynmaskco_cc_gate.py \
        --data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
        --capacity 50 --num_vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 4 --out results/m0dev/gate_noop
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from dynmaskco_cc_replanner import make_dynmaskco_replanner
from feature_only_replanner import make_feature_only_replanner, load_feature_only_scorer
from counterfactual_teacher import _eval
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import service_ok, hard_vector_from_outcome, outcome_protocol_error
from service_first import paired_bootstrap_ci


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    profile = ObjectiveProfile(
        name=data['name'], distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']), energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']), lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'), dev_statistics=data.get('dev_statistics'))
    if profile.profile_hash != data.get('profile_hash'):
        raise ValueError(f"profile hash mismatch: {profile.profile_hash[:12]}")
    return profile


def _make_env(dataset, capacity, num_vehicles, replanner, objective, profile, contract=None):
    if objective == 'coldchain':
        contract = apply_objective_profile(contract or default_pilot_contract(), profile)
    else:
        contract = None
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=replanner, coldchain_contract=contract)


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def _run_instance(dataset, capacity, num_vehicles, objective, profile, contract, inst_idx, K,
                  scorer, method='m1'):
    t0 = time.time()
    env = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile,
                    contract)
    traces_b, served_b = env.run(inst_idx)
    base = _eval(env, inst_idx, traces_b, objective, served_mask=served_b)
    err = outcome_protocol_error(base, objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: baseline PROTOCOL_ERROR {err}")
    t_base = time.time() - t0

    t1 = time.time()
    if method == 'feature_only':
        replanner = make_feature_only_replanner(scorer=scorer, K=K)
    else:
        replanner = make_dynmaskco_replanner(scorer=scorer, K=K)
    menv = _make_env(dataset, capacity, num_vehicles, replanner, objective, profile, contract)
    traces_m, served_m = menv.run(inst_idx)
    m1 = _eval(menv, inst_idx, traces_m, objective, served_mask=served_m)
    err = outcome_protocol_error(m1, objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: M1 PROTOCOL_ERROR {err}")
    t_m1 = time.time() - t1

    stats = getattr(menv.replanner, 'online_stats', {}) if scorer is not None else {}
    timing = getattr(menv.replanner, 'timing', {}) if scorer is not None else {}
    return {'inst_idx': inst_idx, 'baseline': base, 'm1': m1,
            'served_equal': bool(np.array_equal(served_b, served_m)),
            'runtime': time.time() - t0, 'runtime_base': t_base, 'runtime_m1': t_m1,
            'online_stats': stats, 'timing': timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--m1-ckpt', default=None, help='训练好的 M1 checkpoint（model 加载）')
    ap.add_argument('--feature-only-ckpt', default=None, help='M0 feature-only checkpoint（对照）')
    ap.add_argument('--encoder-ckpt', default=None, help='冻结 encoder（--m1-ckpt 时必填）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())

    scorer = None
    method = 'm1'
    if args.m1_ckpt is not None and args.feature_only_ckpt is not None:
        raise SystemExit('--m1-ckpt 与 --feature-only-ckpt 不能同时提供')
    if args.m1_ckpt is not None:
        if args.encoder_ckpt is None:
            raise SystemExit('--m1-ckpt 需要 --encoder-ckpt（冻结 encoder）')
        sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))
        from train_fleet_head import load_base_model
        from dynmaskco_cc import load_m1_model, M1Scorer
        encoder = load_base_model(args.encoder_ckpt)
        model = load_m1_model(args.m1_ckpt, encoder)
        scorer = M1Scorer(model)          # 创建一次，跨实例共享（编译缓存复用）
        print(f"loaded M1 model + scorer from {args.m1_ckpt}", flush=True)
    elif args.feature_only_ckpt is not None:
        scorer = load_feature_only_scorer(args.feature_only_ckpt)
        method = 'feature_only'
        print(f"loaded feature-only scorer from {args.feature_only_ckpt}", flush=True)

    results = []
    os.makedirs(os.path.join(args.out, 'instances'), exist_ok=True)
    for i in range(n):
        r = _run_instance(dataset, args.capacity, args.num_vehicles, args.objective,
                          profile, contract, i, args.K, scorer, method=method)
        results.append(r)
        with open(os.path.join(args.out, 'instances', f'inst_{i}.json'), 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  [inst {i}] base={_cost(r['baseline'], args.objective):.4f} "
              f"m1={_cost(r['m1'], args.objective):.4f} "
              f"accept={r['online_stats'].get('n_accept', 0)} "
              f"keep={r['online_stats'].get('n_keep', 0)} "
              f"t_m1={r['runtime_m1']:.1f}s", flush=True)

    base = [r['baseline'] for r in results]
    m1 = [r['m1'] for r in results]

    def _full_ok(o):
        return (service_ok(o, args.objective) and int(o.get('terminal_unresolved', 0)) == 0
                and int(o.get('ownership_violations', 0)) == 0)

    base_ok = all(_full_ok(o) for o in base)
    m1_ok = all(_full_ok(o) for o in m1)
    paired = [i for i in range(n) if _full_ok(base[i]) and _full_ok(m1[i])]
    deltas = [_cost(m1[i], args.objective) - _cost(base[i], args.objective) for i in paired]
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    parity = all(r['served_equal'] and
                 abs(_cost(r['m1'], args.objective) - _cost(r['baseline'], args.objective)) <= 1e-9
                 for r in results)

    field = 'distance_cost' if args.objective == 'distance' else 'coldchain_cost'
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', f'baseline_{field}', f'm1_{field}', 'delta',
                    'baseline_complete', 'm1_complete', 'served_equal', 'runtime'])
        for i in range(n):
            w.writerow([i, f"{_cost(base[i], args.objective):.4f}",
                        f"{_cost(m1[i], args.objective):.4f}",
                        f"{_cost(m1[i], args.objective) - _cost(base[i], args.objective):+.4f}",
                        int(base[i]['complete']), int(m1[i]['complete']),
                        int(results[i]['served_equal']), f"{results[i]['runtime']:.2f}"])

    summary = {
        'objective': args.objective, 'K': args.K, 'n': n,
        'baseline_service_ok': bool(base_ok), 'm1_service_ok': bool(m1_ok),
        'paired_n': len(paired), 'parity_exact': bool(parity),
        'baseline_mean': float(np.mean([_cost(base[i], args.objective) for i in paired])) if paired else None,
        'm1_mean': float(np.mean([_cost(m1[i], args.objective) for i in paired])) if paired else None,
        'mean_delta': mean_delta,
        'ci': list(paired_bootstrap_ci(deltas)) if deltas else None,
        'online_stats': {k: int(sum(r['online_stats'].get(k, 0) for r in results))
                         for k in ['n_eligible', 'n_candidates', 'n_score', 'n_keep', 'n_defer',
                                   'n_accept', 'n_certificate_reject', 'n_unexpected_error',
                                   'n_skip_no_struct']},
        'runtime_base_mean': float(np.mean([r['runtime_base'] for r in results])),
        'runtime_m1_mean': float(np.mean([r['runtime_m1'] for r in results])),
        'runtime_total_mean': float(np.mean([r['runtime'] for r in results])),
        'timing': {k: float(sum(r['timing'].get(k, 0.0) for r in results))
                   for k in ['input_prep', 'encode', 'model_score', 'cert']},
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== DynMaskCO-CC gate (objective={args.objective}, K={args.K}, n={n}) ===")
    print(f"  [service] baseline={base_ok} m1={m1_ok} (paired n={len(paired)}/{n})")
    print(f"  [cost] baseline={summary['baseline_mean']:.4f} m1={summary['m1_mean']:.4f} "
          f"Δ={mean_delta:+.4f}")
    print(f"  [parity_exact] {parity}  (no-op 应 True)")
    print(f"  saved: {args.out}/per_instance.csv + summary.json")


if __name__ == '__main__':
    main()
