"""品质物理敏感性扫描：扫描衰减率缩放因子，测 D7-a 易腐排序头腔在哪一档转正。

单进程内循环，避免 shell 引号问题。对每个 rate_scale（+ 可选 value_scale）重建合同、
重采 Probe 状态、算距离贪心 vs 易腐重排的 myopic gap，写合并 summary。

用法：
    python scripts/evaluation/run_quality_sweep.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --out results/m0_scale/perish_sweep
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from action_contract import apply_action
from cc_lns_replanner import _plan_suffixes
from mpre_policy import enumerate_legal_actions
from coldchain_contract import default_pilot_contract, apply_objective_profile
from run_headroom_census import Probe, _load_profile, _plan_copy, _eval_plan
from run_perishability_headroom import (scale_quality_physics, _perishability_w,
                                        _plan_from_suffixes)


def run_one(dataset, profile, contract, capacity, num_vehicles, max_instances):
    probe = Probe(contract, capacity, max_per_instance=2)
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.run(inst)
    states = probe.states
    gaps = []
    q_deltas, d_deltas, e_deltas = [], [], []
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']

        P = _plan_copy(partial)
        remaining = list(mask_set)
        while remaining:
            legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
            if not legal:
                P = None
                break
            legal.sort(key=lambda t: (t[2].incremental_distance, t[1].action_id()))
            c, a, _cand = legal[0]
            P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
            remaining.remove(c)
        r_dg = _eval_plan(vis, P, contract, contract.objective) if P is not None else None
        r_perish = None
        if P is not None:
            suf = _plan_suffixes(P)
            reordered = {vid: tuple(sorted(custs, key=lambda c: _perishability_w(env, inst_idx, contract, c)))
                         for vid, custs in suf.items()}
            P_perish = _plan_from_suffixes(partial, reordered)
            r_perish = _eval_plan(vis, P_perish, contract, contract.objective)
        if r_dg is not None and r_perish is not None:
            gaps.append(float(r_dg.J_vis) - float(r_perish.J_vis))
            q_deltas.append(float(r_dg.Q) - float(r_perish.Q))
            d_deltas.append(float(r_dg.D) - float(r_perish.D))
            e_deltas.append(float(r_dg.E) - float(r_perish.E))
    def _ci(vals):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(0)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(2000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}
    return {
        'n_states': len(gaps),
        'gap': _ci(gaps),
        'Q_delta_mean': float(np.mean(q_deltas)) if q_deltas else None,
        'D_delta_mean': float(np.mean(d_deltas)) if d_deltas else None,
        'E_delta_mean': float(np.mean(e_deltas)) if e_deltas else None,
        'n_gap_positive': int(sum(1 for g in gaps if g > 1e-9)),
        'n_gap_negative': int(sum(1 for g in gaps if g < -1e-9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--rate-scales', default='1,2,5,10,20,50')
    ap.add_argument('--value-scales', default='1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)
    rate_scales = [float(x) for x in args.rate_scales.split(',') if x.strip()]
    value_scales = [float(x) for x in args.value_scales.split(',') if x.strip()]

    results = {}
    for rs in rate_scales:
        for vs in value_scales:
            contract = scale_quality_physics(base, rs, vs)
            r = run_one(dataset, profile, contract, args.capacity, args.num_vehicles,
                        args.max_instances)
            key = f'rate_x{rs:g}_value_x{vs:g}'
            results[key] = r
            g = r['gap']
            print(f"  {key}: gap={g['mean']:.4f} ci=[{g['ci_lo']:.4f},{g['ci_hi']:.4f}] "
                  f"Q={r['Q_delta_mean']:.3f} D={r['D_delta_mean']:.3f} E={r['E_delta_mean']:.1f} "
                  f"pos={r['n_gap_positive']}/{r['n_states']}", flush=True)

    with open(os.path.join(args.out, f'sweep_{args.split}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {args.out}/sweep_{args.split}.json")


if __name__ == '__main__':
    main()
