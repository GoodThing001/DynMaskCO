"""D7-c · 全程 commit-least-perishable 在线策略的终局头腔（决定性）。

单决策点 commit 会被未来重规划洗掉（D7-b' 已证 ~0）。这里测**全程策略**：在每个决策点，
把 JF1-H incumbent 每辆车的 committed_next 改成"最不易腐"客户（其余保持），写回后环境推进到
下一个事件，效果跨事件累积。终局 cost vs baseline JF1-H。

便宜（无 clairvoyant rollout，仅 commit 重排 + 一次终局评价），~1 分钟/实例。

用法：
    python scripts/evaluation/run_perishability_oracle.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --rate-scale 5 --max-instances 16 --out results/m0_scale/perish_oracle_rs5_cal
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
from jf1h_repair import make_continuation
from action_contract import build_vehicle_plans, certify_route
from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from recourse_snapshot import capture_recourse_snapshot, restore_recourse_snapshot
from counterfactual_teacher import _eval, _snapshot_with_force, _decision_context
from sequential_oracle import mutable_vehicle_ids
from run_perishability_headroom import scale_quality_physics, _perishability_w


def _incumbent_plans(env, snapshot, continuation):
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    env.prepare_decision_point(clock, vehicles)
    if hasattr(continuation, 'restore_state'):
        continuation.restore_state(snapshot.get('replanner_state'))
    replan_ids = {v.vehicle_id for v in vehicles if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=replan_ids)
    return build_vehicle_plans(env, inst_idx, vehicles)


def _commit_least_perishable(env, inst_idx, contract, plans):
    """可行性感知：把最不易腐客户移到首位，仅当重排后 route 仍 TW/容量/返仓可行。"""
    out = {}
    for vid, p in plans.items():
        suf = [int(x) for x in p.suffix if x > 0]
        if len(suf) <= 1:
            out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, tuple(suf))
            continue
        c_star = min(suf, key=lambda c: _perishability_w(env, inst_idx, contract, c))
        if c_star == suf[0]:
            new_suf = suf
        else:
            cand = [c_star] + [c for c in suf if c != c_star]
            cert = certify_route(env, inst_idx, p.anchor_node, p.anchor_time, p.anchor_load,
                                 tuple(cand))
            new_suf = cand if cert['feasible'] else suf
        out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, tuple(new_suf))
    return out


def make_hook(renv, contract):
    def hook(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers):
        snap = capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx,
                                         vehicles, traces, served_mask, all_customers)
        plans = _incumbent_plans(renv, snap, renv.replanner)
        plans = _commit_least_perishable(renv, inst_idx, contract, plans)
        snap2 = _snapshot_with_force(renv, snap, plans, mutable_ids=mutable_vehicle_ids(snap))
        env.force_suffix = {tuple(k): list(v) for k, v in snap2['force_suffix']}
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--rate-scale', type=float, default=5.0)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    with open(args.objective_profile) as f:
        d = json.load(f)
    profile = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                               quality_scale=float(d['quality_scale']),
                               energy_scale=float(d['energy_scale']),
                               lambda_quality=float(d['lambda_quality']),
                               lambda_energy=float(d['lambda_energy']),
                               scale_source=d.get('scale_source', 'pilot'),
                               dev_statistics=d.get('dev_statistics'))
    contract = scale_quality_physics(apply_objective_profile(default_pilot_contract(), profile),
                                     args.rate_scale, 1.0)

    results = []
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        base = _eval(env, inst, traces, 'coldchain', served_mask=served)

        renv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=contract)
        oenv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=contract)
        oenv.oracle_hook = make_hook(renv, contract)
        traces, served = oenv.run(inst)
        orac = _eval(oenv, inst, traces, 'coldchain', served_mask=served)

        results.append({'inst': inst, 'baseline': float(base['coldchain_cost']),
                        'perish_oracle': float(orac['coldchain_cost']),
                        'delta': float(orac['coldchain_cost']) - float(base['coldchain_cost']),
                        'complete': bool(orac['complete'])})
        print(f"  [inst {inst}] baseline={results[-1]['baseline']:.4f} "
              f"perish={results[-1]['perish_oracle']:.4f} delta={results[-1]['delta']:+.4f} "
              f"complete={results[-1]['complete']}", flush=True)

    deltas = [r['delta'] for r in results]
    from service_first import paired_bootstrap_ci
    lo, hi = paired_bootstrap_ci(deltas) if deltas else (float('nan'), float('nan'))
    summary = {
        'rate_scale': args.rate_scale, 'n_instances': len(results),
        'baseline_mean': float(np.mean([r['baseline'] for r in results])),
        'perish_oracle_mean': float(np.mean([r['perish_oracle'] for r in results])),
        'mean_delta': float(np.mean(deltas)), 'ci_lo': lo, 'ci_hi': hi,
        'n_complete': sum(1 for r in results if r['complete']),
        'per_instance': results,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== D7-c full commit-least-perishable online policy (rate x{args.rate_scale:g}) ===")
    print(f"  baseline={summary['baseline_mean']:.4f} perish_oracle={summary['perish_oracle_mean']:.4f}")
    print(f"  delta={summary['mean_delta']:+.4f} CI=[{lo:+.4f},{hi:+.4f}] complete={summary['n_complete']}/{len(results)}")
    print(f"saved: {args.out}/summary.json")


if __name__ == '__main__':
    main()
