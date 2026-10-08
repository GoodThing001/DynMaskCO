"""D7-b · 易腐排序的终局存活（terminal survival）+ 可预测性（D7-c）特征导出。

对给定 rate_scale 的重标定合同，测：易腐重排（vs 距离贪心）的 myopic 改善是否进入终局。

每状态三路终局 rollout：baseline（P0 无 force）、P_dg（距离贪心 force）、P_perish（易腐重排 force）。
terminal_gap = terminal_J(P_dg) − terminal_J(P_perish)（>0 = 易腐重排终局改善）；
terminal_headroom = terminal_J(P0) − terminal_J(P_perish)（>0 = 相对 baseline 改善）。

用法：
    python scripts/evaluation/run_terminal_perishability.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --rate-scale 5 --out results/m0_scale/perish_terminal_rs5
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
from action_contract import apply_action
from cc_lns_replanner import _plan_suffixes
from mpre_policy import enumerate_legal_actions
from coldchain_contract import default_pilot_contract, apply_objective_profile
from recourse_snapshot import capture_recourse_snapshot, resume_from_snapshot
from counterfactual_teacher import _snapshot_with_force, _eval
from run_headroom_census import Probe, _load_profile, _plan_copy, _eval_plan
from run_perishability_headroom import scale_quality_physics, _perishability_w, _plan_from_suffixes


def _terminal(renv, snapshot, plan, mutable_ids):
    snap2 = _snapshot_with_force(renv, snapshot, plan, mutable_ids=mutable_ids)
    traces, served = renv.run_resumed(snap2)
    return _eval(renv, int(snapshot['instance_id']), traces, 'coldchain', served_mask=served)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--rate-scale', type=float, default=5.0)
    ap.add_argument('--max-snapshots-per-inst', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = scale_quality_physics(apply_objective_profile(default_pilot_contract(), profile),
                                     args.rate_scale, 1.0)
    objective = contract.objective

    snapshots = {}
    per_inst_cap = {}

    def snap_hook(e, i, clk, eid, rid, veh, tr, sm, ac):
        cnt = per_inst_cap.get(int(i), 0)
        if cnt >= args.max_snapshots_per_inst:
            return
        key = (int(i), int(eid))
        if key not in snapshots:
            snapshots[key] = capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac)
            per_inst_cap[int(i)] = cnt + 1

    probe = Probe(contract, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.snapshot_hook = snap_hook
        env.run(inst)
    states = probe.states
    print(f'collected {len(states)} states, {len(snapshots)} snapshots', flush=True)

    rows = []
    for st in states:
        key = (st['inst'], st['event'])
        snap = snapshots.get(key)
        if snap is None:
            continue
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
        P_dg = P
        r_dg = _eval_plan(vis, P_dg, contract, objective) if P_dg is not None else None
        P_perish = None
        r_perish = None
        if P_dg is not None:
            suf = _plan_suffixes(P_dg)
            reordered = {vid: tuple(sorted(custs, key=lambda c: _perishability_w(env, inst_idx, contract, c)))
                         for vid, custs in suf.items()}
            P_perish = _plan_from_suffixes(partial, reordered)
            r_perish = _eval_plan(vis, P_perish, contract, objective)

        renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity, 1.0,
                                    args.num_vehicles, coldchain_contract=contract)
        traces0, served0 = renv.run_resumed(snap)
        o0 = _eval(renv, inst_idx, traces0, 'coldchain', served_mask=served0)
        o_dg = _terminal(renv, snap, P_dg, mutable_ids) if P_dg is not None else None
        o_perish = _terminal(renv, snap, P_perish, mutable_ids) if P_perish is not None else None

        rows.append({
            'inst': inst_idx, 'event': st['event'],
            'myopic_gap': (float(r_dg.J_vis) - float(r_perish.J_vis)) if (r_dg is not None and r_perish is not None) else None,
            'terminal_J0': float(o0['coldchain_cost']),
            'terminal_J_dg': float(o_dg['coldchain_cost']) if o_dg is not None else None,
            'terminal_J_perish': float(o_perish['coldchain_cost']) if o_perish is not None else None,
            'terminal_gap': (float(o_dg['coldchain_cost']) - float(o_perish['coldchain_cost'])) if (o_dg is not None and o_perish is not None) else None,
            'terminal_headroom_perish': (float(o0['coldchain_cost']) - float(o_perish['coldchain_cost'])) if o_perish is not None else None,
            'complete_perish': bool(o_perish['complete']) if o_perish is not None else None,
        })
        print(f"  [inst {inst_idx}/evt {st['event']}] myo_gap={rows[-1]['myopic_gap'] if rows[-1]['myopic_gap'] is None else round(rows[-1]['myopic_gap'],4)} "
              f"term_gap={rows[-1]['terminal_gap'] if rows[-1]['terminal_gap'] is None else round(rows[-1]['terminal_gap'],4)} "
              f"term_hr={rows[-1]['terminal_headroom_perish'] if rows[-1]['terminal_headroom_perish'] is None else round(rows[-1]['terminal_headroom_perish'],4)}",
              flush=True)

    def _inst_agg(key):
        per = {}
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            per.setdefault(r['inst'], []).append(v)
        return [float(np.mean(v)) for v in per.values()]

    def _ci(vals):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(0)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(2000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}

    summary = {
        'split': args.split, 'rate_scale': args.rate_scale, 'n_states': len(rows),
        'myopic_gap': _ci(_inst_agg('myopic_gap')),
        'terminal_gap': _ci(_inst_agg('terminal_gap')),
        'terminal_headroom_perish': _ci(_inst_agg('terminal_headroom_perish')),
        'n_terminal_gap_positive': int(sum(1 for r in rows if (r['terminal_gap'] or 0) > 1e-9)),
        'n_terminal_headroom_positive': int(sum(1 for r in rows if (r['terminal_headroom_perish'] or 0) > 1e-9)),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n=== D7-b terminal survival (perishability reorder) ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
