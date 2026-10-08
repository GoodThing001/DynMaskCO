"""D7-b' · 不可逆 commit 决策的终局存活（决定性）。

全 suffix 重排（run_terminal_perishability.py）终局 ~0（被未来重规划洗掉，符合预期）。
这里测**不可逆的 commit 决策**：把每辆车 suffix 的"最不易腐"客户移到首位（= committed_next），
其余保持原序。commit 是 strict-online 下唯一不可逆的 leg，是最贴近真实在线策略的冷链决策。

用法：
    python scripts/evaluation/run_terminal_commit.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --rate-scale 5 --out results/m0_scale/perish_commit_rs5_train
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


def _commit_least_perishable(env, inst_idx, contract, P):
    """把每辆车 suffix 的最不易腐客户移到首位（其余保持原序）。"""
    out = {}
    for vid, p in P.items():
        suf = [int(x) for x in p.suffix if x > 0]
        if len(suf) <= 1:
            out[vid] = tuple(suf)
            continue
        c_star = min(suf, key=lambda c: _perishability_w(env, inst_idx, contract, c))
        out[vid] = tuple([c_star] + [c for c in suf if c != c_star])
    return _plan_from_suffixes(P, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--rate-scale', type=float, default=5.0)
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
        if cnt >= 2:
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
        snap = snapshots.get((st['inst'], st['event']))
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
        P_commit = _commit_least_perishable(env, inst_idx, contract, P_dg) if P_dg is not None else None
        r_commit = _eval_plan(vis, P_commit, contract, objective) if P_commit is not None else None

        renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity, 1.0,
                                    args.num_vehicles, coldchain_contract=contract)
        traces0, served0 = renv.run_resumed(snap)
        o0 = _eval(renv, inst_idx, traces0, 'coldchain', served_mask=served0)
        o_dg = _terminal(renv, snap, P_dg, mutable_ids) if P_dg is not None else None
        o_commit = _terminal(renv, snap, P_commit, mutable_ids) if P_commit is not None else None

        rows.append({
            'inst': inst_idx, 'event': st['event'],
            'myopic_gap_commit': (float(r_dg.J_vis) - float(r_commit.J_vis)) if (r_dg is not None and r_commit is not None) else None,
            'terminal_J0': float(o0['coldchain_cost']),
            'terminal_J_dg': float(o_dg['coldchain_cost']) if o_dg is not None else None,
            'terminal_J_commit': float(o_commit['coldchain_cost']) if o_commit is not None else None,
            'terminal_gap_commit': (float(o_dg['coldchain_cost']) - float(o_commit['coldchain_cost'])) if (o_dg is not None and o_commit is not None) else None,
            'terminal_headroom_commit': (float(o0['coldchain_cost']) - float(o_commit['coldchain_cost'])) if o_commit is not None else None,
            'complete_commit': bool(o_commit['complete']) if o_commit is not None else None,
        })
        print(f"  [inst {inst_idx}/evt {st['event']}] myo_gap={rows[-1]['myopic_gap_commit'] if rows[-1]['myopic_gap_commit'] is None else round(rows[-1]['myopic_gap_commit'],4)} "
              f"term_gap={rows[-1]['terminal_gap_commit'] if rows[-1]['terminal_gap_commit'] is None else round(rows[-1]['terminal_gap_commit'],4)}",
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
        'myopic_gap_commit': _ci(_inst_agg('myopic_gap_commit')),
        'terminal_gap_commit': _ci(_inst_agg('terminal_gap_commit')),
        'terminal_headroom_commit': _ci(_inst_agg('terminal_headroom_commit')),
        'n_terminal_gap_positive': int(sum(1 for r in rows if (r['terminal_gap_commit'] or 0) > 1e-9)),
        'n_terminal_headroom_positive': int(sum(1 for r in rows if (r['terminal_headroom_commit'] or 0) > 1e-9)),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n=== D7-b' terminal survival (irreversible commit-least-perishable) ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
