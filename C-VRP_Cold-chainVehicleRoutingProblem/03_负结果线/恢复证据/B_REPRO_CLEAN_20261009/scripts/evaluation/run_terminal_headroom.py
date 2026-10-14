"""D3 终局前瞻重评：mask=2 修复在「完整终局 rollout」下是否比 myopic J_vis 暴露更多头腔。

关键问题（决定 G/GV/L 是否值得立项）：局部插入层的头腔是否被 myopic 评价（只看到当前可见
订单 + 距离贪心补全）藏起来了。若终局前瞻下 mask=2 修复出现品质/时间主导的额外头腔，则残差
可学但训练目标必须换成终局代价/未来价值；若终局 ≈ myopic ≈ 0，则局部残差被判死。

做法（无训练、无模型、纯 NumPy + JF1-H continuation）：
  1. 同一 env 同时装 snapshot_hook（capture_recourse_snapshot）与 Probe（采集 mask=2 决策点
     状态，与 run_headroom_census 完全一致），按 (inst, event_id) 对齐；
  2. 对每状态计算 dist-greedy 修复 P_dg 与穷举修复 P_exh（myopic 枚举）；
  3. 对 P0（baseline）/ P_dg / P_exh 分别 _snapshot_with_force + run_resumed 到终局，
     用 _eval(objective='coldchain') 得终局 coldchain_cost；
  4. 报告 myopic gap（J_dg−J_exh）vs 终局 gap（terminal_J_dg−terminal_J_exh）与
     终局头腔（terminal_J0−terminal_J_exh），实例聚合 + 聚类 bootstrap。

用法（服务器）：
    python scripts/evaluation/run_terminal_headroom.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --out results/m0_scale/terminal_headroom_train
"""
import argparse
import copy
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
from action_contract import build_vehicle_plans
from coldchain_contract import default_pilot_contract, apply_objective_profile
from recourse_snapshot import capture_recourse_snapshot, resume_from_snapshot
from counterfactual_teacher import (_incumbent_plans, _snapshot_with_force, _eval)
from run_headroom_census import (Probe, _load_profile, _dist_greedy_repair,
                                 _exhaustive_repair, _eval_plan, _plan_copy)


def _terminal_baseline(renv, snapshot):
    traces, served = renv.run_resumed(snapshot)
    return _eval(renv, int(snapshot['instance_id']), traces, 'coldchain', served_mask=served)


def _terminal_plan(renv, snapshot, plan, mutable_ids):
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
    ap.add_argument('--max-snapshots-per-inst', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective

    # 同时采集 snapshot（pre-plan）与 Probe 状态（post-plan），按 (inst, event_id) 对齐
    snapshots = {}
    per_inst_cap = {}

    def snap_hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        cnt = per_inst_cap.get(int(inst), 0)
        if cnt >= args.max_snapshots_per_inst:
            return
        key = (int(inst), int(eid))
        if key not in snapshots:
            snapshots[key] = capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac)
            per_inst_cap[int(inst)] = cnt + 1

    probe = Probe(contract, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.snapshot_hook = snap_hook
        env.run(inst)
    states = probe.states
    n_missing = 0
    for st in states:
        if (st['inst'], st['event']) not in snapshots:
            n_missing += 1
    print(f'collected {len(states)} states, {len(snapshots)} snapshots, '
          f'missing={n_missing}', flush=True)
    if n_missing:
        print('  WARNING: some states lack a matching snapshot (event beyond cap)', flush=True)

    rows = []
    for st in states:
        key = (st['inst'], st['event'])
        snap = snapshots.get(key)
        if snap is None:
            continue
        inst_idx = st['inst']
        env = st['env']
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']

        # myopic 枚举（复用 D1 逻辑）
        P_dg = _dist_greedy_repair(env, inst_idx, partial, mask_set, mutable_ids)
        r_dg = _eval_plan(vis, P_dg, contract, objective) if P_dg is not None else None
        J_dg = float(r_dg.J_vis) if r_dg is not None else None
        P_exh, J_exh, _stats = _exhaustive_repair(env, inst_idx, partial, mask_set,
                                                  mutable_ids, vis, contract, objective)

        # 终局前瞻 rollout（每状态一个独立 continuation，隔离 deferred 状态）
        renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity, 1.0,
                                    args.num_vehicles, coldchain_contract=contract)
        o0 = _terminal_baseline(renv, snap)
        o_dg = _terminal_plan(renv, snap, P_dg, mutable_ids) if P_dg is not None else None
        o_exh = _terminal_plan(renv, snap, P_exh, mutable_ids) if P_exh is not None else None

        row = {
            'inst': inst_idx, 'event': st['event'],
            'myopic_J0': st['J0'], 'myopic_J_dg': J_dg, 'myopic_J_exh': J_exh,
            'myopic_gap_exh_vs_dg': (J_dg - J_exh) if (J_dg is not None and J_exh is not None) else None,
            'terminal_J0': float(o0['coldchain_cost']),
            'terminal_J_dg': float(o_dg['coldchain_cost']) if o_dg is not None else None,
            'terminal_J_exh': float(o_exh['coldchain_cost']) if o_exh is not None else None,
            'terminal_headroom_dg': (float(o0['coldchain_cost']) - float(o_dg['coldchain_cost'])
                                     if o_dg is not None else None),
            'terminal_headroom_exh': (float(o0['coldchain_cost']) - float(o_exh['coldchain_cost'])
                                      if o_exh is not None else None),
            'terminal_gap_exh_vs_dg': ((float(o_dg['coldchain_cost']) - float(o_exh['coldchain_cost']))
                                       if (o_dg is not None and o_exh is not None) else None),
            'terminal_quality_delta_exh_vs_dg': (
                (float(o_dg['quality_loss']) - float(o_exh['quality_loss']))
                if (o_dg is not None and o_exh is not None) else None),
            'terminal_energy_delta_exh_vs_dg': (
                (float(o_dg['energy_kwh']) - float(o_exh['energy_kwh']))
                if (o_dg is not None and o_exh is not None) else None),
            'terminal_distance_delta_exh_vs_dg': (
                (float(o_dg['distance_km']) - float(o_exh['distance_km']))
                if (o_dg is not None and o_exh is not None) else None),
            'complete_0': bool(o0['complete']),
            'complete_exh': bool(o_exh['complete']) if o_exh is not None else None,
        }
        rows.append(row)
        print(f"  [inst {inst_idx}/evt {st['event']}] myo_gap={row['myopic_gap_exh_vs_dg']} "
              f"term_headroom_exh={row['terminal_headroom_exh']} "
              f"term_gap_exh={row['terminal_gap_exh_vs_dg']}", flush=True)

    def _inst_agg(key):
        per = {}
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            per.setdefault(r['inst'], []).append(v)
        return [float(np.mean(v)) for v in per.values()]

    def _ci(vals, n_boot=2000, seed=0):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(seed)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(n_boot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}

    summary = {
        'split': args.split, 'n_states': len(rows),
        'n_instances': len(set(r['inst'] for r in rows)),
        'myopic_gap_exh_vs_dg': _ci(_inst_agg('myopic_gap_exh_vs_dg')),
        'terminal_gap_exh_vs_dg': _ci(_inst_agg('terminal_gap_exh_vs_dg')),
        'terminal_headroom_dg': _ci(_inst_agg('terminal_headroom_dg')),
        'terminal_headroom_exh': _ci(_inst_agg('terminal_headroom_exh')),
        'terminal_component_delta_exh_vs_dg': {
            'quality': _ci(_inst_agg('terminal_quality_delta_exh_vs_dg')),
            'energy': _ci(_inst_agg('terminal_energy_delta_exh_vs_dg')),
            'distance': _ci(_inst_agg('terminal_distance_delta_exh_vs_dg')),
        },
        'n_states_terminal_headroom_exh_gt_0': int(
            sum(1 for r in rows if (r['terminal_headroom_exh'] or 0) > 1e-9)),
        'n_states_terminal_gap_exh_gt_myopic_gap': int(
            sum(1 for r in rows
                if (r['terminal_gap_exh_vs_dg'] or 0) > (r['myopic_gap_exh_vs_dg'] or 0) + 1e-9)),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n=== D3 terminal-lookahead headroom ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
