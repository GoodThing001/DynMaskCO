"""event_plan_v1 E0 + E1：接口正确性检查 + 20 个 TRAIN 决策点先导。

E0：KEEP parity、副本隔离、拒绝无残留、真实安装、有限值检查。
E1：从 TRAIN baseline 轨迹按固定规则选 20 个决策点，生成候选、计算 g_B、记录耗时与 coverage。

用法（服务器）：
    python scripts/evaluation/run_event_plan_pilot.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 8 --n-points 20 --out results/m0_scale/event_plan_pilot
"""
import argparse
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
from counterfactual_teacher import (_snapshot_with_force, _mutable_ids_from_snapshot,
                                    rollout_baseline, _eval)
from action_contract import build_vehicle_plans
from recourse_snapshot import capture_recourse_snapshot
from dynmaskco_cc_context import compute_incumbent_plans
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from event_plan_candidate import (generate_simple_candidates, full_plan_hash,
                                  dispatch_signature)


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(
        name=data['name'], distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']), energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']), lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'), dev_statistics=data.get('dev_statistics'))


def _cost(o, objective):
    return float(o['distance_cost'] if objective == 'distance' else o['coldchain_cost'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=8)
    ap.add_argument('--n-points', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None
    cont = make_continuation()

    # 跑 baseline 收集决策点 snapshot
    snapshots = []
    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        replan_ids = {v.vehicle_id for v in veh if v.status in ('idle', 'ready') and v.needs_replan}
        if replan_ids:
            snapshots.append(capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac))

    n = min(args.max_instances, dataset['coords'].shape[0])
    benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                           replanner=make_continuation(), coldchain_contract=eff)
    benv.snapshot_hook = hook
    for i in range(n):
        benv.run(i)
    print(f"collected {len(snapshots)} snapshots from {n} instances", flush=True)

    # 选 20 个决策点：跨实例均匀（不按收益选）
    if len(snapshots) <= args.n_points:
        sel = snapshots
    else:
        idxs = np.unique(np.linspace(0, len(snapshots) - 1, args.n_points).round().astype(int))
        sel = [snapshots[int(i)] for i in idxs]

    roll_env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
    rows = []
    e0 = {'keep_parity_ok': 0, 'copy_isolation_ok': 0, 'finite_ok': 0, 'total': 0}
    t0 = time.time()
    for si, snap in enumerate(sel):
        inst = int(snap['instance_id'])
        eid = int(snap['event_id'])
        clock = float(snap['clock'])
        # P_0（baseline incumbent）
        P0, vehicles, replan_ids = compute_incumbent_plans(roll_env, snap, cont)
        mutable_ids = {v.vehicle_id for v in vehicles
                       if v.status in ('idle', 'ready') and v.needs_replan}
        from dynmaskco_cc_context import decision_pool_from_vehicles
        pool = decision_pool_from_vehicles(vehicles, snap['served_mask'],
                                           [int(c) for c in snap['customer_universe']
                                            if bool(snap['visible_mask'][int(c)])])

        # KEEP rollout（parity：== baseline 终局）
        t_k = time.time()
        keep = rollout_baseline(roll_env, snap, args.objective)
        t_keep = time.time() - t_k
        # KEEP parity：force P0 后再 rollout 应与 rollout_baseline(snap) 一致（P0 即 baseline incumbent）
        keep_forced = rollout_baseline(roll_env, _snapshot_with_force(roll_env, snap, P0,
                                                                     mutable_ids=mutable_ids),
                                       args.objective)
        keep_parity = abs(_cost(keep, args.objective) - _cost(keep_forced, args.objective)) < 1e-9

        # E0 副本隔离：生成候选后 P0 不变
        P0_before = full_plan_hash(P0, set())
        t_g = time.time()
        cands = generate_simple_candidates(roll_env, inst, P0, pool, mutable_ids)
        t_gen = time.time() - t_g
        P0_after = full_plan_hash(P0, set())
        copy_iso = (P0_before == P0_after)

        # 候选 rollout：安装候选计划（_snapshot_with_force）后再 rollout
        seen = {full_plan_hash(P0, set())}
        cand_rows = []
        finite_ok = True
        for k, P in cands:
            h = full_plan_hash(P, set())
            if h in seen:
                continue
            seen.add(h)
            t_c = time.time()
            snap_cand = _snapshot_with_force(roll_env, snap, P, mutable_ids=mutable_ids)
            out = rollout_baseline(roll_env, snap_cand, args.objective)
            t_cand = time.time() - t_c
            gB = _cost(keep, args.objective) - _cost(out, args.objective)
            if not np.isfinite(gB):
                finite_ok = False
            cand_rows.append({
                'mask': k, 'gB': gB, 'J': _cost(out, args.objective),
                'D': float(out['distance_cost']), 'Q': float(out['quality_loss']),
                'E': float(out['energy_kwh']), 'service': bool(out.get('complete')),
                'dispatch': dispatch_signature(P), 'runtime': round(t_cand, 3),
            })
        rows.append({'inst': inst, 'event': eid, 'clock': clock, 'n_candidates': len(cand_rows),
                     'keep_J': _cost(keep, args.objective), 'cand_rows': cand_rows,
                     'copy_isolation': copy_iso, 'keep_parity': keep_parity,
                     'finite': finite_ok})
        e0['total'] += 1
        e0['copy_isolation_ok'] += int(copy_iso)
        e0['keep_parity_ok'] += int(keep_parity)
        e0['finite_ok'] += int(finite_ok)
        print(f"  [{si}] inst{inst}/evt{eid} n_cand={len(cand_rows)} t_keep={t_keep:.1f}s "
              f"t_gen={t_gen:.2f}s parity={keep_parity}", flush=True)

    total_t = time.time() - t0
    summary = {'e0': e0, 'n_points': len(sel), 'total_runtime': total_t, 'rows': rows}
    with open(os.path.join(args.out, 'pilot.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  E0: copy_isolation={e0['copy_isolation_ok']}/{e0['total']} "
          f"keep_parity={e0['keep_parity_ok']}/{e0['total']} "
          f"finite={e0['finite_ok']}/{e0['total']}")
    print(f"  total_runtime={total_t:.0f}s  saved: {args.out}/pilot.json")


if __name__ == '__main__':
    main()
