"""event_plan_v1 E2 数据导出：TRAIN baseline 轨迹 → eligible 事件 → 候选 → 标签 → 特征。

采样：每实例先按当前可见 mutable scope + 非空 decision pool 定义 eligible 事件，再按公开
clock 分层固定抽 ≤6 个；不按候选成功/正收益/真实未来筛选。记录全部事件/eligible/有非 KEEP
三个分母。

标签 g_B = J(P0 -> B) - J(P -> B)，正为改善；共同 JF1-H-F continuation。候选由 G_simple_r1
生成，不读 g_B。

用法（服务器）：
    python scripts/evaluation/run_event_plan_export.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 64 --max-events-per-instance 6 --out results/m0_scale/event_plan_export
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
                                    rollout_baseline)
from action_contract import build_vehicle_plans
from recourse_snapshot import capture_recourse_snapshot
from dynmaskco_cc_context import compute_incumbent_plans, decision_pool_from_vehicles
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from coldchain_utility_head import feature_only_context
from coldchain_visible_features import extract_context_features
from feature_local_replanner import PrePrepareContextCollector, _npz_view
from event_plan_candidate import (generate_simple_candidates_r1, full_plan_dict, full_plan_hash)
from event_plan_features import extract_plan_delta


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


def _context_vec(env, inst_idx, vehicles, visible_ids, deindex=True):
    import jax.numpy as jnp
    vis = np.zeros(env.num_nodes, bool); vis[0] = True
    for c in visible_ids:
        vis[int(c)] = True
    snap = {
        'num_vehicles': env.num_vehicles, 'visible_mask': vis,
        'vehicle_node': np.array([v.current_node for v in vehicles], np.int32),
        'vehicle_ready': np.array([v.ready_time for v in vehicles], np.float64),
        'vehicle_load': np.array([v.current_load for v in vehicles], np.float64),
        'needs_replan': np.array([v.needs_replan for v in vehicles], bool),
        'committed_next': np.array([-1 if v.committed_next is None else v.committed_next
                                    for v in vehicles], np.int32),
        'committed_arrive': np.array([np.nan if v.committed_arrive is None else v.committed_arrive
                                      for v in vehicles], np.float64),
        'committed_finish': np.array([np.nan if v.committed_finish is None else v.committed_finish
                                      for v in vehicles], np.float64),
        'vehicle_coldchain_state': [None for _ in vehicles],
    }
    feats = extract_context_features(_npz_view(env, inst_idx), 0, snap, deindex=deindex)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=64)
    ap.add_argument('--max-events-per-instance', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None
    cont = make_continuation()
    roll_env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)

    rows = []
    denom = {'all_events': 0, 'eligible_events': 0, 'non_keep_events': 0}
    t0 = time.time()
    for inst in range(n):
        # 跑 baseline，收集 eligible 事件（snapshot + context）
        collector = PrePrepareContextCollector(deindex=True, save_snapshot=True)
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        benv.snapshot_hook = collector.hook
        benv.run(inst)
        elig = []
        for (i, e, clk), ctx in sorted(collector.cache.items()):
            snap = collector.snapshot_cache[(i, e, clk)]
            # decision pool 是否非空
            from dynmaskco_cc_context import restore_vehicles_from_json
            veh = restore_vehicles_from_json(snap)
            vis_ids = [int(c) for c in snap['customer_universe']
                       if bool(snap['visible_mask'][int(c)])]
            pool = decision_pool_from_vehicles(veh, snap['served_mask'], vis_ids)
            denom['all_events'] += 1
            if pool:
                denom['eligible_events'] += 1
                elig.append((e, clk, snap, ctx))
        # 按 clock 分层抽 ≤6
        if len(elig) > args.max_events_per_instance:
            idx = np.unique(np.linspace(0, len(elig) - 1, args.max_events_per_instance)
                            .round().astype(int))
            elig = [elig[int(j)] for j in idx]

        for (e, clk, snap, ctx) in elig:
            P0, vehicles, replan_ids = compute_incumbent_plans(roll_env, snap, cont)
            mutable_ids = {v.vehicle_id for v in vehicles
                           if v.status in ('idle', 'ready') and v.needs_replan}
            vis_ids = [int(c) for c in snap['customer_universe']
                       if bool(snap['visible_mask'][int(c)])]
            pool = decision_pool_from_vehicles(vehicles, snap['served_mask'], vis_ids)
            keep = rollout_baseline(roll_env, snap, args.objective)
            J0 = _cost(keep, args.objective)

            records, plans = generate_simple_candidates_r1(roll_env, inst, P0, pool, mutable_ids,
                                                           seed=0)
            cands = []
            n_non_keep = 0
            for rec, P in zip(records, plans):
                h = full_plan_hash(P, set())
                snap_c = _snapshot_with_force(roll_env, snap, P, mutable_ids=mutable_ids)
                out = rollout_baseline(roll_env, snap_c, args.objective)
                gB = J0 - _cost(out, args.objective)
                n_non_keep += 1
                cands.append({
                    'candidate_id': h, 'requested_mask_size': rec['requested_mask_size'],
                    'selected_customer_ids': rec['selected_customer_ids'],
                    'applied_edit_customer_ids': rec['applied_edit_customer_ids'],
                    'full_plan': full_plan_dict(P, set()), 'deferred': [],
                    'gB': float(gB), 'J': _cost(out, args.objective),
                    'D': float(out['distance_cost']), 'Q': float(out['quality_loss']),
                    'E': float(out['energy_kwh']), 'service': bool(out.get('complete')),
                    'plan_delta': extract_plan_delta(roll_env, inst, P0, P,
                                                     args.num_vehicles).tolist(),
                })
            if n_non_keep:
                denom['non_keep_events'] += 1
            rows.append({'inst': inst, 'event': int(e), 'clock': float(clk),
                         'keep_J': J0, 'context': ctx.tolist(),
                         'P0': full_plan_dict(P0, set()), 'n_candidates': len(cands),
                         'keep_plan_delta': extract_plan_delta(roll_env, inst, P0, P0,
                                                               args.num_vehicles).tolist(),
                         'cands': cands})
            print(f"  [inst{inst}/evt{e}] n_cand={len(cands)} gB_range="
                  f"{[round(c['gB'],3) for c in cands] if cands else []}", flush=True)

    total_t = time.time() - t0
    summary = {'n_instances': n, 'n_events': len(rows), 'denominators': denom,
               'total_runtime': total_t, 'schema': 'event_plan_v1'}
    with open(os.path.join(args.out, 'data.json'), 'w') as f:
        json.dump({'summary': summary, 'rows': rows}, f, indent=2)
    print(f"\n  n_events={len(rows)} denom={denom} runtime={total_t:.0f}s")
    print(f"saved: {args.out}/data.json")


if __name__ == '__main__':
    main()
