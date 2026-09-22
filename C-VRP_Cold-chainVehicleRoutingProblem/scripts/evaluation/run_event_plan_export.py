"""event_plan_v2 E2 数据导出：TRAIN baseline 轨迹 → eligible 事件 → 候选 → 标签 → 特征。

采样：每实例先按当前可见 mutable scope + 非空 decision pool 定义 eligible 事件，再按公开
clock 分层固定抽 ≤6 个；不按候选成功/正收益/真实未来筛选。记录全部事件/eligible/有非 KEEP
三个分母。

标签 g_B = J(P0 -> B) - J(P -> B)，正为改善；共同 JF1-H-F continuation。候选由 G_simple_r1
生成，不读 g_B。

v2 修正（共同实现偏差，两组 A/B 共用）：
  - 请求/计划绑定：每个成功计划带自己的 request_id / mask / selected / applied；
  - KEEP 入 seen 去重（不再带零增益克隆）；
  - 保存真实 deferred / has_future（WAIT/RETURN 控制指令）/ hard vector / 完整分区认证；
  - 同时导出 A（plan_delta）与 B（可见执行后果）两种表示特征；
  - 保存 snapshot 身份（state_hash）与 contract/profile 身份。

用法（本地/服务器）：
    python scripts/evaluation/run_event_plan_export.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 64 --max-events-per-instance 6 --out results/m0_scale/event_plan_v2_export
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
from dynmaskco_cc_context import (compute_incumbent_plans, decision_pool_from_vehicles,
                                  validate_full_partition)
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from coldchain_utility_head import feature_only_context
from coldchain_visible_features import extract_context_features
from feature_local_replanner import PrePrepareContextCollector, _npz_view
from hard_gate import hard_vector_from_outcome, service_ok
from event_plan_candidate import (generate_simple_candidates_r1, full_plan_dict, full_plan_hash,
                                  plan_distance)
from event_plan_features import (extract_plan_delta, extract_plan_consequence,
                                 plan_dim, B_VEH_DIM, B_FLEET_DIM)
from event_plan_projection import project_plan, proxy_cost


def _plan_proxy_J(env, inst_idx, vehicles, P, contract, has_future):
    proj = project_plan(env, inst_idx, vehicles, P, contract, has_future)
    return float(sum(proxy_cost(p, contract.objective) for p in proj.values()))


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


def _partition_check(env, inst_idx, clock, vehicles, P, deferred, served_mask):
    committed = {int(v.committed_next) for v in vehicles
                 if v.status == 'committed' and v.committed_next not in (None, 0)}
    universe = [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6
                and int(c) not in committed]
    suffix_set = {int(x) for p in P.values() for x in p.suffix}
    trial_deferred = [int(c) for c in deferred if c not in suffix_set and c not in committed]
    ok, detail = validate_full_partition(
        committed, P, trial_deferred, universe, served_mask=served_mask,
        future_fn=lambda c: env.reveal_time[inst_idx, c] > clock + 1e-6)
    return ok, {k: v for k, v in detail.items() if v}


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
    denom = {'all_events': 0, 'eligible_events': 0, 'non_keep_events': 0,
             'n_candidates_total': 0, 'n_keep_duplicates': 0, 'n_partition_fail': 0}
    t0 = time.time()
    for inst in range(n):
        collector = PrePrepareContextCollector(deindex=True, save_snapshot=True)
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        benv.snapshot_hook = collector.hook
        benv.run(inst)
        elig = []
        for (i, e, clk), ctx in sorted(collector.cache.items()):
            snap = collector.snapshot_cache[(i, e, clk)]
            from dynmaskco_cc_context import restore_vehicles_from_json
            veh = restore_vehicles_from_json(snap)
            vis_ids = [int(c) for c in snap['customer_universe']
                       if bool(snap['visible_mask'][int(c)])]
            pool = decision_pool_from_vehicles(veh, snap['served_mask'], vis_ids)
            denom['all_events'] += 1
            if pool:
                denom['eligible_events'] += 1
                elig.append((e, clk, snap, ctx))
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
            has_future = roll_env.has_future_reveal(inst, clk, snap['served_mask'])
            deferred = sorted(int(c) for c in cont.deferred_customers)
            keep = rollout_baseline(roll_env, snap, args.objective)
            J0 = _cost(keep, args.objective)
            keep_dist = plan_distance(roll_env, inst, P0, has_future)
            keep_proxy_J = _plan_proxy_J(roll_env, inst, vehicles, P0, eff, has_future)

            records, plan_entries = generate_simple_candidates_r1(
                roll_env, inst, P0, pool, mutable_ids, seed=0, has_future=has_future)
            denom['n_keep_duplicates'] += sum(1 for r in records if r.get('duplicate'))
            cands = []
            for entry in plan_entries:
                P = entry['plan']
                h = entry['candidate_id']
                snap_c = _snapshot_with_force(roll_env, snap, P, mutable_ids=mutable_ids)
                out = rollout_baseline(roll_env, snap_c, args.objective)
                gB = J0 - _cost(out, args.objective)
                ok, detail = _partition_check(roll_env, inst, clk, vehicles, P,
                                              deferred, snap['served_mask'])
                if not ok:
                    denom['n_partition_fail'] += 1
                denom['n_candidates_total'] += 1
                cands.append({
                    'candidate_id': h, 'request_id': entry['request_id'],
                    'requested_mask_size': entry['requested_mask_size'],
                    'selected_customer_ids': entry['selected_customer_ids'],
                    'applied_edit_customer_ids': entry['applied_edit_customer_ids'],
                    'full_plan': full_plan_dict(P, deferred, has_future),
                    'deferred': deferred,
                    'gB': float(gB), 'J': _cost(out, args.objective),
                    'D': float(out['distance_cost']), 'Q': float(out['quality_loss']),
                    'E': float(out['energy_kwh']),
                    'hard_vector': hard_vector_from_outcome(out),
                    'service_ok': bool(service_ok(out, args.objective)),
                    'partition_ok': bool(ok), 'partition_detail': detail,
                    'distance': float(plan_distance(roll_env, inst, P, has_future)),
                    'proxy_J': float(_plan_proxy_J(roll_env, inst, vehicles, P, eff,
                                                   has_future)),
                    'plan_delta': extract_plan_delta(roll_env, inst, P0, P,
                                                     args.num_vehicles).tolist(),
                    'consequence': extract_plan_consequence(
                        roll_env, inst, vehicles, P0, P, eff, profile, args.capacity,
                        args.num_vehicles, has_future).tolist(),
                })
            if len(plan_entries):
                denom['non_keep_events'] += 1
            rows.append({'inst': inst, 'event': int(e), 'clock': float(clk),
                         'keep_J': J0, 'context': ctx.tolist(),
                         'snapshot_id': snap['state_hash'],
                         'contract_hash': (eff.contract_hash if eff is not None else None),
                         'profile_hash': (profile.profile_hash if profile is not None else None),
                         'deferred': deferred, 'has_future': bool(has_future),
                         'P0': full_plan_dict(P0, deferred, has_future),
                         'keep_distance': float(keep_dist),
                         'keep_proxy_J': float(keep_proxy_J),
                         'n_candidates': len(plan_entries),
                         'records': records,
                         'keep_hard_vector': hard_vector_from_outcome(keep),
                         'keep_service_ok': bool(service_ok(keep, args.objective)),
                         'keep_plan_delta': extract_plan_delta(roll_env, inst, P0, P0,
                                                               args.num_vehicles).tolist(),
                         'keep_consequence': extract_plan_consequence(
                             roll_env, inst, vehicles, P0, P0, eff, profile, args.capacity,
                             args.num_vehicles, has_future).tolist(),
                         'cands': cands})
            print(f"  [inst{inst}/evt{e}] n_cand={len(plan_entries)} gB_range="
                  f"{[round(c['gB'],3) for c in cands] if cands else []}", flush=True)

    total_t = time.time() - t0
    summary = {'n_instances': n, 'n_events': len(rows), 'denominators': denom,
               'total_runtime': total_t, 'schema': 'event_plan_v2',
               'representation_A_dim': plan_dim('A', args.num_vehicles),
               'representation_B_dim': plan_dim('B', args.num_vehicles),
               'B_VEH_DIM': B_VEH_DIM, 'B_FLEET_DIM': B_FLEET_DIM,
               'objective': args.objective,
               'contract_hash': eff.contract_hash if eff is not None else None,
               'profile_hash': profile.profile_hash if profile is not None else None}
    with open(os.path.join(args.out, 'data.json'), 'w') as f:
        json.dump({'summary': summary, 'rows': rows}, f, indent=2)
    print(f"\n  n_events={len(rows)} denom={denom} runtime={total_t:.0f}s")
    print(f"saved: {args.out}/data.json")


if __name__ == '__main__':
    main()
