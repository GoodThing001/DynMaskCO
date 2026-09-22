"""M1/feature-only 效用反事实诊断（Work Package B）。

对选定的决策点 context，计算：
  g_true = J_KEEP − J_a（同一 continuation 的隔离反事实 rollout）
  g_pred = s · [f(a) − f(KEEP)]（模型预测的相对 KEEP 改善）

两者均为正 = 预测与实际都认为有益。输出逐动作的配对，用于回答：
  预测会改善的动作实际改善比例、错误接受的平均/最大损失、是大量小错误还是少数灾难性错误、
  恶化主要来自 D/Q/E 哪个分量。

用法（服务器）：
    python scripts/evaluation/run_utility_diagnosis.py \
        --data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
        --feature-only-ckpt results/m0dev/probe_feature_only/probe.ckpt \
        --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-contexts 12 --out results/m0dev/diag_fo
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from feature_only_replanner import load_feature_only_scorer
from counterfactual_teacher import (_eval, rollout_action, rollout_baseline, _incumbent_plans,
                                    _decision_context)
from action_contract import enumerate_actions_from_plans, find_customer_slot, apply_action
from recourse_snapshot import capture_recourse_snapshot
from coldchain_visible_features import extract_action_features
from coldchain_utility_head import feature_only_context
from dynmaskco_cc_context import compute_incumbent_plans
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import service_ok, outcome_protocol_error


def _load_profile(path):
    with open(path) as f:
        d = json.load(f)
    return ObjectiveProfile(d['name'], d['distance_scale'], d['quality_scale'],
                            d['energy_scale'], d['lambda_quality'], d['lambda_energy'],
                            d.get('scale_source', 'pilot'), d.get('dev_statistics'))


def _make_env(dataset, capacity, num_vehicles, profile, contract):
    cc = apply_objective_profile(contract, profile)
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles, replanner=make_continuation(),
                           coldchain_contract=cc)


def _ctx_summary(env, inst_idx, vehicles, visible_ids):
    import jax.numpy as jnp
    from coldchain_visible_features import extract_context_features
    from dynmaskco_cc_context import cc_state_dict
    N = env.num_nodes
    vis = np.zeros(N, bool)
    vis[0] = True
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
        'vehicle_coldchain_state': [cc_state_dict(v.coldchain_state) for v in vehicles],
    }
    ds = {'coords': env.coords[inst_idx:inst_idx+1], 'demands': env.demands[inst_idx:inst_idx+1],
          'tw_start': env.tw_start[inst_idx:inst_idx+1], 'tw_end': env.tw_end[inst_idx:inst_idx+1],
          'service_time': env.service_time[inst_idx:inst_idx+1],
          'temp_class': env.temp_class[inst_idx:inst_idx+1],
          'initial_quality': env.initial_quality[inst_idx:inst_idx+1],
          'reveal_time': env.reveal_time[inst_idx:inst_idx+1]}
    feats = extract_context_features(ds, 0, snap)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def _score_action(scorer, ctx, action_dict):
    av, avd = extract_action_features(action_dict)
    import jax.numpy as jnp
    s = np.asarray(scorer.score(jnp.asarray(ctx[None]), jnp.asarray(av[None, None], jnp.float32),
                                jnp.asarray(avd[None, None], bool)))[0]
    return float(np.asarray(s).reshape(-1)[0])


def _action_dict(cand):
    return {'is_pseudo': cand.action.incumbent,
            'pseudo': 'KEEP' if cand.action.incumbent else None,
            'action': {'customer': cand.action.customer, 'slot_kind': cand.action.slot.kind,
                       'slot_anchor': cand.action.slot.anchor, 'position': cand.action.position,
                       'predecessor': cand.action.predecessor,
                       'successor': cand.action.successor,
                       'incumbent': cand.action.incumbent}}


def _defer_dict(customer):
    return {'is_pseudo': True, 'pseudo': 'DEFER',
            'action': {'customer': customer, 'kind': 'defer'}}


def _valid_outcome(outcome):
    if outcome_protocol_error(outcome, 'coldchain', strict_repair=True) is not None:
        return False
    if not service_ok(outcome, 'coldchain'):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature-only-ckpt', default=None)
    ap.add_argument('--objective', default='coldchain')
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    ap.add_argument('--max-contexts', type=int, default=12)
    ap.add_argument('--max-instances', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = default_pilot_contract()
    scorer = load_feature_only_scorer(args.feature_only_ckpt) if args.feature_only_ckpt else None
    s_scale = scorer.s if scorer is not None else 1.0

    # 1. 跑 baseline 收集决策点 snapshot
    env = _make_env(dataset, args.capacity, args.num_vehicles, profile, contract)
    snapshots = []

    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        replan_ids = {v.vehicle_id for v in veh
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if replan_ids:
            snapshots.append((inst, capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr,
                                                              sm, ac)))

    env.snapshot_hook = hook
    for inst in range(min(args.max_instances, dataset['coords'].shape[0])):
        env.run(inst)

    # 2. 按 instance 均匀取 context（早/中/晚），每个 context 取第一个 customer
    rows = []
    n_ctx = 0
    inst_snaps = {}
    for inst, snap in snapshots:
        inst_snaps.setdefault(inst, []).append(snap)

    for inst, snaps in inst_snaps.items():
        idx = sorted(set(np.linspace(0, len(snaps) - 1, 3).astype(int))) if len(snaps) >= 3 else range(len(snaps))
        for i in idx:
            if n_ctx >= args.max_contexts:
                break
            snap = snaps[i]
            r = _diagnose_context(env, snap, scorer, profile, contract, s_scale)
            if r is not None:
                rows.append(r)
                n_ctx += 1
        if n_ctx >= args.max_contexts:
            break

    # 3. 汇总
    pred_pos = [r for r in rows if r['g_pred'] > 0]
    true_pos = [r for r in rows if r['g_true'] > 0]
    agree = sum(1 for r in rows if (r['g_pred'] > 0) == (r['g_true'] > 0))
    false_accept = [r for r in rows if r['g_pred'] > 0 > r['g_true']]
    pred_improve_rate = (sum(1 for r in pred_pos if r['g_true'] > 0) / len(pred_pos)
                         if pred_pos else float('nan'))
    fa_loss_mean = np.mean([-r['g_true'] for r in false_accept]) if false_accept else 0.0
    fa_loss_max = max([-r['g_true'] for r in false_accept], default=0.0)

    print("\n=== utility diagnosis ===")
    print(f"  contexts={len(rows)}  pred_improve={len(pred_pos)}  true_improve={len(true_pos)}")
    print(f"  sign_agree={agree}/{len(rows)} ({agree/len(rows):.1%})")
    print(f"  pred_improve_rate={pred_improve_rate:.1%}  "
          f"false_accept={len(false_accept)} mean_loss={fa_loss_mean:.4f} max_loss={fa_loss_max:.4f}")

    with open(os.path.join(args.out, 'diagnosis.json'), 'w') as f:
        json.dump({'rows': rows,
                   'summary': {'n_contexts': len(rows), 'sign_agree': agree,
                               'pred_improve_rate': pred_improve_rate,
                               'n_false_accept': len(false_accept),
                               'false_accept_mean_loss': fa_loss_mean,
                               'false_accept_max_loss': fa_loss_max}}, f, indent=2)
    print(f"saved: {args.out}/diagnosis.json")


def _diagnose_context(env, snap, scorer, profile, contract, s_scale):
    inst_idx = int(snap['instance_id'])
    cont = make_continuation()
    inc_plans, vehicles, replan_ids = compute_incumbent_plans(env, snap, cont)
    from dynmaskco_cc_context import decision_pool_from_vehicles
    served = snap['served_mask']
    vis = [int(c) for c in snap['customer_universe'] if bool(snap['visible_mask'][int(c)])]
    pool = decision_pool_from_vehicles(vehicles, served, vis)
    if not pool:
        return None
    customer = sorted(pool, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c)))[0]
    ctx = _ctx_summary(env, inst_idx, vehicles, vis)
    cands, _ = enumerate_actions_from_plans(env, inst_idx, inc_plans, customer,
                                            allowed_vehicle_ids=replan_ids)
    feasible = [c for c in cands if c.feasible]
    if not feasible:
        return None

    keep_cand = next((c for c in feasible if c.action.incumbent), None)
    if keep_cand is not None:
        f_keep = _score_action(scorer, ctx, _action_dict(keep_cand))
    else:
        f_keep = _score_action(scorer, ctx, _defer_dict(customer))

    keep_outcome = rollout_baseline(env, snap, objective='coldchain')
    if not _valid_outcome(keep_outcome):
        return None
    J_keep = float(keep_outcome['coldchain_cost'])

    best = max((c for c in feasible if not c.action.incumbent),
               key=lambda c: _score_action(scorer, ctx, _action_dict(c)), default=None)
    if best is None:
        return None
    f_best = _score_action(scorer, ctx, _action_dict(best))
    g_pred = s_scale * (f_best - f_keep)
    outcome, _ = rollout_action(env, snap, best.action, cont, inc_plans, objective='coldchain',
                                allowed_vehicle_ids=replan_ids, mutable_ids=replan_ids)
    if not _valid_outcome(outcome):
        return None
    J_a = float(outcome['coldchain_cost'])
    g_true = J_keep - J_a

    dD = float(outcome['distance_cost']) - float(keep_outcome['distance_cost'])
    dQ = float(outcome['quality_loss']) - float(keep_outcome['quality_loss'])
    dE = float(outcome['energy_kwh']) - float(keep_outcome['energy_kwh'])
    wD = dD / profile.distance_scale
    wQ = profile.lambda_quality * dQ / profile.quality_scale
    wE = profile.lambda_energy * dE / profile.energy_scale

    return {
        'inst_idx': inst_idx, 'event_id': int(snap['event_id']), 'customer': int(customer),
        'action_id': best.action.action_id(),
        'has_incumbent': keep_cand is not None,
        'f_action': float(f_best), 'f_keep': float(f_keep),
        'raw_score_diff': float(f_best - f_keep), 's': float(s_scale),
        'g_pred': float(g_pred), 'g_true': float(g_true),
        'J_keep': J_keep, 'J_a': J_a,
        'dD': dD, 'dQ': dQ, 'dE': dE, 'wD': wD, 'wQ': wQ, 'wE': wE,
    }


if __name__ == '__main__':
    main()
