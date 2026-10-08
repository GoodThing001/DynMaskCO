"""第 1 步 · 因果头腔诊断：myopic 顺序 oracle vs clairvoyant 顺序 oracle。

回答：O0-CC 顺序 oracle 的 ΔJ≈−0.21/−0.27 里，有多少是"只看当前可见状态也能因果拿到"的，
有多少依赖 clairvoyant 未来（未来订单揭示后再终局 rollout）。

做法（无训练、无模型、纯 NumPy + JF1-H continuation）：
  - baseline：JF1-H-F 完整轨迹 → 终局 coldchain_cost；
  - clairvoyant 顺序 oracle（已跑，见 results/m0_scale/o0cc_seq_*）：每个决策点按客户顺序，
    对每个动作做终局 rollout（含未来揭示）选最优并写回；
  - **myopic 顺序 oracle（本脚本新增）**：同样的按客户顺序 + 写回，但动作评分换成
    "apply 后 evaluate_visible_plan（myopic J_vis，只看当前可见订单）"。

判读：
  - 若 myopic 顺序 oracle 终局 ≈ clairvoyant（−0.2 量级）→ 头腔几乎全部因果可拿，前瞻价值函数
    方向有明确上界与信号；
  - 若 myopic 顺序 oracle 终局 ≈ baseline（≈0，甚至更差）→ 头腔绝大部分是未来信息（VoFI），
    因果可学的分量未知，需进一步做"终局增益是否可从可见状态预测"的回归诊断再决定是否立项。

用法（服务器）：
    python scripts/evaluation/run_causal_headroom.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 16 --out results/m0_scale/causal_headroom_cal
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from action_contract import (build_vehicle_plans, apply_action, enumerate_actions_from_plans)
from cc_lns_replanner import _plan_suffixes
from visible_state import build_visible_state, evaluate_visible_plan
from coldchain_contract import default_pilot_contract, apply_objective_profile
from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot)
from counterfactual_teacher import (_eval, _snapshot_with_force, _decision_context)
from sequential_oracle import (customer_order_key, decision_pool, mutable_vehicle_ids)
from hard_gate import service_ok, outcome_protocol_error


def _incumbent_plans_and_vis(env, snapshot, continuation, contract, objective):
    """复刻 _incumbent_plans，同时返回 decision-point VisibleState（myopic 评分用）。"""
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    env.prepare_decision_point(clock, vehicles)
    if hasattr(continuation, 'restore_state'):
        continuation.restore_state(snapshot.get('replanner_state'))
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
    plans = build_vehicle_plans(env, inst_idx, vehicles)
    deferred = getattr(continuation, 'deferred_customers', set())
    vis = build_visible_state(env, inst_idx, clock, int(snapshot['event_id']), vehicles,
                              served_mask, visible_ids, replan_ids, deferred)
    return plans, vis


def _myopic_J(vis, plans, contract, objective):
    r = evaluate_visible_plan(vis, _plan_suffixes(plans), contract, objective)
    if not r.finite or not r.feasible:
        return None
    return float(r.J_vis)


def myopic_sequential_oracle_plan(env, snapshot, continuation, contract, objective, log=None):
    """按客户顺序 greedy 的 myopic 顺序 oracle：动作评分 = apply 后 myopic J_vis。"""
    inst_idx = int(snapshot['instance_id'])
    pool = sorted(decision_pool(snapshot), key=customer_order_key(env, inst_idx))
    mutable_ids = mutable_vehicle_ids(snapshot)
    plans, vis = _incumbent_plans_and_vis(env, snapshot, continuation, contract, objective)

    keep_J = _myopic_J(vis, plans, contract, objective)
    if keep_J is None:
        keep_J = float('inf')

    for customer in pool:
        cands, _ = enumerate_actions_from_plans(env, inst_idx, plans, customer,
                                                allowed_vehicle_ids=mutable_ids)
        best_J = keep_J
        best_action = None
        for cand in cands:
            if not cand.feasible:
                continue
            new_plans = apply_action(plans, cand.action, allowed_vehicle_ids=mutable_ids)
            J = _myopic_J(vis, new_plans, contract, objective)
            if J is not None and J < best_J:
                best_J = J
                best_action = cand.action
        if best_action is not None:
            plans = apply_action(plans, best_action, allowed_vehicle_ids=mutable_ids)
            keep_J = best_J
        if log is not None:
            log.append({'event_id': int(snapshot['event_id']), 'customer': int(customer),
                        'n_candidates': len(cands),
                        'n_feasible': sum(1 for c in cands if c.feasible),
                        'selected': (best_action.action_id() if best_action is not None
                                     else '__KEEP__')})
    return plans


def make_myopic_oracle_hook(renv, continuation, contract, objective, log=None):
    def hook(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers):
        snapshot = capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx,
                                             vehicles, traces, served_mask, all_customers)
        plan = myopic_sequential_oracle_plan(renv, snapshot, continuation, contract, objective,
                                             log=log)
        snap2 = _snapshot_with_force(renv, snapshot, plan,
                                     mutable_ids=mutable_vehicle_ids(snapshot))
        env.force_suffix = {tuple(k): list(v) for k, v in snap2['force_suffix']}
    return hook


def _run_instance(inst_idx, dataset, capacity, num_vehicles, contract, objective, out_dir):
    # baseline（JF1-H-F）
    env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                          replanner=make_continuation(), coldchain_contract=contract)
    traces, served = env.run(inst_idx)
    base = _eval(env, inst_idx, traces, 'coldchain', served_mask=served)
    err = outcome_protocol_error(base, 'coldchain', strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: baseline PROTOCOL_ERROR {err}")

    # myopic 顺序 oracle
    log = []
    oenv = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=make_continuation(), coldchain_contract=contract)
    renv = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=make_continuation(), coldchain_contract=contract)
    oenv.oracle_hook = make_myopic_oracle_hook(renv, renv.replanner, contract, objective,
                                               log=log)
    traces, served = oenv.run(inst_idx)
    orac = _eval(oenv, inst_idx, traces, 'coldchain', served_mask=served)
    err = outcome_protocol_error(orac, 'coldchain', strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: myopic oracle PROTOCOL_ERROR {err}")

    res = {'inst_idx': inst_idx,
           'baseline_cost': float(base['coldchain_cost']),
           'myopic_oracle_cost': float(orac['coldchain_cost']),
           'delta': float(orac['coldchain_cost']) - float(base['coldchain_cost']),
           'baseline_complete': bool(base['complete']),
           'myopic_complete': bool(orac['complete']),
           'n_decisions': len(log), 'n_accepted': sum(1 for r in log if r['selected'] != '__KEEP__')}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'inst_{inst_idx}.json'), 'w') as f:
            json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    with open(args.objective_profile) as f:
        d = json.load(f)
    from coldchain_contract import ObjectiveProfile
    profile = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                               quality_scale=float(d['quality_scale']),
                               energy_scale=float(d['energy_scale']),
                               lambda_quality=float(d['lambda_quality']),
                               lambda_energy=float(d['lambda_energy']),
                               scale_source=d.get('scale_source', 'pilot'),
                               dev_statistics=d.get('dev_statistics'))
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective

    results = [_run_instance(i, dataset, args.capacity, args.num_vehicles, contract,
                             objective, os.path.join(args.out, 'instances'))
               for i in range(args.max_instances)]

    deltas = [r['delta'] for r in results]
    from service_first import paired_bootstrap_ci
    lo, hi = paired_bootstrap_ci(deltas) if deltas else (float('nan'), float('nan'))
    summary = {
        'n_instances': len(results),
        'baseline_mean': float(np.mean([r['baseline_cost'] for r in results])),
        'myopic_oracle_mean': float(np.mean([r['myopic_oracle_cost'] for r in results])),
        'mean_delta': float(np.mean(deltas)),
        'ci_lo': lo, 'ci_hi': hi,
        'n_complete_baseline': sum(1 for r in results if r['baseline_complete']),
        'n_complete_myopic': sum(1 for r in results if r['myopic_complete']),
        'n_decisions_total': int(sum(r['n_decisions'] for r in results)),
        'n_accepted_total': int(sum(r['n_accepted'] for r in results)),
        'per_instance': results,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== causal headroom (myopic sequential oracle) ===")
    print(f"  baseline_mean={summary['baseline_mean']:.4f}  "
          f"myopic_oracle_mean={summary['myopic_oracle_mean']:.4f}")
    print(f"  delta(myopic-baseline)={summary['mean_delta']:+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"  complete: baseline={summary['n_complete_baseline']}/{len(results)} "
          f"myopic={summary['n_complete_myopic']}/{len(results)}")
    print(f"  decisions={summary['n_decisions_total']} accepted={summary['n_accepted_total']}")
    print(f"  (reference: clairvoyant sequential oracle delta ~ CAL -0.21 / DEV -0.27)")
    print(f"saved: {args.out}/summary.json")


if __name__ == '__main__':
    main()
