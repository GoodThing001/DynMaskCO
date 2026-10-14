"""A · 第 1 步：终局增益可预测性回归探针（采样版，便宜）。

回答：单条候选动作的**终局增益** `terminal_gain = J_keep(终局) − J_action(终局)` 能否从
决策点可见状态特征预测。预测不了 → A（前瞻价值函数）判死。

与完整顺序 oracle 不同，本版**不做 clairvoyant 逐客户搜索**，而是采样：
  1. baseline（JF1-H）跑一遍，snapshot_hook 采集每实例前 2 个决策点；
  2. 每决策点取前 K 个可见客户，枚举全部可行动作；
  3. 对每个动作做一次终局 rollout 得 terminal_gain + 可见特征 + myopic_gain；
  4. 线性最小二乘探针：特征 → terminal_gain（按实例 train/test），报告 out-of-sample
     Spearman / R²，以及 myopic_gain 与 terminal_gain 的直接 Spearman。

成本：8 实例 × 2 决策点 × 3 客户 × ~17 动作 ≈ ~800 次终局 rollout（每次 ~0.7s）≈ 10 分钟。

判读：Spearman ≈ 0（且 myopic 反相关）→ 可见状态无可学终局信号，A 判死。
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
from action_contract import apply_action, enumerate_actions_from_plans
from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from recourse_snapshot import capture_recourse_snapshot, resume_from_snapshot
from counterfactual_teacher import (_eval, _snapshot_with_force, rollout_action,
                                    _incumbent_plans)
from sequential_oracle import (customer_order_key, decision_pool, mutable_vehicle_ids,
                               _rollout_plan)
from hard_gate import service_ok, outcome_protocol_error
from run_causal_headroom import _incumbent_plans_and_vis, _myopic_J


FEATURE_NAMES = ['tw_start', 'tw_end', 'service_time', 'demand', 'temp_class', 'initial_quality',
                 'reveal_time', 'dist_to_depot', 'incremental_distance', 'clock',
                 'n_pool', 'n_deferred', 'n_vehicles', 'total_load', 'max_load', 'n_idle_ready']


def _visible_features(env, inst_idx, snapshot, vis, incr_dist, customer):
    c = int(customer)
    clock = float(snapshot['clock'])
    loads = [float(v.current_load) for v in vis.vehicles]
    n_idle_ready = sum(1 for v in vis.vehicles if v.status in ('idle', 'ready'))
    return [float(env.tw_start[inst_idx, c]), float(env.tw_end[inst_idx, c]),
            float(env.service_time[inst_idx, c]), float(env.demands[inst_idx, c]),
            float(env.temp_class[inst_idx, c]), float(env.initial_quality[inst_idx, c]),
            float(env.reveal_time[inst_idx, c]), float(env.dist_mat[inst_idx, c, 0]),
            float(incr_dist), clock,
            float(len(vis.pool_customer_ids)), float(len(vis.deferred_customer_ids)),
            float(len(vis.vehicles)), float(sum(loads)), float(max(loads) if loads else 0.0),
            float(n_idle_ready)]


def _spearman(x, y):
    if len(x) < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _linear_fit(X, y, lam=1e-3):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=8)
    ap.add_argument('--snapshots-per-inst', type=int, default=2)
    ap.add_argument('--customers-per-snapshot', type=int, default=3)
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
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective

    records = []
    for inst in range(args.max_instances):
        # baseline 跑一遍，采集前 K 个决策点 snapshot
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        snaps = []

        def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
            if sum(1 for s in snaps if s['instance_id'] == int(i)) < args.snapshots_per_inst:
                snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))

        env.snapshot_hook = hook
        env.run(inst)

        for snap in snaps:
            renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity,
                                        1.0, args.num_vehicles, coldchain_contract=contract)
            plans, vis = _incumbent_plans_and_vis(renv, snap, renv.replanner, contract, objective)
            keep_outcome, _ = _rollout_plan(renv, snap, plans, renv.replanner, 'coldchain',
                                            None, mutable_ids=mutable_vehicle_ids(snap))
            keep_cost = float(keep_outcome['coldchain_cost'])
            keep_myopic = _myopic_J(vis, plans, contract, objective)
            mutable_ids = mutable_vehicle_ids(snap)
            pool = sorted(decision_pool(snap), key=customer_order_key(renv, inst))

            for customer in pool[:args.customers_per_snapshot]:
                cands, _ = enumerate_actions_from_plans(renv, inst, plans, customer,
                                                        allowed_vehicle_ids=mutable_ids)
                for cand in cands:
                    if not cand.feasible:
                        continue
                    outcome, _ = rollout_action(renv, snap, cand.action, renv.replanner, plans,
                                                objective='coldchain',
                                                allowed_vehicle_ids=mutable_ids,
                                                mutable_ids=mutable_ids)
                    err = outcome_protocol_error(outcome, 'coldchain', strict_repair=True)
                    if err is not None:
                        raise RuntimeError(f"PROTOCOL_ERROR {err}")
                    if not service_ok(outcome, 'coldchain'):
                        continue
                    act_cost = float(outcome['coldchain_cost'])
                    terminal_gain = keep_cost - act_cost  # >0 = 改善
                    new_plans = apply_action(plans, cand.action, allowed_vehicle_ids=mutable_ids)
                    jm = _myopic_J(vis, new_plans, contract, objective)
                    myopic_gain = (keep_myopic - jm) if (keep_myopic is not None and jm is not None) else None
                    feats = _visible_features(renv, inst, snap, vis,
                                              float(cand.incremental_distance), customer)
                    records.append({'inst': inst, 'event_id': int(snap['event_id']),
                                    'customer': int(customer),
                                    'terminal_gain': terminal_gain, 'myopic_gain': myopic_gain,
                                    'features': feats,
                                    'action_id': cand.action.action_id()})
        print(f"  [inst {inst}] snapshots={len(snaps)} records={sum(1 for r in records if r['inst']==inst)}",
              flush=True)

    n = len(records)
    with open(os.path.join(args.out, 'records.json'), 'w') as f:
        json.dump(records, f, indent=2, default=str)
    print(f"total records={n}", flush=True)

    # 分析
    tg = np.array([r['terminal_gain'] for r in records])
    pairs = [(r['terminal_gain'], r['myopic_gain']) for r in records if r['myopic_gain'] is not None]
    corr_myopic = _spearman(np.array([p[1] for p in pairs]), np.array([p[0] for p in pairs]))

    Xstruct = np.array([r['features'] for r in records], dtype=float)
    Xstruct = (Xstruct - Xstruct.mean(0)) / (Xstruct.std(0) + 1e-9)
    Xfull = np.zeros((n, Xstruct.shape[1] + 1))
    Xfull[:, :-1] = Xstruct
    for i, r in enumerate(records):
        Xfull[i, -1] = r['myopic_gain'] if r['myopic_gain'] is not None else 0.0
    y = tg

    insts = sorted(set(r['inst'] for r in records))
    n_train = max(1, int(len(insts) * 0.7))
    train_insts = set(insts[:n_train])
    tr = np.array([r['inst'] in train_insts for r in records])
    te = ~tr

    def _probe(cols):
        Xs = Xfull[:, cols]
        w = _linear_fit(Xs[tr], y[tr])
        pred = Xs[te] @ w
        return {'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                'spearman_test': _spearman(pred, y[te]), 'r2_test': _r2(y[te], pred)}

    struct_cols = list(range(Xstruct.shape[1]))
    struct_plus_myopic = struct_cols + [Xfull.shape[1] - 1]

    summary = {
        'n_records': n,
        'terminal_gain': {'mean': float(np.mean(tg)), 'median': float(np.median(tg)),
                          'n_improving': int(sum(1 for x in tg if x > 1e-9)),
                          'n_worsening': int(sum(1 for x in tg if x < -1e-9))},
        'myopic_gain_vs_terminal_gain_spearman': corr_myopic,
        'linear_probe': {
            'structural_only': _probe(struct_cols),
            'structural_plus_myopic': _probe(struct_plus_myopic),
        },
        'feature_names': FEATURE_NAMES + ['myopic_gain'],
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== value predictability probe (sampled) ===")
    print(f"  n_records={n}  terminal_gain mean={np.mean(tg):.4f} median={np.median(tg):.4f} "
          f"improving={summary['terminal_gain']['n_improving']} worsening={summary['terminal_gain']['n_worsening']}")
    print(f"  Spearman(myopic_gain, terminal_gain) = {corr_myopic:+.3f}")
    print(f"  linear probe (structural only):   {json.dumps(summary['linear_probe']['structural_only'])}")
    print(f"  linear probe (structural+myopic): {json.dumps(summary['linear_probe']['structural_plus_myopic'])}")
    print(f"saved: {args.out}/summary.json + records.json")


if __name__ == '__main__':
    main()
