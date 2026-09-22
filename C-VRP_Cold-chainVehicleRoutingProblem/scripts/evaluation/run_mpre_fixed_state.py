"""M-pre 固定状态对照：同一公开状态下，R（regret-2） vs M-pre（预训练 CVRP 边 logits 重构）。

对每个决策点（prepare + JF1-H-F 形成 P0 + 保护集合之后），在同一公开状态上分别跑
`cc_lns_search`（R）与 `cc_lns_mpre_search`（M-pre），比较完整计划 J_vis 与 D/Q/E。
不写回、不训练。回答：预训练 MaskCO 边分数驱动的重构，在固定状态下是否优于/劣于 regret-2。

用法（本地，纯 NumPy + JAX CPU；cvrp100.ckpt 本地）：
    python scripts/evaluation/run_mpre_fixed_state.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 4.0 --n-rounds 2 --seed 0 --max-instances 4 \
        --out results/m0_scale/mpre_fixed_train
"""
import argparse
import csv
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from cc_swap import cc_lns_swap_search
from mpre_replanner import cc_lns_mpre_search
from mpre import load_cvrp_model
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from service_first import paired_bootstrap_ci


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


class MpreFixedStateProbe(JF1HRepairReplanner):
    """JF1-H-F + 预留保护；每决策点同状态跑 R 与 M-pre 两个搜索，只记录不写回。"""

    def __init__(self, contract, model, capacity, budget_s, seed, n_rounds):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.model = model
        self.capacity = float(capacity)
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.n_rounds = int(n_rounds)
        self.rows = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved_ids = set()
        idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
        reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in reserved_ids:
            v = vehicles[vid]
            p = P0.get(vid)
            if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                protected.add(vid)
        _, J_R, sR, _evR = cc_lns_swap_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, self.contract, self.budget_s, seed=self.seed,
            protected=protected, swap=False, n_rounds=self.n_rounds, max_attempts=None)
        _, J_M, sM, _evM = cc_lns_mpre_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, self.contract, self.budget_s, self.model, self.capacity,
            seed=self.seed, protected=protected, n_rounds=self.n_rounds)
        self.rows.append({
            'inst': int(inst_idx), 'event': int(getattr(env, 'event_id', -1)),
            'clock': float(clock),
            'P0_feasible': bool(sR['P0_feasible']), 'pool_nonempty': sR['stop_reason'] != 'empty_pool',
            'J_R': float(J_R), 'J_M': float(J_M), 'delta': float(J_M) - float(J_R),
            'D_R': float(sR['best_D']), 'D_M': float(sM['best_D']),
            'Q_R': float(sR['best_Q']), 'Q_M': float(sM['best_Q']),
            'E_R': float(sR['best_E']), 'E_M': float(sM['best_E']),
            'n_attempts_R': int(sR['n_attempts']), 'n_attempts_M': int(sM['n_attempts']),
            'n_improve_R': int(sR['n_improve']), 'n_improve_M': int(sM['n_improve']),
            'stop_R': sR['stop_reason'], 'stop_M': sM['stop_reason'],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=4.0)
    ap.add_argument('--n-rounds', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-instances', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None
    model, cfg, step = load_cvrp_model(args.cvrp_ckpt)
    print(f"loaded {os.path.basename(args.cvrp_ckpt)} embed_dim={cfg.embed_dim} step={step}",
          flush=True)

    rows = []
    for inst in range(n):
        rep = MpreFixedStateProbe(eff, model, args.capacity, budget_s=args.budget,
                                  seed=args.seed, n_rounds=args.n_rounds)
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=rep, coldchain_contract=eff)
        env.run(inst)
        rows.extend(rep.rows)
        n_better = sum(1 for r in rep.rows if r['delta'] < -1e-9)
        print(f"  [inst {inst}] n_states={len(rep.rows)} M_better={n_better}", flush=True)

    eligible = [r for r in rows if r['P0_feasible'] and r['pool_nonempty']]
    deltas = [r['delta'] for r in eligible]
    n_better = sum(1 for d in deltas if d < -1e-9)
    n_worse = sum(1 for d in deltas if d > 1e-9)
    n_tie = len(deltas) - n_better - n_worse
    comp = {}
    for k in ('D', 'Q', 'E'):
        dr = np.asarray([r[f'{k}_R'] for r in eligible])
        dm = np.asarray([r[f'{k}_M'] for r in eligible])
        comp[k] = {'mean_R': float(np.mean(dr)), 'mean_M': float(np.mean(dm)),
                   'mean_delta': float(np.mean(dm - dr))}
    summary = {
        'budget': args.budget, 'n_rounds': args.n_rounds, 'seed': args.seed,
        'n_instances': n, 'n_states': len(rows), 'n_eligible': len(eligible),
        'mean_J_R': float(np.mean([r['J_R'] for r in eligible])) if eligible else None,
        'mean_J_M': float(np.mean([r['J_M'] for r in eligible])) if eligible else None,
        'mean_delta': float(np.mean(deltas)) if deltas else None,
        'ci': list(paired_bootstrap_ci(deltas)) if len(deltas) >= 2 else None,
        'n_M_better': n_better, 'n_M_worse': n_worse, 'n_tie': n_tie,
        'component_deltas': comp,
        'mean_n_attempts_R': float(np.mean([r['n_attempts_R'] for r in eligible])) if eligible else None,
        'mean_n_attempts_M': float(np.mean([r['n_attempts_M'] for r in eligible])) if eligible else None,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, 'per_state.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['inst', 'event', 'J_R', 'J_M', 'delta', 'D_R', 'D_M', 'Q_R', 'Q_M',
                    'E_R', 'E_M', 'n_attempts_R', 'n_attempts_M', 'P0_feasible'])
        for r in rows:
            w.writerow([r['inst'], r['event'], f"{r['J_R']:.6f}", f"{r['J_M']:.6f}",
                        f"{r['delta']:+.6f}", f"{r['D_R']:.4f}", f"{r['D_M']:.4f}",
                        f"{r['Q_R']:.4f}", f"{r['Q_M']:.4f}", f"{r['E_R']:.4f}",
                        f"{r['E_M']:.4f}", r['n_attempts_R'], r['n_attempts_M'],
                        int(r['P0_feasible'])])
    print(f"\n=== M-pre fixed-state (budget={args.budget}s, n_states={len(rows)}) ===")
    print(f"  eligible={len(eligible)}  mean delta(J_M-J_R)={summary['mean_delta']:+.6f} "
          f"CI={[round(x,6) for x in (summary['ci'] or [])]}")
    print(f"  M_better={n_better}  M_worse={n_worse}  tie={n_tie}")
    print(f"  D/Q/E delta: D={comp['D']['mean_delta']:+.4f} Q={comp['Q']['mean_delta']:+.4f} "
          f"E={comp['E']['mean_delta']:+.4f}")
    print(f"saved: {args.out}/summary.json + per_state.json + per_state.csv")


if __name__ == '__main__':
    main()
