"""固定状态验证（Q1）：同一公开状态下，R vs R+两客户交换 的完整计划 J_vis / D/Q/E。

对每个决策点（prepare + JF1-H-F 形成 P0 + 确定保护集合之后、LNS 搜索之前），在**同一公开
状态**上分别跑 `cc_lns_swap_search(swap=False)`（= R）与 `cc_lns_swap_search(swap=True)`
（= R + 两客户交换），直接比较二者最终最佳计划的 `J_vis` 与 D/Q/E。不写回、不训练、不
改变 baseline 轨迹。

回答：在相同公开状态下，是否存在优于 R、且能用合法插入步骤表达的局部完整修复？
（J_vis 更低仍可能因提前返仓/资源占用损害后续动态服务，故 Q2 闭环另行检验。）

用法（服务器）：
    python scripts/evaluation/run_cc_lns_swap_fixed_state.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 4.0 --n-rounds 2 --max-attempts 64 --seed 0 \
        --max-instances 64 --out results/m0_scale/swap_fixed_train
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


class FixedStateProbe(JF1HRepairReplanner):
    """JF1-H-F + 预留保护；每决策点同状态跑 R 与 R+swap 两个搜索，只记录不写回。"""

    def __init__(self, contract, budget_s, seed, n_rounds, max_attempts, swap=True):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.n_rounds = int(n_rounds)
        self.max_attempts = max_attempts
        self.swap = bool(swap)
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
        # 同一公开状态跑两个搜索（不写回）
        _, J_R, sR, _evR = cc_lns_swap_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, self.contract, self.budget_s, seed=self.seed,
            protected=protected, swap=False, n_rounds=self.n_rounds,
            max_attempts=self.max_attempts)
        _, J_S, sS, _evS = cc_lns_swap_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, self.contract, self.budget_s, seed=self.seed,
            protected=protected, swap=self.swap, n_rounds=self.n_rounds,
            max_attempts=self.max_attempts)
        self.rows.append({
            'inst': int(inst_idx), 'event': int(getattr(env, 'event_id', -1)),
            'clock': float(clock),
            'P0_feasible': bool(sR['P0_feasible']), 'pool_nonempty': sR['stop_reason'] != 'empty_pool',
            'J_R': float(J_R), 'J_S': float(J_S), 'delta': float(J_S) - float(J_R),
            'D_R': float(sR['best_D']), 'D_S': float(sS['best_D']),
            'Q_R': float(sR['best_Q']), 'Q_S': float(sS['best_Q']),
            'E_R': float(sR['best_E']), 'E_S': float(sS['best_E']),
            'n_attempts_R': int(sR['n_attempts']), 'n_attempts_S': int(sS['n_attempts']),
            'n_swap_S': int(sS['n_swap']), 'n_improve_R': int(sR['n_improve']),
            'n_improve_S': int(sS['n_improve']), 'stop_R': sR['stop_reason'],
            'stop_S': sS['stop_reason'],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=4.0)
    ap.add_argument('--n-rounds', type=int, default=2)
    ap.add_argument('--max-attempts', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-instances', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None

    rows = []
    for inst in range(n):
        rep = FixedStateProbe(eff, budget_s=args.budget, seed=args.seed, n_rounds=args.n_rounds,
                              max_attempts=args.max_attempts, swap=True)
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=rep, coldchain_contract=eff)
        env.run(inst)
        rows.extend(rep.rows)
        n_better = sum(1 for r in rep.rows if r['delta'] < -1e-9)
        print(f"  [inst {inst}] n_states={len(rep.rows)} swap_better={n_better}", flush=True)

    eligible = [r for r in rows if r['P0_feasible'] and r['pool_nonempty']]
    deltas = [r['delta'] for r in eligible]
    n_better = sum(1 for d in deltas if d < -1e-9)
    n_worse = sum(1 for d in deltas if d > 1e-9)
    n_tie = len(deltas) - n_better - n_worse
    comp = {}
    for k in ('D', 'Q', 'E'):
        dr = np.asarray([r[f'{k}_R'] for r in eligible])
        ds = np.asarray([r[f'{k}_S'] for r in eligible])
        comp[k] = {'mean_R': float(np.mean(dr)), 'mean_S': float(np.mean(ds)),
                   'mean_delta': float(np.mean(ds - dr))}
    summary = {
        'budget': args.budget, 'n_rounds': args.n_rounds, 'max_attempts': args.max_attempts,
        'seed': args.seed, 'n_instances': n, 'n_states': len(rows),
        'n_eligible': len(eligible),
        'mean_J_R': float(np.mean([r['J_R'] for r in eligible])) if eligible else None,
        'mean_J_S': float(np.mean([r['J_S'] for r in eligible])) if eligible else None,
        'mean_delta': float(np.mean(deltas)) if deltas else None,
        'ci': list(paired_bootstrap_ci(deltas)) if len(deltas) >= 2 else None,
        'n_swap_better': n_better, 'n_swap_worse': n_worse, 'n_tie': n_tie,
        'component_deltas': comp,
        'mean_n_swap': float(np.mean([r['n_swap_S'] for r in eligible])) if eligible else None,
        'mean_n_attempts_R': float(np.mean([r['n_attempts_R'] for r in eligible])) if eligible else None,
        'mean_n_attempts_S': float(np.mean([r['n_attempts_S'] for r in eligible])) if eligible else None,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, 'per_state.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['inst', 'event', 'J_R', 'J_S', 'delta', 'D_R', 'D_S', 'Q_R', 'Q_S',
                    'E_R', 'E_S', 'n_swap', 'n_attempts_R', 'n_attempts_S', 'P0_feasible',
                    'pool_nonempty'])
        for r in rows:
            w.writerow([r['inst'], r['event'], f"{r['J_R']:.6f}", f"{r['J_S']:.6f}",
                        f"{r['delta']:+.6f}", f"{r['D_R']:.4f}", f"{r['D_S']:.4f}",
                        f"{r['Q_R']:.4f}", f"{r['Q_S']:.4f}", f"{r['E_R']:.4f}",
                        f"{r['E_S']:.4f}", r['n_swap_S'], r['n_attempts_R'],
                        r['n_attempts_S'], int(r['P0_feasible']), int(r['pool_nonempty'])])
    print(f"\n=== swap fixed-state (budget={args.budget}s, n_rounds={args.n_rounds}, "
          f"max_attempts={args.max_attempts}, n_states={len(rows)}) ===")
    print(f"  eligible={len(eligible)}  mean delta(J_S-J_R)={summary['mean_delta']:+.6f} "
          f"CI={[round(x,6) for x in (summary['ci'] or [])]}")
    print(f"  swap_better={n_better}  swap_worse={n_worse}  tie={n_tie}")
    print(f"  D/Q/E delta: D={comp['D']['mean_delta']:+.4f} Q={comp['Q']['mean_delta']:+.4f} "
          f"E={comp['E']['mean_delta']:+.4f}")
    print(f"  mean n_swap={summary['mean_n_swap']:.1f}  attempts R/S="
          f"{summary['mean_n_attempts_R']:.1f}/{summary['mean_n_attempts_S']:.1f}")
    print(f"saved: {args.out}/summary.json + per_state.json + per_state.csv")


if __name__ == '__main__':
    main()
