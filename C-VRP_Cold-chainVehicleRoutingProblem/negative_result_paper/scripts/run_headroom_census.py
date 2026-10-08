"""D1+D2 头腔普查（myopic，无训练/无模型/纯 NumPy）。

复用 Probe（与 run_sgbs_fixed_state.py 相同，mask=2、每实例前 2 事件、同 seed）采集决策点
状态，对每个状态做两件事：

D1 穷举修复 oracle：
  - P_dist_greedy：距离贪心（逐步 min incremental_distance）修复；
  - P_exh：穷举所有合法两客户重插（两种插入顺序，plan_hash 去重），取 myopic J_vis 最小；
  - gap_exh = J_vis(P_dist_greedy) − J_vis(P_exh)，并与已有 dist_sgbs 对比。

D2 单步修正普查（残差真正要学的量的逐状态上限）：
  - 第一步 = 距离贪心第 1 个插入（min incremental_distance）；
  - 对每个非距离第 1 步动作 a'，apply 后用距离贪心补全剩余，取 J_vis；
  - correction_signal(s) = J_vis(P_dist_greedy) − min_{a'} J_vis(P(a' + 贪心补全))；
  - 报告 correction_signal 分布（mean/median/p90/max、>阈值状态比例、实例聚合）。

目的：在花 G/GV/L 的 96k 次评价预算前，测出「局部插入层残差」的天花板。若 gap_exh 与
correction_signal 都 ≈0，则该层无可学修正，pilot 结构上判死。

用法（服务器，与 sgbs_fixed 同数据同 profile）：
    python scripts/evaluation/run_headroom_census.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --out results/m0_scale/headroom_census_train \
        --ref-per-state results/m0_scale/sgbs_fixed_train_v2/per_state.json
"""
import argparse
import copy
import json
import os
import sys
from collections import Counter

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
from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, plan_hash, enumerate_actions_from_plans,
                             apply_action)
from cc_lns_replanner import _select_one_mask, _plan_suffixes
from dynmaskco_cc_context import mask_candidate_pool, validate_mask_scope
from mpre_policy import enumerate_legal_actions, validate_repair_plan
from visible_state import build_visible_state, evaluate_visible_plan
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile)


def _load_profile(path):
    with open(path) as f:
        d = json.load(f)
    if d.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                            quality_scale=float(d['quality_scale']),
                            energy_scale=float(d['energy_scale']),
                            lambda_quality=float(d['lambda_quality']),
                            lambda_energy=float(d['lambda_energy']),
                            scale_source=d.get('scale_source', 'pilot'),
                            dev_statistics=d.get('dev_statistics'))


class Probe(JF1HRepairReplanner):
    """与 run_fixed_state_quality.Probe 相同的决策点状态采集（mask=2、每实例前 2 事件）。"""

    def __init__(self, contract, capacity, max_per_instance=2):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.capacity = capacity
        self.max_per_instance = int(max_per_instance)
        self.per_instance = {}
        self.states = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved_ids = set()
        idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
        reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        if self.per_instance.get(int(inst_idx), 0) >= self.max_per_instance:
            return
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in reserved_ids:
            v = vehicles[vid]
            p = P0.get(vid)
            if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                protected.add(vid)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = mask_candidate_pool(vehicles, served_mask, visible_ids,
                                   protected=protected, plans=P0)
        mask = _select_one_mask(env, inst_idx, P0, pool, 2,
                                seed=self.per_instance.get(int(inst_idx), 0))
        if not mask:
            return
        validate_mask_scope(vehicles, served_mask, visible_ids, P0, mask, mutable_ids,
                            protected=protected)
        mask_set = set(mask)
        partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                tuple(x for x in p.suffix if x not in mask_set))
                   for vid, p in P0.items()}
        vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                                  served_mask, visible_ids, replan_ids, self.deferred_customers)
        r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), self.contract, self.contract.objective)
        self.states.append({'inst': int(inst_idx), 'clock': float(clock), 'env': env,
                            'event': int(getattr(env, 'event_id', -1)),
                            'vehicles': copy.deepcopy(vehicles),
                            'served_mask': np.array(served_mask, copy=True),
                            'visible_ids': list(visible_ids),
                            'P0': P0, 'partial': partial, 'mask_set': mask_set,
                            'mutable_ids': mutable_ids, 'protected': protected,
                            'vis': vis, 'J0': float(r0.J_vis)})
        self.per_instance[int(inst_idx)] = self.per_instance.get(int(inst_idx), 0) + 1


def _plan_copy(plans):
    return {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
            for vid, p in plans.items()}


def _eval_plan(vis, plan, contract, objective):
    r = evaluate_visible_plan(vis, _plan_suffixes(plan), contract, objective)
    if not r.finite or not r.feasible:
        return None
    return r


def _dist_greedy_repair(env, inst_idx, partial, mask_set, mutable_ids):
    P = _plan_copy(partial)
    remaining = list(mask_set)
    while remaining:
        legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
        if not legal:
            return None
        legal.sort(key=lambda t: (t[2].incremental_distance, t[1].action_id()))
        c, a, _cand = legal[0]
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        remaining.remove(c)
    return P


def _exhaustive_repair(env, inst_idx, partial, mask_set, mutable_ids, vis, contract, objective):
    best_J = None
    best_P = None
    seen = set()
    n_eval = 0

    def rec(P, rem):
        nonlocal best_J, best_P, n_eval
        if not rem:
            h = plan_hash(P)
            if h in seen:
                return
            seen.add(h)
            r = _eval_plan(vis, P, contract, objective)
            n_eval += 1
            if r is not None and (best_J is None or r.J_vis < best_J):
                best_J = r.J_vis
                best_P = P
            return
        legal = enumerate_legal_actions(env, inst_idx, P, rem, mutable_ids)
        for c, a, _cand in legal:
            P2 = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
            rec(P2, [x for x in rem if x != c])

    rec(_plan_copy(partial), list(mask_set))
    return best_P, best_J, {'n_unique': len(seen), 'n_eval': n_eval}


def _complete_greedy(env, inst_idx, P, remaining, mutable_ids):
    P = _plan_copy(P)
    rem = list(remaining)
    while rem:
        legal = enumerate_legal_actions(env, inst_idx, P, rem, mutable_ids)
        if not legal:
            return None
        legal.sort(key=lambda t: (t[2].incremental_distance, t[1].action_id()))
        c, a, _cand = legal[0]
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        rem.remove(c)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--ref-per-state', default=None,
                    help='已有 sgbs_fixed per_state.json，用于合并 dist_sgbs 对比')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective

    probe = Probe(contract, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.run(inst)
    states = probe.states
    print(f'collected {len(states)} states over {args.max_instances} instances '
          f'({args.split})', flush=True)

    # 合并已有 dist_sgbs（按 (inst, event) 对齐）
    ref = {}
    if args.ref_per_state:
        with open(args.ref_per_state) as f:
            for r in json.load(f):
                ref[(int(r['inst']), int(r['event']))] = r['dist_sgbs']

    rows = []
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']

        P_dg = _dist_greedy_repair(env, inst_idx, partial, mask_set, mutable_ids)
        r_dg = _eval_plan(vis, P_dg, contract, objective) if P_dg is not None else None
        J_dg = float(r_dg.J_vis) if r_dg is not None else None

        P_exh, J_exh, exh_stats = _exhaustive_repair(env, inst_idx, partial, mask_set,
                                                     mutable_ids, vis, contract, objective)

        # D2：单步修正普查
        base = _plan_copy(partial)
        legal = enumerate_legal_actions(env, inst_idx, base, mask_set, mutable_ids)
        legal.sort(key=lambda t: (t[2].incremental_distance, t[1].action_id()))
        best_dev_J = J_dg
        n_dev = 0
        n_dev_improved = 0
        for c, a, _cand in legal[1:]:
            P1 = apply_action(base, a, allowed_vehicle_ids=mutable_ids)
            rem = [x for x in mask_set if x != c]
            Pfull = _complete_greedy(env, inst_idx, P1, rem, mutable_ids)
            if Pfull is None:
                continue
            r = _eval_plan(vis, Pfull, contract, objective)
            if r is None:
                continue
            n_dev += 1
            if r.J_vis < best_dev_J - 1e-9:
                n_dev_improved += 1
                best_dev_J = float(r.J_vis)
        correction_signal = (J_dg - best_dev_J) if (J_dg is not None and best_dev_J is not None) else None

        key = (inst_idx, st['event'])
        row = {
            'inst': inst_idx, 'event': st['event'], 'J0': st['J0'],
            'J_dist_greedy': J_dg, 'J_exhaustive': J_exh,
            'gap_exh_vs_greedy': (J_dg - J_exh) if (J_dg is not None and J_exh is not None) else None,
            'dist_sgbs_ref': ref.get(key),
            'gap_exh_vs_dist_sgbs': (ref[key] - J_exh) if (key in ref and J_exh is not None) else None,
            'correction_signal': correction_signal,
            'n_first_step_actions': len(legal),
            'n_dev_tried': n_dev, 'n_dev_improved': n_dev_improved,
            'exhaustive_stats': exh_stats,
        }
        rows.append(row)
        print(f"  [inst {inst_idx}/evt {st['event']}] J0={st['J0']:.4f} "
              f"greedy={J_dg if J_dg is None else round(J_dg,4)} "
              f"exh={J_exh if J_exh is None else round(J_exh,4)} "
              f"gap_exh={row['gap_exh_vs_greedy'] if row['gap_exh_vs_greedy'] is None else round(row['gap_exh_vs_greedy'],5)} "
              f"corr_sig={correction_signal if correction_signal is None else round(correction_signal,5)} "
              f"n_dev={n_dev}/{len(legal)-1} imp={n_dev_improved}", flush=True)

    def _inst_agg(key):
        per = {}
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            per.setdefault(r['inst'], []).append(v)
        return [float(np.mean(v)) for v in per.values()]

    def _clustered_mean_ci(vals, n_boot=2000, seed=0):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(seed)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(n_boot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}

    gap_exh = [r['gap_exh_vs_greedy'] for r in rows if r['gap_exh_vs_greedy'] is not None]
    corr = [r['correction_signal'] for r in rows if r['correction_signal'] is not None]
    gap_exh_vs_sgbs = [r['gap_exh_vs_dist_sgbs'] for r in rows if r['gap_exh_vs_dist_sgbs'] is not None]

    summary = {
        'split': args.split, 'n_states': len(rows), 'n_instances': len(set(r['inst'] for r in rows)),
        'D1': {
            'gap_exh_vs_greedy': _clustered_mean_ci(gap_exh),
            'gap_exh_vs_dist_sgbs': _clustered_mean_ci(gap_exh_vs_sgbs),
            'per_state_quantiles': {
                'p50': float(np.percentile(gap_exh, 50)) if gap_exh else None,
                'p90': float(np.percentile(gap_exh, 90)) if gap_exh else None,
                'max': float(np.max(gap_exh)) if gap_exh else None,
            },
            'n_states_gap_gt_0p01': int(sum(1 for x in gap_exh if x > 0.01)),
            'n_states_gap_gt_0p005': int(sum(1 for x in gap_exh if x > 0.005)),
        },
        'D2': {
            'correction_signal_mean_ci': _clustered_mean_ci(corr),
            'per_state_quantiles': {
                'p25': float(np.percentile(corr, 25)) if corr else None,
                'p50': float(np.percentile(corr, 50)) if corr else None,
                'p90': float(np.percentile(corr, 90)) if corr else None,
                'max': float(np.max(corr)) if corr else None,
            },
            'n_states_signal_gt_0': int(sum(1 for x in corr if x > 1e-9)),
            'n_states_signal_gt_0p01': int(sum(1 for x in corr if x > 0.01)),
            'n_states_signal_gt_0p005': int(sum(1 for x in corr if x > 0.005)),
            'n_dev_tried_total': int(sum(r['n_dev_tried'] for r in rows)),
            'n_dev_improved_total': int(sum(r['n_dev_improved'] for r in rows)),
        },
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n=== D1 exhaustive vs greedy ===")
    print(json.dumps(summary['D1'], indent=2))
    print("=== D2 correction census ===")
    print(json.dumps(summary['D2'], indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
