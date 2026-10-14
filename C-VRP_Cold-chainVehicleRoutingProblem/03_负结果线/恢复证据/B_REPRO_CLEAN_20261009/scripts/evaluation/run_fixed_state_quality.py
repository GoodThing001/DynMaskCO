"""固定状态质量表：R / M-pre / M-trained 在同一公开状态上的完整修复质量。

对每个决策点（TRAIN / CAL 各 16 实例 × 每实例 2 个事件，固定事件顺序，共 64 状态），固定
mask=2，比较：
  - R: regret-2 重建（确定性，非学习参照）；
  - M-pre: 预训练 CVRP 统一策略（确定性 argmax 1 次 + 采样分布 8 次）；
  - M-trained: 纠正训练后统一策略（确定性 argmax 1 次 + 采样分布 8 次）。

每状态 19 次完整修复尝试（1 + 9 + 9），64 状态共 1216 次（失败也计入）。
输出：完整 J_vis、D/Q/E、失败率、方案多样性、耗时。按实例聚合统计，不把多事件当独立样本。

用法（服务器）：
    python scripts/evaluation/run_fixed_state_quality.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
        --model-ckpt results/m0_scale/mpre_reinforce_s42_corrected/model.ckpt \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --out results/m0_scale/fsq_train
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import jax.numpy as jnp

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
from action_contract import (build_vehicle_plans, plan_hash, enumerate_actions_from_plans,
                             apply_action)
from cc_lns_replanner import _select_one_mask, _plan_suffixes, _reconstruct
from dynmaskco_cc_context import (decision_pool_from_vehicles, mask_candidate_pool,
                                  validate_mask_scope)
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from repair_state import extract_repair_state_v1, freeze_vehicles
from mpre_policy import stepwise_sample, enumerate_legal_actions, validate_repair_plan
from mpre_replanner import retained_edge_ratio
from mpre import load_cvrp_model
from mtrained_replanner import (load_trained_model, mpre_score_fn, mtrained_score_fn,
                                _make_extract)
from visible_state import build_visible_state, evaluate_visible_plan
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)


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
    """采集决策点状态（冻结 + 每实例 max_per_instance 个，固定事件顺序）。"""

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
                            'vehicles': freeze_vehicles(vehicles),
                            'served_mask': np.array(served_mask, copy=True),
                            'visible_ids': list(visible_ids),
                            'replan_ids': list(replan_ids) if replan_ids is not None else None,
                            'P0': P0, 'partial': partial, 'mask_set': mask_set,
                            'mutable_ids': mutable_ids, 'protected': protected,
                            'vis': vis, 'J0': float(r0.J_vis)})
        self.per_instance[int(inst_idx)] = self.per_instance.get(int(inst_idx), 0) + 1


def _eval_plan(vis, plan, contract, objective):
    """评估完整计划 J_vis。失败返回 None。"""
    r = evaluate_visible_plan(vis, _plan_suffixes(plan), contract, objective)
    if not r.finite or not r.feasible:
        return None
    return r


def _reconstruct_diag(env, inst_idx, partial, mask_customers, mutable_ids):
    """regret-2 重建的诊断镜像：与 _reconstruct 决策一致，但记录每步剩余/可行动作数。

    不改算法——只在 _reconstruct 返回 None 时定位失败步与无可行插入的客户。
    返回 (P_or_None, diag)。diag.trace[i] = 第 i 步的剩余客户与各客户可行动作数。
    """
    P = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
         for vid, p in partial.items()}
    remaining = list(mask_customers)
    step = 0
    trace = []
    while remaining:
        best_regret = -float('inf')
        best_c = None
        best_action = None
        feas_counts = {}
        for c in remaining:
            cands, _ = enumerate_actions_from_plans(env, inst_idx, P, c,
                                                    allowed_vehicle_ids=mutable_ids)
            feas = [x for x in cands if x.feasible]
            feas_counts[int(c)] = len(feas)
            if not feas:
                continue
            feas.sort(key=lambda x: (x.incremental_distance, x.action.action_id()))
            best_cost = feas[0].incremental_distance
            second_cost = (feas[1].incremental_distance if len(feas) > 1 else best_cost + 1e6)
            regret = second_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_c = c
                best_action = feas[0].action
        trace.append({'step': step, 'remaining': [int(c) for c in remaining],
                      'feasible_count': feas_counts})
        if best_c is None:
            return None, {'failed_step': step, 'remaining_at_fail': [int(c) for c in remaining],
                          'feasible_count_at_fail': feas_counts, 'trace': trace}
        P = apply_action(P, best_action, allowed_vehicle_ids=mutable_ids)
        remaining.remove(best_c)
        step += 1
    return P, {'failed_step': None, 'remaining_at_fail': [],
               'feasible_count_at_fail': {}, 'trace': trace}


def _audit_mask_scope(vehicles, served_mask, visible_ids, P0, mask_set, mutable_ids, protected):
    """审计 mask 客户是否越界（相对 decision_pool_from_vehicles 口径）。

    对每个 mask 客户记录：是否在 decision_pool / 是否 committed_next / 是否在非可变车 tail /
    是否在 protected 车 tail / 是否位于可变车计划。返回汇总越界计数。
    """
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    committed_next = {int(v.committed_next) for v in vehicles
                      if v.status == 'committed' and v.committed_next not in (None, 0)}
    non_mutable_tail = set()
    for v in vehicles:
        if v.vehicle_id not in replan_ids:
            for n in v.mutable_suffix:
                if int(n) != 0:
                    non_mutable_tail.add(int(n))
    decision_pool = set(decision_pool_from_vehicles(vehicles, served_mask, visible_ids))
    protected_tail = set()
    for vid in protected:
        p = P0.get(vid)
        if p is not None:
            for x in p.suffix:
                if int(x) > 0:
                    protected_tail.add(int(x))
    mutable_plan_customers = set()
    for vid, p in P0.items():
        if vid in mutable_ids:
            for x in p.suffix:
                if int(x) > 0:
                    mutable_plan_customers.add(int(x))
    per_cust = {}
    n_out_of_pool = 0
    for c in mask_set:
        c = int(c)
        info = {
            'in_decision_pool': c in decision_pool,
            'is_committed_next': c in committed_next,
            'in_non_mutable_tail': c in non_mutable_tail,
            'in_protected_tail': c in protected_tail,
            'in_mutable_vehicle': c in mutable_plan_customers,
        }
        if not info['in_decision_pool']:
            n_out_of_pool += 1
        per_cust[c] = info
    return {'per_customer': per_cust, 'n_out_of_pool': n_out_of_pool,
            'decision_pool_size': len(decision_pool), 'mask_size': len(mask_set)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--n-samples', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = default_pilot_contract()
    eff = apply_objective_profile(contract, profile)
    tw_max = float(dataset['tw_end'][:, 0].max())
    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)
    mpre_score = mpre_score_fn(backbone)
    mtrained_score = mtrained_score_fn(model)
    objective = eff.objective

    probe = Probe(eff, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=eff)
        env.run(inst)
    states = probe.states
    print(f'collected {len(states)} states over {args.max_instances} instances '
          f'({args.split})', flush=True)

    rows = []
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in st['visible_ids'] if int(i) != 0])
        extract_fn = _make_extract(env, inst_idx, st['clock'], st['vehicles'], st['served_mask'],
                                   st['visible_ids'], args.capacity, tw_max,
                                   node_to_local, local_to_node, Nv, st['P0'], st['mutable_ids'])
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']
        P0 = st['P0']
        protected = st['protected']

        def _run_method(score_fn, deterministic, seed=0):
            t0 = time.perf_counter()
            fail = None
            diag = None
            if score_fn is None:  # R: regret-2 重建（确定性）
                P = _reconstruct(env, inst_idx, partial, mask_set, mutable_ids)
                if P is None:
                    fail = 'no_reconstruct'
                    _, diag = _reconstruct_diag(env, inst_idx, partial, mask_set, mutable_ids)
            else:
                rng = np.random.default_rng(seed)
                P, _steps, _rec, fail = stepwise_sample(
                    score_fn, extract_fn, env, inst_idx, partial, mask_set, mutable_ids,
                    deterministic=deterministic, rng=rng)
            dt = time.perf_counter() - t0
            cert_ok = None
            cert = None
            if fail is None and P is not None:
                cert_ok, cert = validate_repair_plan(P, P0, mask_set, protected)
            if fail is not None or P is None:
                return {'J': None, 'D': None, 'Q': None, 'E': None, 'fail': True,
                        'fail_reason': fail if fail is not None else 'P_is_None',
                        'time': dt, 'hash': None, 'cert_ok': cert_ok, 'cert': cert,
                        'diag': diag}
            r = _eval_plan(vis, P, eff, objective)
            if r is None:
                return {'J': None, 'D': None, 'Q': None, 'E': None, 'fail': True,
                        'fail_reason': 'eval_infeasible', 'time': dt, 'hash': None,
                        'cert_ok': cert_ok, 'cert': cert, 'diag': diag}
            return {'J': float(r.J_vis), 'D': float(r.D), 'Q': float(r.Q), 'E': float(r.E),
                    'fail': False, 'fail_reason': None, 'time': dt, 'hash': plan_hash(P),
                    'cert_ok': cert_ok, 'cert': cert, 'diag': diag}

        row = {'inst': inst_idx, 'event': st['event'], 'J0': st['J0']}
        row['mask_audit'] = _audit_mask_scope(st['vehicles'], st['served_mask'],
                                              st['visible_ids'], P0, mask_set, mutable_ids,
                                              protected)
        # R（regret-2 确定性）
        r_r = _run_method(None, True)
        row['R'] = {'det': r_r, 'samples': []}
        # M-pre / M-trained：确定性 argmax 1 次 + 采样分布 n_samples 次
        for name, score_fn in (('Mpre', mpre_score), ('Mtrained', mtrained_score)):
            det = _run_method(score_fn, True)
            samples = [_run_method(score_fn, False, seed=i) for i in range(args.n_samples)]
            row[name] = {'det': det, 'samples': samples}
        rows.append(row)
        r_j = row['R']['det']['J']
        r_disp = f"{r_j:.4f}" if r_j is not None else f"FAIL({row['R']['det']['fail_reason']})"
        print(f"  [inst {inst_idx}/evt {st['event']}] J0={st['J0']:.4f} R={r_disp} "
              f"audit_out={row['mask_audit']['n_out_of_pool']} "
              f"Mpre_det={row['Mpre']['det']['J'] if row['Mpre']['det']['J'] is not None else 'FAIL'} "
              f"Mtrained_det={row['Mtrained']['det']['J'] if row['Mtrained']['det']['J'] is not None else 'FAIL'}",
              flush=True)

    # 聚合（按实例聚合，不把多事件当独立样本）
    def _deployed_agg(method, key='det'):
        """部署语义：失败状态用 J0 代替（保留 P0），不静默删除。"""
        per_inst = {}
        n_fail = 0
        n_total = 0
        for r in rows:
            v = r[method][key]
            n_total += 1
            if v['fail']:
                n_fail += 1
            j = v['J'] if not v['fail'] else r['J0']
            per_inst.setdefault(r['inst'], []).append(j)
        inst_mean = [np.mean(v) for v in per_inst.values()] if per_inst else []
        return {'n_states': n_total, 'n_fail': n_fail,
                'fail_rate': float(n_fail / max(n_total, 1)),
                'mean_J': float(np.mean(inst_mean)) if inst_mean else None,
                'inst_std': float(np.std(inst_mean)) if inst_mean else None}

    def _samples_agg(method):
        all_j = []
        per_inst = {}
        n_fail = 0
        n_total = 0
        hashes = set()
        for r in rows:
            for s in r[method]['samples']:
                n_total += 1
                if s['fail']:
                    n_fail += 1
                    continue
                all_j.append(s['J'])
                per_inst.setdefault(r['inst'], []).append(s['J'])
                if s['hash'] is not None:
                    hashes.add(s['hash'])
        inst_mean = [np.mean(v) for v in per_inst.values()] if per_inst else []
        return {'mean_J': float(np.mean(inst_mean)) if inst_mean else None,
                'inst_std': float(np.std(inst_mean)) if inst_mean else None,
                'fail_rate': float(n_fail / max(n_total, 1)),
                'n_unique_plans': len(hashes)}

    def _mask_audit_agg():
        n_states_out = 0
        n_cust_out = 0
        reasons = Counter()
        for r in rows:
            a = r['mask_audit']
            if a['n_out_of_pool'] > 0:
                n_states_out += 1
                n_cust_out += a['n_out_of_pool']
            for _c, info in a['per_customer'].items():
                if not info['in_decision_pool']:
                    reasons['out_of_decision_pool'] += 1
                if info['is_committed_next']:
                    reasons['is_committed_next'] += 1
                if info['in_non_mutable_tail']:
                    reasons['in_non_mutable_tail'] += 1
                if info['in_protected_tail']:
                    reasons['in_protected_tail'] += 1
                if not info['in_mutable_vehicle']:
                    reasons['not_in_mutable_vehicle'] += 1
        return {'n_states_with_out_of_pool': n_states_out, 'n_out_of_pool_customers': n_cust_out,
                'reason_counts': dict(reasons)}

    summary = {
        'split': args.split, 'n_instances': args.max_instances, 'n_states': len(rows),
        'R': _deployed_agg('R'), 'Mpre_det': _deployed_agg('Mpre'),
        'Mtrained_det': _deployed_agg('Mtrained'),
        'Mpre_samples': _samples_agg('Mpre'), 'Mtrained_samples': _samples_agg('Mtrained'),
        'mask_audit': _mask_audit_agg(),
    }
    # 相对 R 的 delta（部署语义）
    if summary['R']['mean_J'] is not None:
        summary['Mpre_det_minus_R'] = (summary['Mpre_det']['mean_J'] - summary['R']['mean_J'])
        summary['Mtrained_det_minus_R'] = (summary['Mtrained_det']['mean_J']
                                           - summary['R']['mean_J'])
        summary['Mtrained_det_minus_Mpre'] = (summary['Mtrained_det']['mean_J']
                                              - summary['Mpre_det']['mean_J'])
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "None"

    print(f"\n=== fixed-state quality ({args.split}, n={len(rows)}) ===")
    print(f"  R            deployed mean J_vis = {_fmt(summary['R']['mean_J'])} "
          f"(raw_fail={summary['R']['n_fail']}/{summary['R']['n_states']})")
    print(f"  Mpre_det     deployed mean J_vis = {_fmt(summary['Mpre_det']['mean_J'])}  "
          f"ΔvsR={summary.get('Mpre_det_minus_R', float('nan')):+.4f} "
          f"(raw_fail={summary['Mpre_det']['n_fail']})")
    print(f"  Mtrained_det deployed mean J_vis = {_fmt(summary['Mtrained_det']['mean_J'])}  "
          f"ΔvsR={summary.get('Mtrained_det_minus_R', float('nan')):+.4f}  "
          f"ΔvsMpre={summary.get('Mtrained_det_minus_Mpre', float('nan')):+.4f} "
          f"(raw_fail={summary['Mtrained_det']['n_fail']})")
    print(f"  Mpre_samples     mean={_fmt(summary['Mpre_samples']['mean_J'])} "
          f"fail={summary['Mpre_samples']['fail_rate']:.3f} uniq={summary['Mpre_samples']['n_unique_plans']}")
    print(f"  Mtrained_samples mean={_fmt(summary['Mtrained_samples']['mean_J'])} "
          f"fail={summary['Mtrained_samples']['fail_rate']:.3f} uniq={summary['Mtrained_samples']['n_unique_plans']}")
    print(f"  mask_audit: states_out_of_pool={summary['mask_audit']['n_states_with_out_of_pool']} "
          f"cust_out={summary['mask_audit']['n_out_of_pool_customers']} "
          f"reasons={summary['mask_audit']['reason_counts']}")
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
