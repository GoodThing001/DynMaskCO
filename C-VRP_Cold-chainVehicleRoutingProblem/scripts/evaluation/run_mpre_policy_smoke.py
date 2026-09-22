"""M-pre / M-trained 统一逐步策略的真实状态端到端验收（含反例与隔离）。

对真实决策点选 mask、移除得部分计划，用统一逐步策略分别跑 M-pre 与 M-trained（零残差），验收：
  1. 继承：逐步动作完整记录（customer/vehicle_id/position/pred/succ）+ log-prob 一致。
  2. 计划对账：validate_repair_plan 全查（exact-once/逐车有序/非 mask 归属/保护车/车辆集合）。
  3. 隔离：stepwise 在副本上操作，输入 plans 快照前后一致。
  4. 反例：损坏计划（漏客户/改保护车/改非 mask 顺序）被 validate_repair_plan 拒绝。
  5. 非空：至少覆盖规定事件数，零事件判失败。
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
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
from action_contract import build_vehicle_plans
from cc_lns_replanner import _select_one_mask
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from repair_state import extract_repair_state_v1, F_NODE, F_ACTION_EXPLICIT
from mpre_policy import (stepwise_sample, replay_log_prob, validate_repair_plan,
                         enumerate_legal_actions)
from mpre import load_cvrp_model
from mpre_trained import load_mpre_trained, edge_insertion_scores
from mpre_replanner import retained_edge_ratio
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
    def __init__(self, contract, capacity, backbone, model, max_events, tw_max):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.capacity = capacity
        self.backbone = backbone
        self.model = model
        self.max_events = int(max_events)
        self.tw_max = tw_max
        self.results = []
        self._done = 0

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        if self._done >= self.max_events:
            super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                         replan_ids=replan_ids)
            return
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
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = [int(c) for c in visible_ids if not served_mask[int(c)]]
        mask = _select_one_mask(env, inst_idx, P0, pool, 2, seed=0)
        if not mask:
            return
        mask_set = set(mask)
        partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                tuple(x for x in p.suffix if x not in mask_set))
                   for vid, p in P0.items()}
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in visible_ids if int(i) != 0])

        def make_fns(model_):
            def extract_fn(plans, remaining, legal):
                st = extract_repair_state_v1(env, inst_idx, clock, vehicles, served_mask,
                                             visible_ids, plans, remaining, legal,
                                             self.capacity, self.tw_max,
                                             allowed_vehicle_ids=mutable_ids,
                                             node_to_local=node_to_local,
                                             local_to_node=local_to_node, Nv=Nv)
                A0 = build_plan_adjacency(P0, node_to_local, Nv)
                st['timestep'] = retained_edge_ratio(A0, st['adjmat'][0])
                return st

            def score_fn(st):
                s, _ = model_.score_actions(jnp.array(st['raw_3d']), jnp.array(st['node_valid']),
                                            jnp.array(st['node_feats']),
                                            jnp.array([st['timestep']], jnp.float32),
                                            jnp.array(st['adjmat'][0]), jnp.array(st['cust']),
                                            jnp.array(st['pred']), jnp.array(st['succ']),
                                            jnp.array(st['action_feats']))
                return s[0]
            return extract_fn, score_fn

        # M-pre scorer（backbone 边 logits，无 adapter）
        def mpre_extract(plans, remaining, legal):
            st = make_fns(self.model)[0](plans, remaining, legal)
            return st

        def mpre_score(st):
            H = self.backbone.encode(jnp.array(st['raw_3d']), attn_options={})
            L = self.backbone.decode(H, jnp.array([st['timestep']], jnp.float32),
                                     jnp.array(st['adjmat'][0]), target='logit')
            s = edge_insertion_scores(L, jnp.array(st['cust']), jnp.array(st['pred']),
                                      jnp.array(st['succ']))
            return s[0]

        # 输入快照（隔离检查）
        partial_snapshot = {vid: tuple(p.suffix) for vid, p in partial.items()}

        rng = np.random.default_rng(0)
        P_mpre, steps_mpre, recs_mpre, fail_mpre = stepwise_sample(
            mpre_score, mpre_extract, env, inst_idx, partial, mask_set, mutable_ids,
            temperature=1.0, rng=rng)
        extract_mt, score_mt = make_fns(self.model)
        rng2 = np.random.default_rng(0)
        P_mtrained, steps_mtrained, recs_mtrained, fail_mtrained = stepwise_sample(
            score_mt, extract_mt, env, inst_idx, partial, mask_set, mutable_ids,
            temperature=1.0, rng=rng2)

        # 隔离：partial 未被修改
        partial_after = {vid: tuple(p.suffix) for vid, p in partial.items()}
        isolated = partial_after == partial_snapshot

        # 完整动作记录一致
        def sig(s):
            return [(x['customer'], x['vehicle_id'], x['slot_kind'], x['position'],
                     x['predecessor'], x['successor']) for x in s]
        seq_match = sig(steps_mpre) == sig(steps_mtrained)

        # 计划对账 + 反例
        ok_recon, detail = validate_repair_plan(P_mtrained, P0, mask_set, protected)
        # 反例：漏一个客户
        corr_missing = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                     tuple(x for x in p.suffix if x != next(iter(mask_set))))
                        for vid, p in P_mtrained.items()}
        bad_missing, _ = validate_repair_plan(corr_missing, P0, mask_set, protected)
        # 反例：改非 mask 顺序（每车交换前两个非 mask 客户）
        def _corrupt_order(plans):
            out = {}
            for vid, p in plans.items():
                suf = [int(x) for x in p.suffix if int(x) > 0]
                non_mask = [x for x in suf if x not in mask_set]
                if len(non_mask) >= 2:
                    a, b = non_mask[0], non_mask[1]
                    new_suf = [b if x == a else (a if x == b else x) for x in suf]
                    out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                       tuple(new_suf))
                else:
                    out[vid] = p
            return out
        bad_order, _ = validate_repair_plan(_corrupt_order(P_mtrained), P0, mask_set, protected)
        # 反例：改保护车（给保护车塞一个 mask 客户，应被拒绝）
        if protected:
            vid0 = next(iter(protected))
            c0 = next(iter(mask_set))
            corr_prot = {}
            for vid, p in P_mtrained.items():
                if vid == vid0:
                    corr_prot[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time,
                                             p.anchor_load, tuple(list(p.suffix) + [c0]))
                else:
                    corr_prot[vid] = p
            bad_prot, _ = validate_repair_plan(corr_prot, P0, mask_set, protected)
        else:
            bad_prot = False  # 无保护车，跳过

        # log-prob（回放）
        lp_mpre = float(replay_log_prob(lambda st: jnp.array(mpre_score(st)), recs_mpre))
        lp_mtrained = float(replay_log_prob(score_mt, recs_mtrained))

        self.results.append({
            'inst': int(inst_idx), 'event': int(getattr(env, 'event_id', -1)),
            'n_mask': len(mask_set), 'n_steps': len(steps_mtrained),
            'seq_match': bool(seq_match),
            'lp_diff': float(abs(lp_mpre - lp_mtrained)),
            'recon_ok': bool(ok_recon), 'isolated': bool(isolated),
            'bad_missing_rejected': bool(not bad_missing),
            'bad_order_rejected': bool(not bad_order),
            'bad_protected_rejected': bool(not bad_prot),
            'fail_mpre': fail_mpre, 'fail_mtrained': fail_mtrained,
        })
        self._done += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-events', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile)
    tw_max = float(dataset['tw_end'][:, 0].max())
    backbone, cfg, _ = load_cvrp_model(args.cvrp_ckpt)
    model, _, _ = load_mpre_trained(args.cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, seed=0)

    rep = Probe(eff, args.capacity, backbone, model, args.max_events, tw_max)
    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                          replanner=rep, coldchain_contract=eff)
    env.run(0)
    rows = rep.results

    nonempty = len(rows) >= args.max_events
    seq = all(r['seq_match'] for r in rows) and nonempty
    recon = all(r['recon_ok'] for r in rows) and nonempty
    isolated = all(r['isolated'] for r in rows) and nonempty
    counterexamples = all(r['bad_missing_rejected'] and r['bad_order_rejected']
                          and r['bad_protected_rejected']
                          and r['fail_mpre'] is None and r['fail_mtrained'] is None for r in rows)
    max_lp_diff = max(r['lp_diff'] for r in rows) if rows else 1e9
    report = {
        'n_events': len(rows), 'nonempty': bool(nonempty),
        'seq_match': bool(seq), 'recon_ok': bool(recon), 'isolated': bool(isolated),
        'counterexamples_ok': bool(counterexamples), 'max_lp_diff': float(max_lp_diff),
        'rows': rows,
        'ALL_PASS': bool(nonempty and seq and recon and isolated and counterexamples
                         and max_lp_diff < 1e-5),
    }
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('=== M-pre/M-trained stepwise policy smoke ===')
    for k in ('n_events', 'nonempty', 'seq_match', 'recon_ok', 'isolated',
              'counterexamples_ok', 'max_lp_diff', 'ALL_PASS'):
        print(f'  {k} = {report[k]}')
    sys.exit(0 if report['ALL_PASS'] else 1)


if __name__ == '__main__':
    main()
