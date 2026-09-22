"""M-trained / M-pre 统一策略在线 replanner + 四方法 gate 用的搜索。

M-pre = 预训练 CVRP backbone 的边 logits（无 adapter）；M-trained = 训练后全模型（+冷链 adapter + delta head）。
二者共用统一逐步策略（stepwise_sample deterministic=argmax）、相同动作枚举、认证与 J_vis 接受。
"""
from __future__ import annotations

import pickle
import time
from collections import defaultdict

import numpy as np
import jax.numpy as jnp

from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans, plan_hash
from dynmaskco_cc_context import mask_candidate_pool, validate_mask_scope
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from visible_state import build_visible_state, evaluate_visible_plan
from cc_lns_replanner import (REQUEST_SEQUENCE, N_ROUNDS, _select_one_mask, _plan_suffixes,
                              _partition_ok)
from repair_state import extract_repair_state_v1, F_NODE, F_ACTION_EXPLICIT
from mpre_policy import stepwise_sample
from mpre_replanner import retained_edge_ratio
from mpre import load_cvrp_model
from mpre_trained import load_mpre_trained, partition, edge_insertion_scores


def load_trained_model(cvrp_ckpt, model_ckpt, seed=0):
    """加载训练后的 M-trained 模型（冻结 backbone + 训练后 decoder/adapters）。"""
    from flax import nnx
    model, cfg, _ = load_mpre_trained(cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, seed=seed)
    gd, _, frozen_state, _ = partition(model)
    with open(model_ckpt, 'rb') as f:
        data = pickle.load(f)
    return nnx.merge(gd, data['train_state'], frozen_state), cfg


def _make_extract(env, inst_idx, clock, vehicles, served_mask, visible_ids, capacity, tw_max,
                  node_to_local, local_to_node, Nv, P0, mutable_ids):
    def extract_fn(plans, remaining, legal):
        st = extract_repair_state_v1(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                                     plans, remaining, legal, capacity, tw_max,
                                     allowed_vehicle_ids=mutable_ids,
                                     node_to_local=node_to_local, local_to_node=local_to_node,
                                     Nv=Nv)
        A0 = build_plan_adjacency(P0, node_to_local, Nv)
        st['timestep'] = retained_edge_ratio(A0, st['adjmat'][0])
        return st
    return extract_fn


def mpre_score_fn(backbone, Nv_max=None, M_max=None):
    import jax
    @jax.jit
    def _score(raw_3d, node_valid, timestep, adjmat, cust, pred, succ):
        pair_valid = node_valid[..., None] * node_valid[..., None, :]
        enc_bias = jnp.where(pair_valid, 0.0, -1e9)
        H = backbone.encode(raw_3d, attn_options={'bias': enc_bias})
        dec_bias = jnp.where(pair_valid, adjmat, -1e9)
        L = backbone.decode(H, timestep, dec_bias, target='logit')
        return edge_insertion_scores(L, cust, pred, succ)[0]

    def score_fn(state):
        raw = jnp.array(state['raw_3d']); nv = jnp.array(state['node_valid'])
        adj = jnp.array(state['adjmat'][0])
        cust = jnp.array(state['cust']); pred = jnp.array(state['pred'])
        succ = jnp.array(state['succ']); M = cust.shape[1]
        if Nv_max is not None:
            pn = Nv_max - raw.shape[1]
            raw = jnp.pad(raw, ((0, 0), (0, pn), (0, 0)))
            nv = jnp.pad(nv, ((0, 0), (0, pn)), constant_values=False)
            adj = jnp.pad(adj, ((0, pn), (0, pn)))
        if M_max is not None:
            pm = M_max - M
            cust = jnp.pad(cust, ((0, 0), (0, pm)))
            pred = jnp.pad(pred, ((0, 0), (0, pm)))
            succ = jnp.pad(succ, ((0, 0), (0, pm)))
        s = _score(raw, nv, jnp.array([state['timestep']], jnp.float32), adj, cust, pred, succ)
        return np.asarray(s[:M])
    return score_fn


def mtrained_score_fn(model, Nv_max=None, M_max=None):
    import jax
    @jax.jit
    def _score(raw_3d, node_valid, node_feats, timestep, adjmat, cust, pred, succ, action_feats):
        s, _ = model.score_actions(raw_3d, node_valid, node_feats, timestep, adjmat,
                                   cust, pred, succ, action_feats)
        return s[0]

    def score_fn(state):
        raw = jnp.array(state['raw_3d']); nv = jnp.array(state['node_valid'])
        nf = jnp.array(state['node_feats']); adj = jnp.array(state['adjmat'][0])
        cust = jnp.array(state['cust']); pred = jnp.array(state['pred'])
        succ = jnp.array(state['succ']); af = jnp.array(state['action_feats'])
        M = cust.shape[1]
        if Nv_max is not None:
            pn = Nv_max - raw.shape[1]
            raw = jnp.pad(raw, ((0, 0), (0, pn), (0, 0)))
            nv = jnp.pad(nv, ((0, 0), (0, pn)), constant_values=False)
            nf = jnp.pad(nf, ((0, 0), (0, pn), (0, 0)))
            adj = jnp.pad(adj, ((0, pn), (0, pn)))
        if M_max is not None:
            pm = M_max - M
            cust = jnp.pad(cust, ((0, 0), (0, pm)))
            pred = jnp.pad(pred, ((0, 0), (0, pm)))
            succ = jnp.pad(succ, ((0, 0), (0, pm)))
            af = jnp.pad(af, ((0, 0), (0, pm), (0, 0)))
        s = _score(raw, nv, nf, jnp.array([state['timestep']], jnp.float32), adj,
                   cust, pred, succ, af)
        return np.asarray(s[:M])
    return score_fn


def cc_lns_policy_search(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
                         deferred, contract, budget_s, score_fn, extract_fn, seed=0,
                         protected=frozenset(), n_rounds=N_ROUNDS):
    """统一逐步策略（argmax）的搜索。返回 (best, best_J, stats, event)。"""
    objective = contract.objective
    mutable_ids = {int(v) for v in replan_ids} - set(protected)
    pool = mask_candidate_pool(vehicles, served_mask, visible_ids, protected=protected)
    t0 = time.perf_counter()
    vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                              served_mask, visible_ids, replan_ids, deferred)
    P0 = build_vehicle_plans(env, inst_idx, vehicles)
    r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), contract, objective)
    J0 = r0.J_vis
    P0_feasible = bool(r0.feasible) and bool(r0.finite)
    best = P0
    best_J = J0
    stats = {'n_attempts': 0, 'n_reconstruct_fail': 0, 'n_improve': 0, 'n_eval': 0,
             'n_partition_fail': 0, 'J0': J0, 'J_best': J0, 'P0_feasible': P0_feasible,
             'best_D': r0.D, 'best_Q': r0.Q, 'best_E': r0.E,
             'stop_reason': 'normal', 'elapsed_s': 0.0}
    if not P0_feasible or not pool:
        stats['stop_reason'] = 'P0_invalid' if not P0_feasible else 'empty_pool'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _ev(clock, env, J0, best_J, stats)

    seen = {plan_hash(P0)}
    for _round in range(n_rounds):
        for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
            if time.perf_counter() - t0 > budget_s:
                stats['stop_reason'] = 'budget'
                break
            rng_seed = seed * 10000 + _round * 100 + req_idx
            mask = _select_one_mask(env, inst_idx, best, pool, mask_size, rng_seed)
            if not mask:
                continue
            validate_mask_scope(vehicles, served_mask, visible_ids, best, mask, mutable_ids,
                                protected=protected)
            partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                    tuple(x for x in p.suffix if x not in mask))
                       for vid, p in best.items()}
            cand, _steps, _rec, fail = stepwise_sample(
                score_fn, extract_fn, env, inst_idx, partial, list(mask), mutable_ids,
                deterministic=True)
            stats['n_attempts'] += 1
            if fail is not None or cand is None:
                stats['n_reconstruct_fail'] += 1
                continue
            ch = plan_hash(cand)
            if ch in seen:
                continue
            seen.add(ch)
            ok, _ = _partition_ok(env, inst_idx, clock, vehicles, cand, deferred, served_mask)
            if not ok:
                stats['n_partition_fail'] += 1
                continue
            r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract, objective)
            stats['n_eval'] += 1
            if not r.finite or not r.feasible:
                continue
            if r.J_vis < best_J - 1e-9:
                best = cand
                best_J = r.J_vis
                stats['best_D'], stats['best_Q'], stats['best_E'] = r.D, r.Q, r.E
                stats['n_improve'] += 1
        if stats['stop_reason'] == 'budget':
            break
    stats['J_best'] = best_J
    stats['elapsed_s'] = time.perf_counter() - t0
    return best, best_J, stats, _ev(clock, env, J0, best_J, stats)


def _ev(clock, env, J0, best_J, stats):
    return {'event': int(getattr(env, 'event_id', -1)), 'clock': float(clock),
            'J0': float(J0), 'J_best': float(best_J), 'n_attempts': stats['n_attempts'],
            'n_improve': stats['n_improve'], 'stop_reason': stats['stop_reason'],
            'elapsed_s': stats['elapsed_s'], 'P0_feasible': bool(stats.get('P0_feasible', True))}


class PolicyReplanner(JF1HRepairReplanner):
    """统一逐步策略在线 replanner（M-pre 或 M-trained）。"""

    def __init__(self, score_fn, extract_factory, contract=None, budget_s=1.0, seed=0,
                 capacity=50.0, num_vehicles=25, guard_reserved=True, tw_max=24.0,
                 n_rounds=N_ROUNDS):
        super().__init__(slack_vehicles=1)
        self.score_fn = score_fn
        self.extract_factory = extract_factory
        self.contract = contract
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.guard_reserved = bool(guard_reserved)
        self.tw_max = tw_max
        self.n_rounds = int(n_rounds)
        self.online_stats = defaultdict(int)
        self.event_log = []

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = defaultdict(int)
            self.event_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        t_replanner = time.perf_counter()
        reserved_ids = set()
        if self.guard_reserved:
            idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
            reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        self._reset(inst_idx)
        contract = self.contract if self.contract is not None else env.coldchain_contract
        if contract is None:
            return
        protected = set()
        if self.guard_reserved and reserved_ids:
            P0_view = build_vehicle_plans(env, inst_idx, vehicles)
            for vid in reserved_ids:
                v = vehicles[vid]
                p = P0_view.get(vid)
                if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                    protected.add(vid)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        mutable_ids -= protected
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in visible_ids if int(i) != 0])
        extract_fn = self.extract_factory(env, inst_idx, clock, vehicles, served_mask,
                                          visible_ids, self.capacity, self.tw_max,
                                          node_to_local, local_to_node, Nv, P0, mutable_ids)
        best, best_J, stats, ev = cc_lns_policy_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, contract, self.budget_s, self.score_fn, extract_fn,
            seed=self.seed, protected=protected, n_rounds=self.n_rounds)
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        ok, _detail = _partition_ok(env, inst_idx, clock, vehicles, best,
                                    self.deferred_customers, served_mask)
        changed = 0
        if ok:
            for v in vehicles:
                if v.vehicle_id not in mutable_ids:
                    continue
                p = best.get(v.vehicle_id)
                if p is None:
                    continue
                new_suffix = (list(p.suffix) + [0] if p.suffix
                              else ([] if (p.anchor_node != 0 and has_future) else [0]))
                if new_suffix != v.mutable_suffix:
                    changed += 1
                v.mutable_suffix = new_suffix
            self.sync_deferred_from_vehicles(vehicles)
        for k, v in stats.items():
            if k in ('J0', 'J_best', 'stop_reason', 'elapsed_s', 'best_D', 'best_Q', 'best_E'):
                self.online_stats[k] = v
            else:
                self.online_stats[k] += v
        self.online_stats['n_vehicles_changed'] += changed
        self.online_stats['n_partition_fail_writeback'] += (0 if ok else 1)
        ev['replanner_elapsed_s'] = time.perf_counter() - t_replanner
        ev['writeback_changed'] = changed
        self.event_log.append(ev)
