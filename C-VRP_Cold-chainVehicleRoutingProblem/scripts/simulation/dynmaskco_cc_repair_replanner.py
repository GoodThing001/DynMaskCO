"""N1/N2 在线 replanner：MaskCO 条件联合修复器（greedy 插入重构）。

流程：JF1-H-F + 预留保护 → 对每个 mask 用训练好的模型做 greedy 插入重构（整体移除 mask →
模型对合法「客户—槽位—插入位置」打分 → 选最高分插入 → 重复），完整候选统一认证 + J_vis 比较，
事件结束写回最佳计划。接受始终在完整计划层面，不回退到「每移一个客户立即接受」。

模型 = 冻结 encoder + 可训练 MaskCODecoder + 插入头（train_dynmaskco_cc_repair 的 model.ckpt）。
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action,
                             plan_hash)
from dynmaskco_cc_context import decision_pool_from_vehicles
from dynmaskco_cc_graph import (build_visible_index, build_plan_adjacency, resolve_action_endpoints)
from visible_state import build_visible_state, evaluate_visible_plan
from cc_lns_replanner import REQUEST_SEQUENCE, _select_one_mask, _plan_suffixes, _partition_ok


def load_repair_model(ckpt_path, encoder_ckpt_path):
    from flax import nnx
    import jax
    from dynmaskco_cc import MaskCODecoder, ScoringMLP, DynMaskCOModel
    from train_fleet_head import load_base_model
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    d_model = data['d_model']
    explicit_dim = data['explicit_dim']
    decoder = MaskCODecoder(d_enc=d_model, d_dec=d_model, num_layers=6, num_heads=8, rngs=0)
    head = ScoringMLP(4 * d_model * 2 + 4 + explicit_dim, hidden=(128, 64), rngs=0)
    d_gd, _ = nnx.split(decoder)
    h_gd, _ = nnx.split(head)
    decoder = nnx.merge(d_gd, data['decoder'])
    head = nnx.merge(h_gd, data['head'])
    encoder = load_base_model(encoder_ckpt_path)
    return DynMaskCOModel(encoder, decoder, head, rngs=0)


class RepairScorer:
    """持久 JIT 评分器：encoder Hv（每事件一次）+ decoder Z（每候选一次）+ 插入头打分（每步）。"""

    def __init__(self, model):
        from flax import nnx
        import jax
        import jax.numpy as jnp
        from dynmaskco_cc import gather_endpoints
        self.encoder = model.encoder
        self.d_graphdef, self.d_params = nnx.split(model.decoder)
        self.h_graphdef, self.h_params = nnx.split(model.utility_head)

        @jax.jit
        def _decode(d_params, Hv, ts, A_in):
            dec = nnx.merge(self.d_graphdef, d_params)
            return dec(Hv, ts, A_in)

        @jax.jit
        def _score_Z(h_params, Hv, Z, endpoints, valid, explicit):
            hd = nnx.merge(self.h_graphdef, h_params)
            struct = gather_endpoints(Hv, Z, endpoints, valid)
            x = jnp.concatenate([struct, explicit], axis=-1)
            return hd(x)

        self._decode = _decode
        self._score_Z = _score_Z

    def encode(self, raw, visible_mask):
        return self.encoder.encode(raw, visible_mask=visible_mask)

    def decode(self, Hv, ts, A_in):
        return self._decode(self.d_params, Hv, ts, A_in)

    def score_with_Z(self, Hv, Z, endpoints, valid, explicit):
        return self._score_Z(self.h_params, Hv, Z, endpoints, valid, explicit)


def greedy_reconstruct(model, Hv, partial, mask_customers, mutable_ids, env, inst_idx,
                       node_to_local, Nv):
    """模型 greedy 插入重构：返回 (plan, steps)。不可行返回 (None, [])。"""
    import jax.numpy as jnp
    P = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
         for vid, p in partial.items()}
    A_partial = build_plan_adjacency(P, node_to_local, Nv)
    ts = jnp.array([0.5], jnp.float32)
    A_in = jnp.array(A_partial[None].astype(np.float32))
    Z = model.decode(Hv[None], ts, A_in)   # 每候选解码一次
    remaining = list(mask_customers)
    steps = []
    while remaining:
        ends, vals, exps = [], [], []
        cands_by = []   # (customer, action)
        for c in remaining:
            cands, _ = enumerate_actions_from_plans(env, inst_idx, P, int(c),
                                                    allowed_vehicle_ids=mutable_ids)
            for cand in cands:
                if not cand.feasible:
                    continue
                a = cand.action
                e, v = resolve_action_endpoints(a.customer, a.slot, a.predecessor, a.successor,
                                                P, node_to_local)
                ends.append(e); vals.append(v)
                exps.append([float(cand.incremental_distance),
                             0.0 if cand.tw_slack is None else float(cand.tw_slack),
                             0.0 if cand.cap_slack is None else float(cand.cap_slack),
                             0.0 if cand.return_slack is None else float(cand.return_slack)])
                cands_by.append((int(c), a))
        if not cands_by:
            return None, []
        endpoints = jnp.array(np.stack(ends)[None].astype(np.int32))
        valid = jnp.array(np.stack(vals)[None].astype(bool))
        explicit = jnp.array(np.stack(exps)[None].astype(np.float32))
        scores = np.asarray(model.score_with_Z(Hv[None], Z, endpoints, valid, explicit))[0]
        if not np.isfinite(scores).all():
            return None, []
        best_i = int(np.argmax(scores))
        c, a = cands_by[best_i]
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        steps.append({'customer': int(a.customer), 'slot_kind': a.slot.kind,
                      'slot_anchor': int(a.slot.anchor), 'position': int(a.position),
                      'predecessor': int(a.predecessor), 'successor': int(a.successor)})
        remaining.remove(c)
    return P, steps


class RepairReplanner(JF1HRepairReplanner):
    """MaskCO 条件联合修复器在线 replanner（N2 主方法）。"""

    def __init__(self, model, scorer, contract=None, budget_s=1.0, seed=0, capacity=50.0,
                 num_vehicles=25, guard_reserved=True):
        super().__init__(slack_vehicles=1)
        self.model = model
        self.scorer = scorer
        self.contract = contract
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.guard_reserved = bool(guard_reserved)
        self.online_stats = {}
        self.event_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = {'n_attempts': 0, 'n_changed': 0, 'elapsed_s': 0.0}
            self.event_log = []
        t0 = time.perf_counter()
        reserved_ids = set()
        if self.guard_reserved:
            idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
            reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
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
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
        vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                                  served_mask, visible_ids, replan_ids, self.deferred_customers)
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), contract, contract.objective)
        best = P0
        best_J = r0.J_vis
        n_attempts = 0
        if r0.feasible and r0.finite and pool:
            # encoder Hv（每事件一次）
            from train_dynmaskco_cc import _build_encoder_raw
            num_nodes = env.coords.shape[1]
            visible = np.zeros(num_nodes, bool); visible[0] = True
            visible |= (env.reveal_time[inst_idx] <= clock + 1e-6)
            visible_ids_enc = [int(i) for i in range(1, num_nodes) if visible[i]]
            node_to_local, local_to_node, Nv = build_visible_index(visible_ids_enc)
            tw_max = float(env.tw_end[:, 0].max())
            raw_j, vis_j = _build_encoder_raw(
                {'coords': env.coords, 'demands': env.demands, 'tw_start': env.tw_start,
                 'tw_end': env.tw_end, 'temp_class': env.temp_class,
                 'reveal_time': env.reveal_time}, inst_idx, clock, visible, self.capacity, tw_max)
            H = self.scorer.encode(raw_j, visible_mask=vis_j)
            Hv = np.asarray(H[0])[np.asarray(local_to_node, dtype=np.int32)]
            for _round in range(2):
                for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
                    if time.perf_counter() - t0 > self.budget_s:
                        break
                    rng_seed = self.seed * 10000 + _round * 100 + req_idx
                    mask = _select_one_mask(env, inst_idx, best, pool, mask_size, rng_seed)
                    if not mask:
                        continue
                    partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time,
                                            p.anchor_load,
                                            tuple(x for x in p.suffix if x not in mask))
                               for vid, p in best.items()}
                    cand, _steps = greedy_reconstruct(self.scorer, Hv, partial, mask, mutable_ids,
                                                      env, inst_idx, node_to_local, Nv)
                    n_attempts += 1
                    if cand is None:
                        continue
                    ok, _ = _partition_ok(env, inst_idx, clock, vehicles, cand,
                                          self.deferred_customers, served_mask)
                    if not ok:
                        continue
                    r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract,
                                              contract.objective)
                    if r.finite and r.feasible and r.J_vis < best_J - 1e-9:
                        best = cand
                        best_J = r.J_vis
        # 写回（WAIT/RETURN 语义，保护车不改写）
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        changed = 0
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
        self.online_stats['n_attempts'] += n_attempts
        self.online_stats['n_changed'] += changed
        self.online_stats['elapsed_s'] += time.perf_counter() - t0
        self.event_log.append({'event': int(getattr(env, 'event_id', -1)), 'clock': float(clock),
                               'n_attempts': n_attempts, 'J_best': float(best_J),
                               'changed': changed})
