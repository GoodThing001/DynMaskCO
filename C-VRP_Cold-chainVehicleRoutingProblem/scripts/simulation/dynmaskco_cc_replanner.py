"""DynMaskCO-CC 在线 replanner（M1-edge-v1，K 轮 refinement）。

在 JF1-H-F incumbent 之上：K 轮内对 decision_pool 的客户做 event mask + masked
reconstruction + 候选批量评分 + 当前信息下认证 + 写回。

关键修正（v2）：
  - 候选池边界 = decision_pool（只锁 committed + 非 mutable 车冻结 tail；mutable 车 suffix
    客户允许重分配）；
  - 未分配客户必须有真正参与评分的 DEFER（不因 keep_score is None 直接返回）；
  - 批量候选评分 [1,M,4]→[1,M]（修复形状错误）；
  - 意外异常向上抛（开发 gate 报错），只有合法 certificate 拒绝才算 fallback；
  - mutable scope 在 baseline 生成 incumbent 前保存，refinement 全程沿用。

model=None 时完全等价 JF1-H-F（no-op，step 1）。
"""
import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action)
from dynmaskco_cc_graph import (build_visible_index, build_plan_adjacency, build_event_mask,
                                compute_timestep, resolve_action_endpoints)
from dynmaskco_cc_context import (decision_pool_from_vehicles, mutable_scope,
                                  validate_full_partition, cc_state_dict)
from coldchain_visible_features import extract_action_features
from coldchain_utility_head import feature_only_context


def make_dynmaskco_replanner(model=None, scorer=None, K=1, slack_vehicles=1, capacity=50.0,
                             tw_max=24.0):
    if scorer is None and model is not None:
        from dynmaskco_cc import M1Scorer
        scorer = M1Scorer(model)
    return DynMaskCOReplanner(scorer=scorer, K=K, slack_vehicles=slack_vehicles,
                              capacity=capacity, tw_max=tw_max)


class DynMaskCOReplanner(JF1HRepairReplanner):
    def __init__(self, scorer=None, K=1, slack_vehicles=1, capacity=50.0, tw_max=24.0):
        super().__init__(slack_vehicles=slack_vehicles)
        self.scorer = scorer
        self.K = int(K)
        self.capacity = capacity
        self.tw_max = tw_max
        self.online_stats = {'n_eligible': 0, 'n_candidates': 0, 'n_score': 0, 'n_keep': 0,
                             'n_defer': 0, 'n_accept': 0, 'n_certificate_reject': 0,
                             'n_unexpected_error': 0, 'n_skip_no_struct': 0,
                             'reject_reasons': {}}
        self.timing = {'input_prep': 0.0, 'encode': 0.0, 'model_score': 0.0, 'cert': 0.0}

    def _reset_stats(self, inst_idx):
        if not hasattr(self, '_stats_inst') or self._stats_inst != int(inst_idx):
            self._stats_inst = int(inst_idx)
            self.online_stats = {'n_eligible': 0, 'n_candidates': 0, 'n_score': 0, 'n_keep': 0,
                                 'n_defer': 0, 'n_accept': 0, 'n_certificate_reject': 0,
                                 'n_unexpected_error': 0, 'n_skip_no_struct': 0,
                                 'reject_reasons': {}}
            self.timing = {'input_prep': 0.0, 'encode': 0.0, 'model_score': 0.0, 'cert': 0.0}

    def _record_reject(self, detail):
        for k, v in detail.items():
            if v:  # 非空列表 = 该类别有违例
                self.online_stats['reject_reasons'][k] = \
                    self.online_stats['reject_reasons'].get(k, 0) + 1

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=replan_ids)
        if self.scorer is None:
            return
        self._reset_stats(inst_idx)
        self._reconstruct(env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids)

    def _reconstruct(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids):
        import time
        import jax.numpy as jnp
        node_to_local, local_to_node, Nv = build_visible_index(visible_ids)
        t0 = time.time()
        raw, vis = self._encoder_input(env, inst_idx, visible_ids)
        H = self.scorer.encode(raw, visible_mask=vis)              # [1, N_total, D]
        Hv = H[:, np.asarray(local_to_node, dtype=np.int32)]        # [1, Nv, D] 留设备
        Hv.block_until_ready()                                      # 等设备计算完成再计时
        self.timing['encode'] += time.time() - t0

        for _round in range(self.K):
            accepted_this_round = False
            P = build_vehicle_plans(env, inst_idx, vehicles)
            pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
            pool = sorted(pool, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c)))
            self.online_stats['n_eligible'] += len(pool)
            if not pool:
                break
            for customer in pool:
                ok = self._process_customer(env, inst_idx, clock, vehicles, served_mask,
                                            visible_ids, node_to_local, local_to_node, Nv,
                                            P, Hv, customer, mutable_ids)
                if ok:
                    accepted_this_round = True
                    P = build_vehicle_plans(env, inst_idx, vehicles)
            if not accepted_this_round:
                break

    def _process_customer(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
                          node_to_local, local_to_node, Nv, P, Hv, customer, mutable_ids):
        import time
        import jax.numpy as jnp
        t_prep = time.time()
        cands, _ = enumerate_actions_from_plans(env, inst_idx, P, customer,
                                                allowed_vehicle_ids=mutable_ids)
        feasible = [c for c in cands if c.feasible]
        self.online_stats['n_candidates'] += len(cands)
        if not feasible:
            self.timing['input_prep'] += time.time() - t_prep
            return False

        A0 = build_plan_adjacency(P, node_to_local, Nv)
        adjs = []
        for c in feasible:
            try:
                adjs.append(build_plan_adjacency(
                    apply_action(P, c.action, allowed_vehicle_ids=mutable_ids),
                    node_to_local, Nv))
            except (KeyError, ValueError):
                pass
        if not adjs:
            self.timing['input_prep'] += time.time() - t_prep
            return False
        M, A_in = build_event_mask(A0, adjs)
        if int(M.astype(int).sum()) == 0:
            self.online_stats['n_skip_no_struct'] += 1
            self.timing['input_prep'] += time.time() - t_prep
            return False

        ts = compute_timestep(A0, A_in)

        # 批量构造候选端点 + 显式特征（含 incumbent；无 incumbent 时加 DEFER 伪候选）
        ctx_summary = self._ctx_summary(env, inst_idx, clock, vehicles, visible_ids)
        rows = []          # (candidate_or_None, endpoints, valid, explicit, is_incumbent, is_defer)
        for c in feasible:
            e, v = resolve_action_endpoints(c.action.customer, c.action.slot,
                                            c.action.predecessor, c.action.successor, P,
                                            node_to_local)
            rows.append((c, e, v, self._explicit(ctx_summary, c), c.action.incumbent, False))
        incumbent_row = next((r for r in rows if r[4]), None)
        if incumbent_row is None:
            e, v = resolve_action_endpoints(customer, None, None, None, P, node_to_local)
            exp = self._explicit_defer(ctx_summary, customer)
            rows.append((None, e, v, exp, False, True))

        endpoints = np.stack([r[1] for r in rows])[None]           # [1, M, 4]
        valid = np.stack([r[2] for r in rows])[None]               # [1, M, 4]
        explicit = np.stack([r[3] for r in rows]).astype(np.float32)[None]  # [1, M, F]
        self.timing['input_prep'] += time.time() - t_prep
        t_score = time.time()
        scores = np.asarray(self.scorer.score(
            Hv, jnp.array([ts], np.float32), jnp.asarray(A_in[None], np.float32),
            jnp.asarray(endpoints, jnp.int32), jnp.asarray(valid, bool),
            jnp.asarray(explicit, jnp.float32)))[0]                 # [M]
        self.timing['model_score'] += time.time() - t_score
        self.online_stats['n_score'] += len(rows)

        keep_score = None
        best_idx, best_score = None, float('-inf')
        for i, r in enumerate(rows):
            sc = float(scores[i])
            if r[4]:          # incumbent → KEEP
                keep_score = sc
            elif r[5]:        # defer
                keep_score = sc if keep_score is None else keep_score
            if sc > best_score:
                best_score, best_idx = sc, i

        if keep_score is None:
            self.online_stats['n_unexpected_error'] += 1
            raise RuntimeError("KEEP/DEFER 基准分数缺失")

        margin = 0.0
        best_row = rows[best_idx]
        if best_row[4] or best_row[5] or best_score <= keep_score + margin:
            if best_row[5]:
                self.online_stats['n_defer'] += 1
            else:
                self.online_stats['n_keep'] += 1
            return False

        # 认证 + 写回（完整状态分区：committed ⊎ suffix ⊎ deferred）
        t_cert = time.time()
        new_plan = apply_action(P, best_row[0].action, allowed_vehicle_ids=mutable_ids)
        committed = {int(v.committed_next) for v in vehicles
                     if v.status == 'committed' and v.committed_next not in (None, 0)}
        trial_deferred = set(self.deferred_customers)
        trial_deferred.discard(int(best_row[0].action.customer))
        universe = [int(c) for c in range(1, env.num_nodes)
                    if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                    and env.reveal_time[inst_idx, c] <= clock + 1e-6
                    and int(c) not in committed]
        ok, detail = validate_full_partition(
            committed, new_plan, trial_deferred, universe, served_mask=served_mask,
            future_fn=lambda c: env.reveal_time[inst_idx, c] > clock + 1e-6)
        if not ok:
            self.online_stats['n_certificate_reject'] += 1
            self._record_reject(detail)
            return False
        self.deferred_customers.discard(int(best_row[0].action.customer))
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = new_plan.get(v.vehicle_id)
            if p is None:
                continue
            v.mutable_suffix = list(p.suffix) + [0] if p.suffix else ([0] if p.anchor_node == 0 else [])
        self.timing['cert'] += time.time() - t_cert
        self.online_stats['n_accept'] += 1
        return True

    def _encoder_input(self, env, inst_idx, visible_ids):
        import jax.numpy as jnp
        from cvrptw_utils import coord_normalize_visible
        N = env.num_nodes
        visible = np.zeros(N, bool)
        visible[0] = True
        for c in visible_ids:
            visible[int(c)] = True
        coords = env.coords[inst_idx].astype(np.float32)
        demands = env.demands[inst_idx].astype(np.float32)
        tw_start = env.tw_start[inst_idx].astype(np.float32)
        tw_end = env.tw_end[inst_idx].astype(np.float32)
        temp_class = env.temp_class[inst_idx].astype(np.float32)
        reveal = env.reveal_time[inst_idx].astype(np.float32)
        raw = np.concatenate([coords, (demands / self.capacity)[..., None],
                              (tw_start / self.tw_max)[..., None], (tw_end / self.tw_max)[..., None],
                              (temp_class / 2.0)[..., None], (reveal / self.tw_max)[..., None]],
                             axis=-1).astype(np.float32)
        v = visible[..., None]
        raw[..., 2:] *= v
        raw[..., :2] = raw[..., :2] * v + (1.0 - v) * 0.5
        raw_j = jnp.array(raw[None])
        raw_j = raw_j.at[..., :2].set(coord_normalize_visible(raw_j[..., :2], jnp.array(visible[None])))
        return raw_j, jnp.array(visible[None])

    def _ctx_summary(self, env, inst_idx, clock, vehicles, visible_ids):
        import jax.numpy as jnp
        from coldchain_visible_features import extract_context_features
        snap = self._snapshot_for_features(env, inst_idx, vehicles, visible_ids)
        feats = extract_context_features(self._npz_view(env, inst_idx), 0, snap)
        return np.asarray(feature_only_context(
            jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
            jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))

    def _npz_view(self, env, inst_idx):
        return {'coords': env.coords[inst_idx:inst_idx+1], 'demands': env.demands[inst_idx:inst_idx+1],
                'tw_start': env.tw_start[inst_idx:inst_idx+1], 'tw_end': env.tw_end[inst_idx:inst_idx+1],
                'service_time': env.service_time[inst_idx:inst_idx+1],
                'temp_class': env.temp_class[inst_idx:inst_idx+1],
                'initial_quality': env.initial_quality[inst_idx:inst_idx+1],
                'reveal_time': env.reveal_time[inst_idx:inst_idx+1]}

    def _snapshot_for_features(self, env, inst_idx, vehicles, visible_ids):
        N = env.num_nodes
        vis = np.zeros(N, bool)
        vis[0] = True
        for c in visible_ids:
            vis[int(c)] = True
        return {
            'num_vehicles': env.num_vehicles,
            'visible_mask': vis,
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

    def _explicit(self, ctx_summary, c):
        av, avd = extract_action_features({
            'action': {'customer': c.action.customer, 'slot_kind': c.action.slot.kind,
                       'slot_anchor': c.action.slot.anchor, 'position': c.action.position,
                       'predecessor': c.action.predecessor, 'successor': c.action.successor,
                       'incumbent': c.action.incumbent},
            'is_pseudo': c.action.incumbent})
        cert_vals = np.array([c.incremental_distance or 0.0,
                              c.tw_slack if c.tw_slack is not None else 0.0,
                              c.cap_slack if c.cap_slack is not None else 0.0,
                              c.return_slack if c.return_slack is not None else 0.0], np.float32)
        ptype = 'KEEP' if c.action.incumbent else None
        type_oh = np.array([ptype is None, ptype == 'KEEP', ptype == 'DEFER'], np.float32)
        return np.concatenate([ctx_summary, av, avd.astype(np.float32), cert_vals, type_oh])

    def _explicit_defer(self, ctx_summary, customer):
        av, avd = extract_action_features({
            'is_pseudo': True, 'pseudo': 'DEFER',
            'action': {'customer': customer, 'kind': 'defer'}})
        cert_vals = np.zeros(4, np.float32)
        type_oh = np.array([0.0, 0.0, 1.0], np.float32)   # DEFER
        return np.concatenate([ctx_summary, av, avd.astype(np.float32), cert_vals, type_oh])

    def _universe(self, env, inst_idx, clock, served_mask):
        return [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6]
