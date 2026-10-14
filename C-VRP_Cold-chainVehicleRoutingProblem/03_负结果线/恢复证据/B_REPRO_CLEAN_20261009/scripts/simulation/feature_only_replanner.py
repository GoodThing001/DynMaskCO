"""Feature-only 重排器（M0 对照）：用 feature-only utility head 评分候选并接受改善。

与 M1 共享同一在线流程（decision_pool 边界 / 候选枚举 / 认证 / 写回），只是评分器换成
feature-only head（feature_only_context + 候选 action 特征），无 encoder/decoder/event mask。

这是「简单学习对照」：验证简单效用学习是否足以在线改进，不是 M1 的替代。
"""
import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action)
from dynmaskco_cc_context import (decision_pool_from_vehicles, validate_full_partition,
                                  cc_state_dict)
from coldchain_visible_features import extract_action_features
from coldchain_utility_head import feature_only_context


def load_feature_only_scorer(ckpt_path):
    """加载 M0 feature-only checkpoint，构造 jitted 评分器（含尺度 s）。"""
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from train_coldchain_utility_probe import load_checkpoint, build_head
    ckpt = load_checkpoint(ckpt_path)
    graphdef = build_head(ckpt['config'])
    params = ckpt['params']
    s = ckpt['config'].get('s')
    if s is None or s <= 0:
        raise ValueError(f"feature-only checkpoint 缺正数 s：{s!r}")
    gd = graphdef

    @jax.jit
    def _score(p, context, action_vals, action_valid):
        hd = nnx.merge(gd, p)
        return hd(context, action_vals, action_valid)

    class _S:
        def __init__(self, s):
            self.s = float(s)

        def score(self, context, action_vals, action_valid):
            return _score(params, context, action_vals, action_valid)

    return _S(s)


def make_feature_only_replanner(scorer=None, ckpt_path=None, K=1, slack_vehicles=1):
    if scorer is None and ckpt_path is not None:
        scorer = load_feature_only_scorer(ckpt_path)
    return FeatureOnlyReplanner(scorer=scorer, K=K, slack_vehicles=slack_vehicles)


class FeatureOnlyReplanner(JF1HRepairReplanner):
    def __init__(self, scorer=None, K=1, slack_vehicles=1):
        super().__init__(slack_vehicles=slack_vehicles)
        self.scorer = scorer
        self.K = int(K)
        self.online_stats = {'n_eligible': 0, 'n_candidates': 0, 'n_score': 0, 'n_keep': 0,
                             'n_defer': 0, 'n_accept': 0, 'n_certificate_reject': 0,
                             'n_unexpected_error': 0}
        self.timing = {'input_prep': 0.0, 'model_score': 0.0, 'cert': 0.0}

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = {'n_eligible': 0, 'n_candidates': 0, 'n_score': 0, 'n_keep': 0,
                                 'n_defer': 0, 'n_accept': 0, 'n_certificate_reject': 0,
                                 'n_unexpected_error': 0}
            self.timing = {'input_prep': 0.0, 'model_score': 0.0, 'cert': 0.0}

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=replan_ids)
        if self.scorer is None:
            return
        self._reset(inst_idx)
        self._reconstruct(env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids)

    def _ctx_summary(self, env, inst_idx, vehicles, visible_ids):
        import jax.numpy as jnp
        from coldchain_visible_features import extract_context_features
        N = env.num_nodes
        vis = np.zeros(N, bool)
        vis[0] = True
        for c in visible_ids:
            vis[int(c)] = True
        snap = {
            'num_vehicles': env.num_vehicles, 'visible_mask': vis,
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
        ds = {'coords': env.coords[inst_idx:inst_idx+1], 'demands': env.demands[inst_idx:inst_idx+1],
              'tw_start': env.tw_start[inst_idx:inst_idx+1], 'tw_end': env.tw_end[inst_idx:inst_idx+1],
              'service_time': env.service_time[inst_idx:inst_idx+1],
              'temp_class': env.temp_class[inst_idx:inst_idx+1],
              'initial_quality': env.initial_quality[inst_idx:inst_idx+1],
              'reveal_time': env.reveal_time[inst_idx:inst_idx+1]}
        feats = extract_context_features(ds, 0, snap)
        return np.asarray(feature_only_context(
            jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
            jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))

    def _reconstruct(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids):
        import time
        ctx = self._ctx_summary(env, inst_idx, vehicles, visible_ids)
        for _round in range(self.K):
            accepted = False
            P = build_vehicle_plans(env, inst_idx, vehicles)
            pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
            pool = sorted(pool, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c)))
            self.online_stats['n_eligible'] += len(pool)
            if not pool:
                break
            for customer in pool:
                ok = self._process(env, inst_idx, clock, vehicles, served_mask, P, ctx,
                                   customer, mutable_ids)
                if ok:
                    accepted = True
                    P = build_vehicle_plans(env, inst_idx, vehicles)
            if not accepted:
                break

    def _action_dict(self, a):
        return {'customer': a.customer, 'slot_kind': a.slot.kind, 'slot_anchor': a.slot.anchor,
                'position': a.position, 'predecessor': a.predecessor, 'successor': a.successor,
                'incumbent': a.incumbent}

    def _process(self, env, inst_idx, clock, vehicles, served_mask, P, ctx, customer, mutable_ids):
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

        rows = []  # (candidate_or_None, action_vals, action_valid, is_incumbent, is_defer)
        for c in feasible:
            av, avd = extract_action_features({
                'action': self._action_dict(c.action), 'is_pseudo': c.action.incumbent})
            rows.append((c, av, avd, c.action.incumbent, False))
        incumbent = next((r for r in rows if r[3]), None)
        if incumbent is None:
            av, avd = extract_action_features({
                'is_pseudo': True, 'pseudo': 'DEFER',
                'action': {'customer': customer, 'kind': 'defer'}})
            rows.append((None, av, avd, False, True))

        action_vals = np.stack([r[1] for r in rows])[None]          # [1, M, 8]
        action_valid = np.stack([r[2] for r in rows])[None]         # [1, M, 8]
        self.timing['input_prep'] += time.time() - t_prep

        t_score = time.time()
        scores = np.asarray(self.scorer.score(
            jnp.asarray(ctx[None]), jnp.asarray(action_vals, jnp.float32),
            jnp.asarray(action_valid, bool)))[0]                    # [M]
        self.timing['model_score'] += time.time() - t_score
        self.online_stats['n_score'] += len(rows)

        keep_score = None
        best_idx, best_score = None, float('-inf')
        for i, r in enumerate(rows):
            sc = float(scores[i])
            if r[3]:
                keep_score = sc
            elif r[4] and keep_score is None:
                keep_score = sc
            if sc > best_score:
                best_score, best_idx = sc, i
        if keep_score is None:
            self.online_stats['n_unexpected_error'] += 1
            raise RuntimeError("KEEP/DEFER 基准分数缺失")

        best_row = rows[best_idx]
        if best_row[3] or best_row[4] or best_score <= keep_score + 0.0:
            if best_row[4]:
                self.online_stats['n_defer'] += 1
            else:
                self.online_stats['n_keep'] += 1
            return False

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
            self.online_stats.setdefault('reject_reasons', {})
            for k, v in detail.items():
                if v:
                    self.online_stats['reject_reasons'][k] = \
                        self.online_stats['reject_reasons'].get(k, 0) + 1
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

    def _universe(self, env, inst_idx, clock, served_mask):
        return [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6]
