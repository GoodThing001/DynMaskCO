"""event_plan_v1 E3 在线 replanner：事件级完整计划生成、评价与选择。

EventPlanReplanner(JF1HRepairReplanner)：baseline 后从 P0 生成候选，按 mode 选择：
  - distance：选完整计划距离最小的候选（含 KEEP）；
  - learned：评分器给 s·(f(P)-f(P0))，选最高分且 > tau，否则 KEEP。
候选不读 g_B；安装时整体写回 mutable_suffix（含 WAIT/RETURN 语义），失败回退 KEEP。
"""
import pickle

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from dynmaskco_cc_context import decision_pool_from_vehicles, cc_state_dict
from event_plan_candidate import generate_simple_candidates_r1
from event_plan_features import extract_plan_delta


def load_plan_scorer(ckpt_path):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from train_event_plan import PlanEvalMLP, CONTEXT_DIM, PLAN_DIM
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    s = ck['s']
    if not (isinstance(s, (int, float)) and np.isfinite(s) and s > 0):
        raise ValueError(f"bad s={s!r}")
    gd, _ = nnx.split(PlanEvalMLP(ck['in_dim'], rngs=ck.get('seed', 0)))
    params = ck['params']

    @jax.jit
    def _f(p, x):
        return nnx.merge(gd, p)(x)

    class _S:
        def __init__(self, s):
            self.s = float(s)

        def score(self, ctx, keep_delta, deltas):
            # 返回 (f(P) - f(P0)) 的分数数组（higher=better）
            import jax.numpy as jnp
            B = len(deltas)
            ctx_b = np.broadcast_to(ctx[None, :], (B, ctx.shape[0]))
            X = jnp.concatenate([jnp.asarray(ctx_b, jnp.float32),
                                 jnp.asarray(deltas, jnp.float32)], axis=-1)
            XK = jnp.concatenate([jnp.asarray(ctx_b, jnp.float32),
                                  jnp.asarray(keep_delta[None].repeat(B, axis=0), jnp.float32)],
                                 axis=-1)
            fP = np.asarray(_f(params, X))
            fK = np.asarray(_f(params, XK))
            return fP - fK

    return _S(s)


def _plan_distance(env, inst_idx, plans):
    from event_plan_candidate import plan_distance
    return plan_distance(env, inst_idx, plans)


class EventPlanReplanner(JF1HRepairReplanner):
    def __init__(self, scorer=None, mode='learned', deindex=True, tau=0.0, capacity=50.0,
                 num_vehicles=25, context_source=None, seed=0):
        super().__init__(slack_vehicles=1)
        self.scorer = scorer
        self.mode = mode
        self.tau = float(tau)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.deindex = bool(deindex)
        self.context_source = context_source
        self.seed = int(seed)
        self.online_stats = {'n_events': 0, 'n_candidates': 0, 'n_accept': 0, 'n_keep': 0,
                             'n_no_candidate': 0}

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = {'n_events': 0, 'n_candidates': 0, 'n_accept': 0, 'n_keep': 0,
                                 'n_no_candidate': 0}

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=replan_ids)
        if self.scorer is None and self.mode != 'distance':
            return
        self._reset(inst_idx)
        self._reconstruct(env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids)

    def _reconstruct(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids):
        ctx = self.context_source.get(inst_idx, env.event_id, clock)
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
        self.online_stats['n_events'] += 1
        if not pool:
            self.online_stats['n_no_candidate'] += 1
            return
        records, plans = generate_simple_candidates_r1(env, inst_idx, P0, pool, mutable_ids,
                                                       seed=self.seed)
        if not plans:
            self.online_stats['n_no_candidate'] += 1
            return
        self.online_stats['n_candidates'] += len(plans)

        if self.mode == 'distance':
            scores = [-_plan_distance(env, inst_idx, P) for P in plans]
            keep_score = -_plan_distance(env, inst_idx, P0)
            best_idx = int(np.argmax(scores))
            gain = scores[best_idx] - keep_score  # >0 = 更短
            accept = gain > 0  # 距离选择无阈值，只选更短者
            best_plan = plans[best_idx]
        else:
            keep_delta = extract_plan_delta(env, inst_idx, P0, P0, self.num_vehicles)
            deltas = np.stack([extract_plan_delta(env, inst_idx, P0, P, self.num_vehicles)
                               for P in plans])
            raw = np.asarray(self.scorer.score(ctx, keep_delta, deltas))  # f(P)-f(P0)
            best_idx = int(np.argmax(raw))
            gB_hat = self.scorer.s * float(raw[best_idx])
            accept = gB_hat > self.tau
            best_plan = plans[best_idx]

        if not accept:
            self.online_stats['n_keep'] += 1
            return
        # 安装：整体写回 mutable_suffix（WAIT/RETURN 语义）
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = best_plan.get(v.vehicle_id)
            if p is None:
                continue
            v.mutable_suffix = (list(p.suffix) + [0] if p.suffix
                                else ([] if (p.anchor_node != 0 and has_future) else [0]))
        self.online_stats['n_accept'] += 1

    def _universe(self, env, inst_idx, clock, served_mask):
        return [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6]
