"""event_plan_v2 E3 在线 replanner：事件级完整计划生成、评价与选择。

EventPlanReplanner(JF1HRepairReplanner)：baseline 后从 P0 生成候选，按 mode 选择：
  - distance：选完整计划距离最小（含返仓段；WAIT/RETURN 区分）的候选（含 KEEP）；
  - proxy：无学习物理代理 S_proxy —— 按可见投影 J(P0)-J(P) 选计划，同 J 单位阈值；
  - learned：评分器给 s·(f(P)-f(P0))，选最高分且 > tau，否则 KEEP（A/B 两种表示）。

候选不读 g_B；安装时整体写回 mutable_suffix（含 WAIT/RETURN 语义），失败回退 KEEP。
完整分区认证 + 分数 finite 检查。
"""
import pickle

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from dynmaskco_cc_context import (decision_pool_from_vehicles, validate_full_partition)
from event_plan_candidate import generate_simple_candidates_r1, plan_distance
from event_plan_features import extract_plan_delta, extract_plan_consequence
from event_plan_projection import project_plan, proxy_cost


def load_plan_scorer(ckpt_path):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from train_event_plan import PlanEvalMLP, CONTEXT_DIM
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    s = ck['s']
    if not (isinstance(s, (int, float)) and np.isfinite(s) and s > 0):
        raise ValueError(f"bad s={s!r}")
    representation = ck.get('representation')
    if representation not in ('A', 'B'):
        raise ValueError(f"unknown representation {representation!r}")
    gd, _ = nnx.split(PlanEvalMLP(ck['in_dim'], rngs=ck.get('seed', 0)))
    params = ck['params']

    @jax.jit
    def _f(p, x):
        return nnx.merge(gd, p)(x)

    class _S:
        def __init__(self, s):
            self.s = float(s)
            self.representation = representation

        def score(self, ctx, keep_feat, feats):
            B = len(feats)
            ctx_b = np.broadcast_to(ctx[None, :], (B, ctx.shape[0]))
            X = jnp.concatenate([jnp.asarray(ctx_b, jnp.float32),
                                 jnp.asarray(feats, jnp.float32)], axis=-1)
            XK = jnp.concatenate([jnp.asarray(ctx_b, jnp.float32),
                                  jnp.asarray(keep_feat[None].repeat(B, axis=0), jnp.float32)],
                                 axis=-1)
            fP = np.asarray(_f(params, X))
            fK = np.asarray(_f(params, XK))
            return fP - fK

    return _S(s)


def plan_proxy_J(env, inst_idx, vehicles, P, contract, has_future):
    """整队可见投影代理 J（无学习 S_proxy 的目标）。"""
    proj = project_plan(env, inst_idx, vehicles, P, contract, has_future)
    return float(sum(proxy_cost(p, contract.objective) for p in proj.values()))


class EventPlanReplanner(JF1HRepairReplanner):
    def __init__(self, scorer=None, mode='learned', representation='A', deindex=True,
                 tau=0.0, capacity=50.0, num_vehicles=25, context_source=None, seed=0,
                 contract=None):
        super().__init__(slack_vehicles=1)
        self.scorer = scorer
        self.mode = mode
        self.representation = representation
        self.tau = float(tau)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.deindex = bool(deindex)
        self.context_source = context_source
        self.seed = int(seed)
        self.contract = contract
        self.online_stats = {'n_events': 0, 'n_candidates': 0, 'n_accept': 0, 'n_keep': 0,
                             'n_no_candidate': 0, 'n_certificate_reject': 0,
                             'n_unexpected_error': 0}
        self.decision_log = []

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = {'n_events': 0, 'n_candidates': 0, 'n_accept': 0, 'n_keep': 0,
                                 'n_no_candidate': 0, 'n_certificate_reject': 0,
                                 'n_unexpected_error': 0}
            self.decision_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        if self.scorer is None and self.mode == 'learned':
            return
        self._reset(inst_idx)
        self._reconstruct(env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids)

    def _reconstruct(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids):
        if self.context_source is None:
            raise RuntimeError("缺 pre-prepare context source（必须注册 snapshot_hook 采集器）")
        ctx = self.context_source.get(inst_idx, env.event_id, clock)
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
        self.online_stats['n_events'] += 1
        if not pool:
            self.online_stats['n_no_candidate'] += 1
            return
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        records, plan_entries = generate_simple_candidates_r1(
            env, inst_idx, P0, pool, mutable_ids, seed=self.seed, has_future=has_future)
        if not plan_entries:
            self.online_stats['n_no_candidate'] += 1
            return
        self.online_stats['n_candidates'] += len(plan_entries)

        plans = [e['plan'] for e in plan_entries]
        if self.mode == 'distance':
            gains = [-(plan_distance(env, inst_idx, P, has_future)
                       - plan_distance(env, inst_idx, P0, has_future)) for P in plans]
            gains = np.asarray(gains, np.float64)
        elif self.mode == 'proxy':
            contract = self.contract if self.contract is not None else env.coldchain_contract
            J0 = plan_proxy_J(env, inst_idx, vehicles, P0, contract, has_future)
            gains = np.asarray([J0 - plan_proxy_J(env, inst_idx, vehicles, P, contract,
                                                  has_future) for P in plans], np.float64)
        else:
            contract = self.contract if self.contract is not None else env.coldchain_contract
            if self.representation == 'A':
                keep_feat = extract_plan_delta(env, inst_idx, P0, P0, self.num_vehicles)
                feats = np.stack([extract_plan_delta(env, inst_idx, P0, P, self.num_vehicles)
                                  for P in plans])
            else:
                keep_feat = extract_plan_consequence(
                    env, inst_idx, vehicles, P0, P0, contract, None, self.capacity,
                    self.num_vehicles, has_future)
                feats = np.stack([extract_plan_consequence(
                    env, inst_idx, vehicles, P0, P, contract, None, self.capacity,
                    self.num_vehicles, has_future) for P in plans])
            raw = np.asarray(self.scorer.score(ctx, keep_feat, feats))
            if not np.isfinite(raw).all():
                self.online_stats['n_unexpected_error'] += 1
                raise RuntimeError("评分含 NaN/Inf")
            gains = self.scorer.s * raw

        if not np.isfinite(gains).all():
            self.online_stats['n_unexpected_error'] += 1
            raise RuntimeError("gain 含 NaN/Inf")
        best_idx = int(np.argmax(gains))
        accept = bool(float(gains[best_idx]) > self.tau)
        best_plan = plans[best_idx]

        if not accept:
            self.online_stats['n_keep'] += 1
            self.decision_log.append({'event': int(env.event_id), 'clock': float(clock),
                                      'result': 'keep', 'best_gain': float(gains[best_idx]),
                                      'tau': self.tau})
            return

        # 完整分区认证（原子安装，拒绝无残留）
        committed = {int(v.committed_next) for v in vehicles
                     if v.status == 'committed' and v.committed_next not in (None, 0)}
        universe = [int(c) for c in range(1, env.num_nodes)
                    if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                    and env.reveal_time[inst_idx, c] <= clock + 1e-6
                    and int(c) not in committed]
        suffix_set = {int(x) for p in best_plan.values() for x in p.suffix}
        trial_deferred = set(self.deferred_customers) - suffix_set
        ok, detail = validate_full_partition(
            committed, best_plan, trial_deferred, universe, served_mask=served_mask,
            future_fn=lambda c: env.reveal_time[inst_idx, c] > clock + 1e-6)
        if not ok:
            self.online_stats['n_certificate_reject'] += 1
            self.online_stats['n_keep'] += 1
            self.decision_log.append({'event': int(env.event_id), 'clock': float(clock),
                                      'result': 'cert_reject', 'detail': detail,
                                      'best_gain': float(gains[best_idx]), 'tau': self.tau})
            return

        # 写回：整体写回 mutable_suffix（WAIT/RETURN 语义）
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = best_plan.get(v.vehicle_id)
            if p is None:
                continue
            v.mutable_suffix = (list(p.suffix) + [0] if p.suffix
                                else ([] if (p.anchor_node != 0 and has_future) else [0]))
        self.online_stats['n_accept'] += 1
        self.decision_log.append({'event': int(env.event_id), 'clock': float(clock),
                                  'result': 'accept', 'best_gain': float(gains[best_idx]),
                                  'tau': self.tau})

    def _universe(self, env, inst_idx, clock, served_mask):
        return [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6]
