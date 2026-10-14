"""70 维 feature-local 评分器在线重排器（F_H/F_R 的部署适配层）。

与 feature_only_replanner 同一在线流程（decision_pool 边界 / 候选枚举 / 认证 / 写回），
差异：
  - 模型 = ScoringMLP(70, (128,64))（train_state_2x2 的 F 组 70 维，无 encoder/状态）；
  - 输入 = context(25) + action(8+8) + 候选局部(25+4) = 70，与离线
    `train_feature_only_local.build_dataset` / `train_m1_compact.build_explicit_and_labels`
    逐字段一致（硬不变量 #9）。

接受规则：`g = s * (f(a) - f(KEEP/DEFER))`，accept iff `g > tau`（tau 来自 CAL 选阈，
s 来自 checkpoint，二者均不重标定；tau=∞ 时只保持 baseline）。

这是④接口对账所需的「真实在线路径」——在线一侧从这里构造特征/评分/写回，不是把离线
构造器调用两遍冒充一致性。⑤闭环运行仍待④通过且协议固定后另行授权。
"""
import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action)
from dynmaskco_cc_context import (decision_pool_from_vehicles, validate_full_partition,
                                  cc_state_dict)
from coldchain_visible_features import (extract_action_features,
                                        extract_candidate_local_features,
                                        extract_context_features,
                                        ACTION_FEAT_DIM, LOCAL_VAL_DIM, LOCAL_VALID_DIM)
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM

IN_DIM = FEATURE_CONTEXT_DIM + 2 * ACTION_FEAT_DIM + LOCAL_VAL_DIM + LOCAL_VALID_DIM  # 70


def _action_dict_of(a):
    """FleetAction → 与 teacher 导出 `_action_payload` 同字段的动作 dict。"""
    return {'customer': a.customer, 'slot_kind': a.slot.kind, 'slot_anchor': a.slot.anchor,
            'position': a.position, 'predecessor': a.predecessor, 'successor': a.successor,
            'incumbent': a.incumbent}


def _npz_view(env, inst_idx):
    """单实例切片（与离线 data_npz[inst_idx] 等价，供特征提取索引 0）。"""
    return {'coords': env.coords[inst_idx:inst_idx+1], 'demands': env.demands[inst_idx:inst_idx+1],
            'tw_start': env.tw_start[inst_idx:inst_idx+1], 'tw_end': env.tw_end[inst_idx:inst_idx+1],
            'service_time': env.service_time[inst_idx:inst_idx+1],
            'temp_class': env.temp_class[inst_idx:inst_idx+1],
            'initial_quality': env.initial_quality[inst_idx:inst_idx+1],
            'reveal_time': env.reveal_time[inst_idx:inst_idx+1]}


def plan_to_dict(plans):
    """FleetPlan → JSON 可序列化 dict（vid -> anchor/suffix），供决策日志与离线归因恢复。"""
    return {int(vid): {'anchor_node': int(p.anchor_node),
                       'suffix': [int(x) for x in p.suffix]}
            for vid, p in plans.items()}


def plan_distance(env, inst_idx, plans):
    """完整车队计划距离（当前已知计划差，非候选 incremental_distance）。

    每辆车：anchor_node → suffix[0] → … → suffix[-1]（suffix 尾部常含 0 返仓）。
    """
    total = 0.0
    for p in plans.values():
        prev = int(p.anchor_node)
        for c in p.suffix:
            total += float(env.dist_mat[inst_idx, prev, int(c)])
            prev = int(c)
    return total


class PrePrepareContextCollector:
    """在 snapshot_hook 触发时保存 pre-prepare（prepare_decision_point 之前）的 25 维 context。

    只读：不修改车辆/计划/deferred/时钟/冷链。保存由白名单字段构造的 context，归属键
    (inst_idx, event_id, clock)。get 缺失/归属不符时显式报错，绝不回退 post-prepare。

    用途：评分器需要与训练 snapshot 同时点的 context；物理执行仍走 prepare_decision_point
    的 bump 后状态。这是方案 (a)「保留训练语义」的扩展侧实现，不改 strict_online_env.py。
    """

    def __init__(self, deindex, save_snapshot=False):
        self.deindex = bool(deindex)
        self.save_snapshot = bool(save_snapshot)
        self.cache = {}
        self.snapshot_cache = {}

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers):
        import jax.numpy as jnp
        replan_ids = {v.vehicle_id for v in vehicles
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if not replan_ids:
            return  # 与 teacher snapshot 一致：无重规划需求不保存
        if self.save_snapshot:
            from recourse_snapshot import capture_recourse_snapshot
            snap = capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                             traces, served_mask, all_customers)
            self.snapshot_cache[(int(inst_idx), int(event_id), float(clock))] = snap
        vis = np.zeros(env.num_nodes, bool)
        vis[0] = True
        for c in all_customers:
            if env.reveal_time[inst_idx, c] <= clock + 1e-6:
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
        ds = _npz_view(env, inst_idx)
        feats = extract_context_features(ds, 0, snap, deindex=self.deindex)
        ctx = np.asarray(feature_only_context(
            jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
            jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))
        self.cache[(int(inst_idx), int(event_id), float(clock))] = ctx.copy()

    def get(self, inst_idx, event_id, clock):
        key = (int(inst_idx), int(event_id), float(clock))
        if key not in self.cache:
            raise RuntimeError(f"缺 pre-prepare context：inst={inst_idx} event={event_id} "
                               f"clock={clock}；已有键={sorted(self.cache)}")
        return self.cache[key]


def load_feature_local_scorer(ckpt_path):
    """加载 F 组 70 维 ScoringMLP checkpoint（train_state_2x2 的 model.ckpt）。"""
    import pickle
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from dynmaskco_cc_compact import ScoringMLP
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    in_dim = ck.get('in_dim')
    if in_dim != IN_DIM:
        raise ValueError(f"feature-local checkpoint in_dim={in_dim}，期望 {IN_DIM}（70 维）")
    if ck.get('use_state'):
        raise ValueError("use_state=True 是 S 组 83 维，不是本适配层的 F 组 70 维")
    s = ck.get('s')
    if not (isinstance(s, (int, float)) and np.isfinite(s) and s > 0):
        raise ValueError(f"checkpoint 的 s 必须有限且为正：{s!r}")
    graphdef, _ = nnx.split(ScoringMLP(in_dim, (128, 64), rngs=0))
    params = ck['params']

    @jax.jit
    def _score(p, x):
        return nnx.merge(graphdef, p)(x)

    class _S:
        def __init__(self, s):
            self.s = float(s)

        def score(self, x):
            return _score(params, x)

    return _S(s)


class FeatureLocalReplanner(JF1HRepairReplanner):
    def __init__(self, scorer=None, ckpt_path=None, K=1, slack_vehicles=1, tau=0.0,
                 capacity=50.0, deindex=None, context_source=None, max_total_accepts=None,
                 max_accepts_per_event=None):
        if deindex is None:
            raise ValueError("deindex 必须显式给出（与训练 manifest 一致），不得默认猜测")
        super().__init__(slack_vehicles=slack_vehicles)
        if scorer is None and ckpt_path is not None:
            scorer = load_feature_local_scorer(ckpt_path)
        self.scorer = scorer
        self.K = int(K)
        self.tau = float(tau)          # J 单位接受阈值；τ=∞ 表示只保持 baseline
        self.capacity = float(capacity)
        self.deindex = bool(deindex)
        self.context_source = context_source
        self.max_total_accepts = max_total_accepts   # None=不限；1=ONE（整条轨迹首次接受后回 baseline）
        self.max_accepts_per_event = max_accepts_per_event  # None=不限；1=EVENT-ONE（每事件最多一次）
        self._accepts_total = 0
        self._accepts_this_event = 0
        self._decision_seq = 0
        self.decision_log = []         # 每次客户决策一条（含 accept 的 P_before/P_after）
        self.event_plans = []          # 每事件三份计划 hash：baseline 前/后、learned 后
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
            self._accepts_total = 0
            self._decision_seq = 0
            self.decision_log = []
            self.event_plans = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self._accepts_this_event = 0  # 每事件重置（EVENT-ONE 语义）
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        from action_contract import plan_hash
        P_before = build_vehicle_plans(env, inst_idx, vehicles)
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=replan_ids)
        P_base = build_vehicle_plans(env, inst_idx, vehicles)
        if self.scorer is None:
            return
        self._reset(inst_idx)
        if self.max_total_accepts is not None and self._accepts_total >= self.max_total_accepts:
            self.event_plans.append({'instance': int(inst_idx), 'event': int(env.event_id),
                                     'clock': float(clock),
                                     'baseline_before': plan_hash(P_before),
                                     'baseline_after': plan_hash(P_base),
                                     'learned_after': plan_hash(P_base)})
            return
        self._reconstruct(env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids)
        P_learned = build_vehicle_plans(env, inst_idx, vehicles)
        self.event_plans.append({'instance': int(inst_idx), 'event': int(env.event_id),
                                 'clock': float(clock),
                                 'baseline_before': plan_hash(P_before),
                                 'baseline_after': plan_hash(P_base),
                                 'learned_after': plan_hash(P_learned)})

    def _reconstruct(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, mutable_ids):
        import time
        if self.context_source is None:
            raise RuntimeError("缺 pre-prepare context source（必须注册 snapshot_hook 采集器）")
        ctx = self.context_source.get(inst_idx, env.event_id, clock)
        ds = _npz_view(env, inst_idx)
        for _round in range(self.K):
            accepted = False
            P = build_vehicle_plans(env, inst_idx, vehicles)
            pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
            pool = sorted(pool, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c)))
            self.online_stats['n_eligible'] += len(pool)
            if not pool:
                break
            for customer in pool:
                ok = self._process(env, inst_idx, clock, vehicles, served_mask, P, ctx, ds,
                                   customer, mutable_ids)
                if ok:
                    accepted = True
                    P = build_vehicle_plans(env, inst_idx, vehicles)
                    if self.max_total_accepts is not None and self._accepts_total >= self.max_total_accepts:
                        break
                    if (self.max_accepts_per_event is not None
                            and self._accepts_this_event >= self.max_accepts_per_event):
                        break
            if not accepted:
                break
            if self.max_total_accepts is not None and self._accepts_total >= self.max_total_accepts:
                break
            if (self.max_accepts_per_event is not None
                    and self._accepts_this_event >= self.max_accepts_per_event):
                break

    def _action_dict(self, a):
        return {'customer': a.customer, 'slot_kind': a.slot.kind, 'slot_anchor': a.slot.anchor,
                'position': a.position, 'predecessor': a.predecessor, 'successor': a.successor,
                'incumbent': a.incumbent}

    def _log_decision(self, env, inst_idx, clock, customer, result, action=None, g_hat=None,
                      best_score=None, keep_score=None, P_before=None, P_after=None,
                      reject_reasons=None):
        rec = {
            'instance': int(inst_idx), 'event': int(env.event_id), 'clock': float(clock),
            'seq': int(self._decision_seq), 'customer': int(customer), 'result': result,
            'g_hat': (float(g_hat) if g_hat is not None else None),
            'best_score': (float(best_score) if best_score is not None else None),
            'keep_score': (float(keep_score) if keep_score is not None else None),
            'tau': float(self.tau),
        }
        self._decision_seq += 1
        if action is not None:
            rec['action'] = action
        if P_before is not None:
            rec['P_before'] = plan_to_dict(P_before)
            rec['d_plan_before'] = plan_distance(env, inst_idx, P_before)
        if P_after is not None:
            rec['P_after'] = plan_to_dict(P_after)
            rec['d_plan_after'] = plan_distance(env, inst_idx, P_after)
        if reject_reasons:
            rec['reject_reasons'] = reject_reasons
        self.decision_log.append(rec)

    def _build_input(self, ctx, rows, ds):
        """把候选动作/局部特征拼成 [1, M, 70]，与离线 build_dataset 逐字段一致。"""
        action_vals = np.stack([r[1] for r in rows])[None]
        action_valid = np.stack([r[2] for r in rows])[None]
        lv = np.stack([r[3] for r in rows])[None]
        lvd = np.stack([r[4] for r in rows])[None]
        ctx_b = np.broadcast_to(ctx[None, None, :], (1, action_vals.shape[1], ctx.shape[0]))
        import jax.numpy as jnp
        return jnp.concatenate([
            jnp.asarray(ctx_b, jnp.float32),
            jnp.asarray(action_vals, jnp.float32),
            jnp.asarray(action_valid.astype(np.float32), jnp.float32),
            jnp.asarray(lv, jnp.float32),
            jnp.asarray(lvd.astype(np.float32), jnp.float32)], axis=-1)

    def _process(self, env, inst_idx, clock, vehicles, served_mask, P, ctx, ds, customer,
                 mutable_ids):
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

        # (candidate_or_None, action_vals, action_valid, local_vals, local_valid, is_incumbent, is_defer)
        rows = []
        for c in feasible:
            a = self._action_dict(c.action)
            av, avd = extract_action_features({'action': a, 'is_pseudo': c.action.incumbent},
                                              deindex=self.deindex)
            lv, lvd = extract_candidate_local_features(ds, 0, a, c.incremental_distance,
                                                       self.capacity)
            rows.append((c, av, avd, lv, lvd, c.action.incumbent, False))
        incumbent = next((r for r in rows if r[5]), None)
        if incumbent is None:
            av, avd = extract_action_features({
                'is_pseudo': True, 'pseudo': 'DEFER',
                'action': {'customer': customer, 'kind': 'defer'}}, deindex=self.deindex)
            lv, lvd = extract_candidate_local_features(ds, 0, {'customer': customer, 'kind': 'defer'},
                                                       None, self.capacity)
            rows.append((None, av, avd, lv, lvd, False, True))
        self.timing['input_prep'] += time.time() - t_prep

        t_score = time.time()
        x = self._build_input(ctx, rows, ds)
        scores = np.asarray(self.scorer.score(x))[0]      # [M]
        if not np.isfinite(scores).all():
            self.online_stats['n_unexpected_error'] += 1
            raise RuntimeError("评分含 NaN/Inf")
        self.timing['model_score'] += time.time() - t_score
        self.online_stats['n_score'] += len(rows)

        keep_score = None
        best_idx, best_score = None, float('-inf')
        for i, r in enumerate(rows):
            sc = float(scores[i])
            if r[5]:
                keep_score = sc
            elif r[6] and keep_score is None:
                keep_score = sc
            if sc > best_score:
                best_score, best_idx = sc, i
        if keep_score is None:
            self.online_stats['n_unexpected_error'] += 1
            raise RuntimeError("KEEP/DEFER 基准分数缺失")

        # 接受规则：g = s*(f(a)-f(keep)) > tau（τ=∞ 恒不接受，等价只保持 baseline）
        g = self.scorer.s * (best_score - keep_score)
        best_row = rows[best_idx]
        if best_row[5] or best_row[6] or g <= self.tau:
            if best_row[6]:
                self.online_stats['n_defer'] += 1
            else:
                self.online_stats['n_keep'] += 1
            self._log_decision(env, inst_idx, clock, customer, 'score_reject',
                               g_hat=float(g), best_score=float(best_score),
                               keep_score=float(keep_score))
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
            reasons = {}
            for k, v in detail.items():
                if v:
                    self.online_stats['reject_reasons'][k] = \
                        self.online_stats['reject_reasons'].get(k, 0) + 1
                    reasons[k] = v
            self._log_decision(env, inst_idx, clock, customer, 'cert_reject',
                               action=self._action_dict(best_row[0].action),
                               g_hat=float(g), best_score=float(best_score),
                               keep_score=float(keep_score), reject_reasons=reasons)
            return False
        self.deferred_customers.discard(int(best_row[0].action.customer))
        # 空计划写回语义与 teacher/JF1-H-F 一致：非仓库且无客户且有未来 reveal → WAIT，
        # 否则 [0] 返仓。不能一律写 []。
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = new_plan.get(v.vehicle_id)
            if p is None:
                continue
            v.mutable_suffix = (list(p.suffix) + [0] if p.suffix
                                else ([] if (p.anchor_node != 0 and has_future) else [0]))
        self.timing['cert'] += time.time() - t_cert
        self.online_stats['n_accept'] += 1
        self._accepts_total += 1
        self._accepts_this_event += 1
        self._log_decision(env, inst_idx, clock, customer, 'accept',
                           action=self._action_dict(best_row[0].action),
                           g_hat=float(g), best_score=float(best_score),
                           keep_score=float(keep_score), P_before=P, P_after=new_plan)
        return True

    def _universe(self, env, inst_idx, clock, served_mask):
        return [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6]
