"""统一逐步动作策略（M-pre / M-trained 共用）+ 采样/可微回放分离 + 完整计划对账。

采样与可微回放分开：`stepwise_sample` 用 NumPy 采样并记录每步状态 + 选中索引；
`replay_log_prob` 用相同 JAX score_fn 重算所选动作 log-prob（保留梯度）。
训练器据此在参数更新前回放，一批只更新一次。

- 无合法动作/数值异常：保留已采样前缀与失败原因（有限失败奖励可用）；数值异常显式标 'nan'。
- 完整修复 log-prob = 各步选中动作 log-prob 之和。
- 动作回放记录解析后的真实 vehicle_id（非 slot.anchor）+ slot_anchor。
"""
import numpy as np
import jax
import jax.numpy as jnp

from action_contract import enumerate_actions_from_plans, apply_action


def enumerate_legal_actions(env, inst_idx, plans, remaining, mutable_ids):
    legal = []
    for c in remaining:
        cands, _ = enumerate_actions_from_plans(env, inst_idx, plans, int(c),
                                                allowed_vehicle_ids=mutable_ids)
        for cand in cands:
            if cand.feasible:
                legal.append((int(c), cand.action, cand))
    return legal


def _plan_copy(plans):
    return {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
            for vid, p in plans.items()}


def stepwise_sample(score_fn, extract_fn, env, inst_idx, plans, mask_set, mutable_ids,
                    temperature=1.0, rng=None, deterministic=False):
    """采样（或确定性 argmax）完整修复。score_fn(state)->scores[M]，extract_fn->state dict。

    deterministic=True 时取 argmax（评价用）；否则按 softmax 采样（训练用）。
    Returns (plan, steps, records, failure_reason)。failure_reason ∈ {None, 'no_legal', 'nan'}。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    P = _plan_copy(plans)
    remaining = list(mask_set)
    records = []
    steps = []
    while remaining:
        legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
        if not legal:
            return P, steps, records, 'no_legal'
        state = extract_fn(P, remaining, legal)
        scores = np.asarray(score_fn(state))
        if not np.isfinite(scores).all():
            return P, steps, records, 'nan'
        if deterministic:
            idx = int(np.argmax(scores))
        else:
            logits = scores / max(temperature, 1e-9)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            idx = int(rng.choice(len(legal), p=probs))
        records.append({'state': state, 'chosen': idx, 'temp': float(temperature)})
        c, a, _cand = legal[idx]
        vid = state.get('vehicle_target_vid', [None] * len(legal))[idx]
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        steps.append({'customer': int(c), 'vehicle_id': vid, 'slot_kind': a.slot.kind,
                      'slot_anchor': int(a.slot.anchor), 'position': int(a.position),
                      'predecessor': int(a.predecessor), 'successor': int(a.successor)})
        remaining.remove(c)
    return P, steps, records, None


def replay_log_prob(score_fn, records):
    """用相同参数重算所选动作的 JAX log-prob（可微）。返回 JAX 标量。"""
    total = jnp.array(0.0)
    for rec in records:
        scores = score_fn(rec['state'])                       # JAX [M]
        lp = jax.nn.log_softmax(scores / rec['temp'], axis=-1)
        total = total + lp[rec['chosen']]
    return total


def validate_repair_plan(plans, base_plans, mask_set, protected_vehicle_ids):
    """完整计划对账（不用客户集合代替）。返回 (ok, detail)。

    检查：车辆集合 + anchor 不变；无重复；原客户全集恰好保留（无缺失/无额外）；
    非 mask 客户的车辆归属与相对顺序不变；保护车辆完整计划不变。
    """
    detail = {'dup': [], 'missing': [], 'extra': [], 'order_changed': [],
              'protected_vehicle_changed': [], 'vehicle_changed': []}
    # 1. 车辆集合 + anchor 不变
    if set(plans.keys()) != set(base_plans.keys()):
        detail['vehicle_changed'].append('keyset')
    for vid in base_plans:
        if vid not in plans:
            detail['vehicle_changed'].append(vid)
            continue
        bp, p = base_plans[vid], plans[vid]
        if (p.anchor_node != bp.anchor_node or p.anchor_time != bp.anchor_time
                or p.anchor_load != bp.anchor_load):
            detail['vehicle_changed'].append(vid)
    # 2. exact-once（无重复）
    seen = {}
    for vid, p in plans.items():
        for x in p.suffix:
            x = int(x)
            if x <= 0:
                continue
            if x in seen:
                detail['dup'].append(x)
            seen[x] = vid
    # 3. 原客户全集恰好保留
    base_customers = {int(x) for p in base_plans.values() for x in p.suffix if int(x) > 0}
    cur_customers = set(seen.keys())
    detail['missing'] = sorted(base_customers - cur_customers)
    detail['extra'] = sorted(cur_customers - base_customers)
    # 4. 非 mask 客户的归属与相对顺序不变
    immutable = base_customers - set(mask_set)
    for vid in base_plans:
        if vid not in plans:
            continue
        base_suffix = [int(x) for x in base_plans[vid].suffix if int(x) > 0]
        cur_suffix = [int(x) for x in plans[vid].suffix if int(x) > 0]
        if [x for x in base_suffix if x in immutable] != [x for x in cur_suffix if x in immutable]:
            detail['order_changed'].append(vid)
    # 5. 保护车辆完整计划不变
    for vid in protected_vehicle_ids:
        if vid in plans and vid in base_plans:
            if list(plans[vid].suffix) != list(base_plans[vid].suffix):
                detail['protected_vehicle_changed'].append(vid)
    ok = not any(detail.values())
    return ok, detail
