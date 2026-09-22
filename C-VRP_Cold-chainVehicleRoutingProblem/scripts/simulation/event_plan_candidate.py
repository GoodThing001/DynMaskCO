"""event_plan_v1/2：事件级完整计划候选生成与反事实标签。

G_simple_r1：从 baseline incumbent P_0 的独立副本出发，按固定、可复现的 7 个请求生成
≤8 个完整计划（含 KEEP）。每个请求绑定不同的 mask/顺序/插入策略；组合请求使用不同客户子集，
单个请求覆盖不同客户。允许从可见插入代价 top-3 中确定性选取，保留一个纯最小代价请求作对照。
生成不读取 g_B。

v2 修正：
  - 每个成功计划绑定自己的 request_id / requested_mask_size / selected_customer_ids /
    applied_edit_customer_ids（不再 records/plans 直接 zip 错位）；
  - `seen` 初始包含 KEEP 计划哈希，KEEP-identical 候选被去重剔除（不再带 25 条零增益克隆）；
  - `plan_distance` 计入末尾返仓段，并区分 WAIT（不虚构即时返仓）与 RETURN。
"""
import hashlib
import json

from action_contract import (enumerate_actions_from_plans, apply_action)


def full_plan_dict(plans, deferred, has_future=False):
    return {
        'plans': {int(vid): {'anchor_node': int(p.anchor_node),
                             'anchor_time': float(p.anchor_time),
                             'anchor_load': float(p.anchor_load),
                             'suffix': [int(x) for x in p.suffix]}
                  for vid, p in sorted(plans.items())},
        'deferred': sorted(int(x) for x in deferred),
        'has_future': bool(has_future),
    }


def full_plan_hash(plans, deferred, has_future=False):
    payload = json.dumps(full_plan_dict(plans, deferred, has_future), sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def plan_distance(env, inst_idx, plans, has_future=None):
    """完整车队计划距离（anchor → suffix → 返仓 0）。

    suffix 尾部不保证含 0（build_vehicle_plans 已剔除尾 0），这里**补上返仓段**：
      - 非空 suffix：anchor→suffix...→last，再 last→0；
      - 空 suffix：WAIT（anchor≠0 且 has_future）不计返仓；否则 anchor→0 即时返仓。
    has_future=None 时按「非空即返仓、空即 WAIT 若 anchor≠0」的默认（调用方应显式传入）。
    """
    total = 0.0
    for p in plans.values():
        prev = int(p.anchor_node)
        for c in p.suffix:
            total += float(env.dist_mat[inst_idx, prev, int(c)])
            prev = int(c)
        if p.suffix:
            total += float(env.dist_mat[inst_idx, prev, 0])
        else:
            if has_future is None:
                wait = (p.anchor_node != 0)
            else:
                wait = (p.anchor_node != 0 and has_future)
            if not wait:
                total += float(env.dist_mat[inst_idx, int(p.anchor_node), 0])
    return total


def _customer_positions(plans):
    """customer -> (vid, position_in_suffix)；用于判断是否真正被移动。"""
    pos = {}
    for vid, p in plans.items():
        for idx, c in enumerate(p.suffix):
            if int(c) != 0:
                pos[int(c)] = (int(vid), idx)
    return pos


def _plan_copy(P0):
    return {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
            for vid, p in P0.items()}


def _sorted_pool(env, inst_idx, pool):
    return sorted(pool, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c)))


def _pick_action(cands, seed, request_id, pure_min_cost):
    """从 feasible 非 incumbent 候选中按固定规则选一个。"""
    feas = [c for c in cands if c.feasible and not c.action.incumbent]
    if not feas:
        return None
    feas.sort(key=lambda c: (c.incremental_distance, c.action.action_id()))
    if pure_min_cost or len(feas) <= 1:
        return feas[0]
    top = feas[:3]
    idx = (seed * 31 + request_id * 7 + len(top)) % len(top)
    return top[idx]


def generate_simple_candidates_r1(env, inst_idx, P0, pool, mutable_ids, seed=0,
                                  has_future=False):
    """G_simple_r1：7 个请求（1,1,1,2,2,4,4）。

    返回 (records, plans)：
      records — 全部 7 次请求尝试日志（含失败/去重原因），逐条含 request_id；
      plans   — 去重后的成功候选计划列表，每项 dict 绑定自己的
                candidate_id / request_id / requested_mask_size / selected_customer_ids /
                applied_edit_customer_ids / plan。
    候选不含与 KEEP 完全相同的计划（seen 初始含 KEEP 哈希）。
    """
    order = _sorted_pool(env, inst_idx, pool)
    requests = [
        (1, [0], True),            # 纯最小代价对照
        (1, [1], False),
        (1, [2], False),
        (2, [0, 1], False),
        (2, [2, 3], False),
        (4, [0, 1, 2, 3], False),
        (4, [4, 5, 6, 7], False),
    ]
    records = []
    plans = []
    seen = {full_plan_hash(P0, set(), has_future)}  # KEEP 入 seen → 剔除 KEEP-identical 候选
    for req_id, (k, sel_idx, pure) in enumerate(requests):
        sel = [order[i] for i in sel_idx if i < len(order)]
        if len(sel) < min(k, len(order)):
            records.append({'request_id': req_id, 'requested_mask_size': k,
                            'selected_customer_ids': [], 'applied_edit_customer_ids': [],
                            'ok': False, 'fail_reason': 'pool_too_small', 'candidate_id': None})
            continue
        P = _plan_copy(P0)
        before_pos = _customer_positions(P)
        applied = []
        ok = True
        for customer in sel:
            cands, _ = enumerate_actions_from_plans(env, inst_idx, P, customer,
                                                    allowed_vehicle_ids=mutable_ids)
            act = _pick_action(cands, seed, req_id, pure)
            if act is None:
                ok = False
                break
            P = apply_action(P, act.action, allowed_vehicle_ids=mutable_ids)
        if ok:
            after_pos = _customer_positions(P)
            for customer in sel:
                if before_pos.get(customer) != after_pos.get(customer):
                    applied.append(int(customer))
        h = full_plan_hash(P, set(), has_future) if ok else None
        dup = ok and h in seen
        records.append({'request_id': req_id, 'requested_mask_size': k,
                        'selected_customer_ids': [int(c) for c in sel],
                        'applied_edit_customer_ids': applied if ok else [], 'ok': ok,
                        'fail_reason': (None if ok else 'no_legal_insertion'),
                        'candidate_id': h, 'duplicate': dup})
        if ok and not dup:
            seen.add(h)
            plans.append({'candidate_id': h, 'request_id': req_id,
                          'requested_mask_size': k,
                          'selected_customer_ids': [int(c) for c in sel],
                          'applied_edit_customer_ids': applied, 'plan': P})
    return records, plans


def dict_to_plans(d):
    from action_contract import VehiclePlan
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), float(v['anchor_time']),
                                  float(v['anchor_load']), tuple(int(x) for x in v['suffix']))
            for vid, v in d['plans'].items()}
