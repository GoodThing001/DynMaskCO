"""event_plan_v1：事件级完整计划候选生成与反事实标签（EP-v1.1）。

G_simple_r1：从 baseline incumbent P_0 的独立副本出发，按固定、可复现的 7 个请求生成
≤8 个完整计划（含 KEEP）。每个请求绑定不同的 mask/顺序/插入策略；组合请求使用不同客户子集，
单个请求覆盖不同客户。允许从可见插入代价 top-3 中确定性选取，保留一个纯最小代价请求作对照。
生成不读取 g_B。

记录 requested_mask_size / selected_customer_ids / applied_edit_customer_ids（真正被移动、
而非数组 index 因前方插入被动推移的客户）与去重/失败原因。
"""
import hashlib
import json

from action_contract import (enumerate_actions_from_plans, apply_action)


def full_plan_dict(plans, deferred):
    return {
        'plans': {int(vid): {'anchor_node': int(p.anchor_node),
                             'anchor_time': float(p.anchor_time),
                             'anchor_load': float(p.anchor_load),
                             'suffix': [int(x) for x in p.suffix]}
                  for vid, p in sorted(plans.items())},
        'deferred': sorted(int(x) for x in deferred),
    }


def full_plan_hash(plans, deferred):
    payload = json.dumps(full_plan_dict(plans, deferred), sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def plan_distance(env, inst_idx, plans):
    """完整车队计划距离（anchor → suffix，suffix 尾部常含 0 返仓）。"""
    total = 0.0
    for p in plans.values():
        prev = int(p.anchor_node)
        for c in p.suffix:
            total += float(env.dist_mat[inst_idx, prev, int(c)])
            prev = int(c)
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
    """从 feasible 非 incumbent 候选中按固定规则选一个。

    pure_min_cost=True 时选最小 incremental_distance（对照请求）；
    否则在按 (incremental_distance, action_id) 排序的前 3 个里用 seed 确定选取。
    """
    feas = [c for c in cands if c.feasible and not c.action.incumbent]
    if not feas:
        return None
    feas.sort(key=lambda c: (c.incremental_distance, c.action.action_id()))
    if pure_min_cost or len(feas) <= 1:
        return feas[0]
    top = feas[:3]
    idx = (seed * 31 + request_id * 7 + len(top)) % len(top)
    return top[idx]


def generate_simple_candidates_r1(env, inst_idx, P0, pool, mutable_ids, seed=0):
    """G_simple_r1：7 个请求（1,1,1,2,2,4,4），返回 (records, plans)。

    records 每项含 requested_mask_size / selected_customer_ids / applied_edit_customer_ids /
    ok / fail_reason；plans 为去重后的候选计划列表（不含 KEEP）。
    """
    order = _sorted_pool(env, inst_idx, pool)
    # 7 个请求的 mask 定义（客户下标）；组合请求用不同客户子集
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
    seen = set()
    for req_id, (k, sel_idx, pure) in enumerate(requests):
        sel = [order[i] for i in sel_idx if i < len(order)]
        if len(sel) < min(k, len(order)):
            records.append({'requested_mask_size': k, 'selected_customer_ids': [],
                            'applied_edit_customer_ids': [], 'ok': False,
                            'fail_reason': 'pool_too_small'})
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
        records.append({'requested_mask_size': k,
                        'selected_customer_ids': [int(c) for c in sel],
                        'applied_edit_customer_ids': applied, 'ok': ok,
                        'fail_reason': None if ok else 'no_legal_insertion'})
        if ok:
            h = full_plan_hash(P, set())
            if h not in seen:
                seen.add(h)
                plans.append(P)
    return records, plans


def dict_to_plans(d):
    from action_contract import VehiclePlan
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), float(v['anchor_time']),
                                  float(v['anchor_load']), tuple(int(x) for x in v['suffix']))
            for vid, v in d['plans'].items()}
