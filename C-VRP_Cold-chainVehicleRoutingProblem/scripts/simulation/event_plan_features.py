"""event_plan_v1/2：完整计划评价器的输入特征（P vs P0 的两种表示）。

A（旧表示修正版）：`extract_plan_delta` —— 每辆车的 [anchor, next, n_suffix, changed] 有序
路线摘要（changed = 相对 P0 的 anchor/suffix 是否变化）。物理车辆 ID 不作为学习特征
（车辆按 vid 固定位置，等价车由 anchor/next/路线表达）。节点编号不能表达地理/时间关系。

B（可见执行后果表示）：`extract_plan_consequence` —— 每辆车的可见状态 + 有序路线时间/资源
投影 + 冷链 D/Q/E/J 代理及其相对 KEEP 的差值。投影由 event_plan_projection 提供（C0 转移，
不读未来）。

两种表示都返回「P 相对 P0 的候选特征向量」；模型用 f([ctx, feat(P0,P)]) - f([ctx, feat(P0,P0)])
的结构，与旧 PlanEvalMLP 相同。B 的字段顺序 / 有效位 / 归一化在训练前一次冻结。
"""
import numpy as np

from event_plan_projection import project_plan, proxy_cost


PLAN_VEH_DIM = 4  # A：anchor / next / n_suffix / changed

# B 每车 21 维
B_VEH_DIM = 21
# B 车队汇总 4 维
B_FLEET_DIM = 4


def _A_dim(num_vehicles):
    return num_vehicles * PLAN_VEH_DIM


def _B_dim(num_vehicles):
    return num_vehicles * B_VEH_DIM + B_FLEET_DIM


def plan_dim(representation, num_vehicles):
    if representation == 'A':
        return _A_dim(num_vehicles)
    if representation == 'B':
        return _B_dim(num_vehicles)
    raise ValueError(f"unknown representation {representation!r}")


def extract_plan_delta(env, inst_idx, P0, P, num_vehicles):
    """A 表示：返回 (num_vehicles * PLAN_VEH_DIM,) 的 plan-delta 特征。"""
    N = max(env.num_nodes, 1)
    feats = np.zeros((num_vehicles, PLAN_VEH_DIM), np.float32)
    for vid in range(num_vehicles):
        p0 = P0.get(vid)
        p = P.get(vid)
        anchor0 = int(p0.anchor_node) if p0 else 0
        anchor = int(p.anchor_node) if p else 0
        s0 = tuple(int(x) for x in p0.suffix) if p0 else ()
        s = tuple(int(x) for x in p.suffix) if p else ()
        nxt0 = next((x for x in s0 if x != 0), 0)
        nxt = next((x for x in s if x != 0), 0)
        n0 = sum(1 for x in s0 if x != 0)
        n = sum(1 for x in s if x != 0)
        changed = 1.0 if (anchor != anchor0 or s != s0) else 0.0
        feats[vid] = [anchor / N, nxt / N, min(n, 20) / 20.0, changed]
    return feats.reshape(-1)


def extract_plan_consequence(env, inst_idx, vehicles, P0, P, contract, profile,
                             capacity, num_vehicles, has_future):
    """B 表示：返回 (num_vehicles * B_VEH_DIM + B_FLEET_DIM,) 的特征向量。

    输入为可见状态（vehicles 来自 snapshot 恢复）与 P0/P；投影只推进当前已知计划。
    尺度来自公开合同 / depot horizon / capacity / objective profile，禁止未来统计。
    """
    horizon = float(env.tw_end[inst_idx].max())
    objective = profile.objective_config() if profile is not None else contract.objective
    ds = float(profile.distance_scale) if profile is not None else contract.objective.distance_scale
    qs = float(profile.quality_scale) if profile is not None else contract.objective.quality_scale
    es = float(profile.energy_scale) if profile is not None else contract.objective.energy_scale

    proj0 = project_plan(env, inst_idx, vehicles, P0, contract, has_future)
    projP = project_plan(env, inst_idx, vehicles, P, contract, has_future)
    j0 = {vid: proxy_cost(p, objective) for vid, p in proj0.items()}
    jP = {vid: proxy_cost(p, objective) for vid, p in projP.items()}

    coords0 = env.coords[inst_idx, 0]
    feats = np.zeros((num_vehicles, B_VEH_DIM), np.float32)
    for vid in range(num_vehicles):
        p = P.get(vid)
        if p is None:
            feats[vid, 0] = 0.0
            continue
        pp = projP.get(vid)
        p0 = proj0.get(vid)
        anchor = int(p.anchor_node)
        ax, ay = float(env.coords[inst_idx, anchor, 0]) - coords0[0], \
            float(env.coords[inst_idx, anchor, 1]) - coords0[1]

        def _d(key):
            a = pp[key] if pp is not None else 0.0
            b = p0[key] if p0 is not None else 0.0
            return float(a) - float(b)

        dD = _d('d_distance')
        dQ = _d('d_quality')
        dE = _d('d_energy')
        dJ = float(jP.get(vid, 0.0) - j0.get(vid, 0.0))

        slack = float(pp['min_tw_slack']) / max(horizon, 1e-6) if pp is not None else 0.0
        slack = float(np.clip(slack, -1.0, 1.0))
        feats[vid] = [
            1.0,                                   # 0 valid
            float(ax), float(ay),                  # 1,2 anchor 相对 depot 坐标
            float(p.anchor_time) / max(horizon, 1e-6),          # 3 anchor_time
            float(p.anchor_load) / max(capacity, 1e-6),         # 4 anchor_load
            float(pp['is_committed']) if pp is not None else 0.0,  # 5 is_committed
            min(len([x for x in p.suffix if int(x) != 0]), 20) / 20.0,  # 6 suffix_len
            float(pp['has_cargo']) if pp is not None else 0.0,  # 7 has_cargo
            min(float(pp['cargo_count']), 20) / 20.0 if pp is not None else 0.0,  # 8
            float(pp['cargo_age']) / max(horizon, 1e-6) if pp is not None else 0.0,  # 9
            float(pp['cargo_quality']) if pp is not None else 1.0,  # 10
            float(pp['distance']) / max(horizon, 1e-6) if pp is not None else 0.0,  # 11
            float(pp['return_time']) / max(horizon, 1e-6) if pp is not None else 0.0,  # 12
            slack,                                 # 13 min_tw_slack
            float(pp['max_load']) / max(capacity, 1e-6) if pp is not None else 0.0,  # 14
            float(pp['wait_time']) / max(horizon, 1e-6) if pp is not None else 0.0,  # 15
            float(pp['open_plan']) if pp is not None else 0.0,  # 16 WAIT
            float(dD) / max(ds, 1e-6),             # 17 dD
            float(dQ) / max(qs, 1e-6),             # 18 dQ
            float(dE) / max(es, 1e-6),             # 19 dE
            float(dJ),                             # 20 dJ
        ]

    # 车队汇总
    fleet = np.zeros(B_FLEET_DIM, np.float32)
    for vid in range(num_vehicles):
        p = P.get(vid)
        if p is None:
            continue
        pp = projP.get(vid)
        p0 = proj0.get(vid)
        if pp is None or p0 is None:
            continue
        fleet[0] += float(proxy_cost(pp, objective)) - float(proxy_cost(p0, objective))
        fleet[1] += float(pp['d_distance'] - p0['d_distance']) / max(ds, 1e-6)
        fleet[2] += float(pp['d_quality'] - p0['d_quality']) / max(qs, 1e-6)
        fleet[3] += float(pp['d_energy'] - p0['d_energy']) / max(es, 1e-6)

    return np.concatenate([feats.reshape(-1), fleet]).astype(np.float32)
