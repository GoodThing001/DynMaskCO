"""event_plan_v1：完整计划评价器的输入特征（P vs P0 的 plan-delta 表示）。

context（共享）复用现有 feature_only_context 的 25 维可见状态；plan-delta（每候选）是每辆车的
[anchor, next, n_suffix, changed] 有序路线摘要（changed = 相对 P0 的 anchor/suffix 是否变化）。
物理车辆 ID 不作为学习特征（车辆按 vid 固定位置，等价车由 anchor/next/路线表达）。
"""
import numpy as np


PLAN_VEH_DIM = 4  # anchor / next / n_suffix / changed


def extract_plan_delta(env, inst_idx, P0, P, num_vehicles):
    """返回 (num_vehicles * PLAN_VEH_DIM,) 的 plan-delta 特征。"""
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
