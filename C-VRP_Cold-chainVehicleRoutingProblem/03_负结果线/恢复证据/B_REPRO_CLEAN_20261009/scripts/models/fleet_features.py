"""
JF2 特征构造（Step 8）— 从 strict-online FleetState 算 vehicle / pair 特征 + JF1-H base score。

纯 NumPy，无 JAX / model，可本地测。输入/输出形状为单实例（B=1，head 调用时再扩 batch 维）。

特征（主控文档 §11-13 / Part XVII）：
  veh_feat  [K, Fv=4]  : t_k/T, load_k/Q, remaining_cap_k/Q, status_k(0=idle,1=ready)
  pair_feat [K, N, Fp=12] : travel, ETA, wait, TWSlack, RelativeUrgency, ReturnSlack,
                            CapSlack, scarcity(m_j), coverage(n_k), TravelRegret,
                            SlackRegret, nPending(global)

base_score [K, N] = -travel（JF1-H min-travel，可行处），不可行处 -1e9。
candidate_mask [K, N] = sound hard mask（Step 5 的 build_sound_candidate_set）。

归一化：时间类 / T（horizon），负载类 / Q，计数类 / K 或 / N。
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from joint_fleet import build_sound_candidate_set, CandidateSet, get_fleet_anchors

Fv = 4
Fp = 12


@dataclass
class JF2Features:
    inst_idx: int
    clock: float
    candidate: CandidateSet                 # Step 5 候选集（customer_order/feasible/return_slack）
    active_vehicle_ids: List[int]           # [K] 升序
    customer_order: List[int]
    anchor_ids: np.ndarray                  # [K] int
    veh_feat: np.ndarray                    # [K, Fv]
    pair_feat: np.ndarray                   # [K, N, Fp]
    base_score: np.ndarray                  # [K, N]
    candidate_mask: np.ndarray              # [K, N] bool


def build_jf2_features(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                       replan_ids=None):
    """从 FleetState 构造 JF2 特征张量 + base_score + candidate_mask（单实例）。"""
    cs = build_sound_candidate_set(env, inst_idx, clock, vehicles, served_mask,
                                   visible_ids, replan_ids)
    K = len(cs.active_vehicle_ids)
    N = env.num_nodes
    Q = float(env.capacity)
    T = float(env.tw_end[inst_idx].max())
    eps = 1e-6

    vid_to_row = {vid: i for i, vid in enumerate(cs.active_vehicle_ids)}
    # anchor_ids [K], veh_feat [K, Fv]
    anchors = get_fleet_anchors(vehicles)
    status_by_vid = {a.vehicle_id: (0.0 if a.status == 'idle' else 1.0) for a in anchors}
    anchor_ids = np.zeros(K, dtype=np.int32)
    veh_feat = np.zeros((K, Fv), dtype=np.float32)
    for vid, (node, t, ld) in cs.anchor_info.items():
        if vid not in vid_to_row:
            continue
        i = vid_to_row[vid]
        anchor_ids[i] = node
        status = status_by_vid.get(vid, 0.0)
        veh_feat[i] = [t / T, ld / Q, max(0.0, Q - ld) / Q, status]

    # pair_feat [K, N, Fp]
    pair_feat = np.zeros((K, N, Fp), dtype=np.float32)
    tw_start = env.tw_start[inst_idx]
    tw_end = env.tw_end[inst_idx]
    service = env.service_time[inst_idx]
    dist = env.dist_mat[inst_idx]
    demand = env.demands[inst_idx]
    speed = env.tw_speed

    # 每个客户的可行车辆数 m_j + 每辆车的覆盖客户数 n_k + 每客户的 travel/slack 排序
    feasible_vehicles = cs.feasible_vehicles          # customer -> list[vehicle]
    m_j = {j: len(feasible_vehicles[j]) for j in cs.customer_order}
    n_k = {vid: 0 for vid in cs.active_vehicle_ids}
    for j in cs.customer_order:
        for vid in feasible_vehicles[j]:
            n_k[vid] = n_k.get(vid, 0) + 1

    # 预计算每个客户 j 的 travel 与 TWSlack 排序（用于 regret）
    def _travel(vid, j):
        node, t, _ = cs.anchor_info[vid]
        return float(dist[node, j]) / speed

    def _twslack(vid, j):
        node, t, _ = cs.anchor_info[vid]
        eta = t + float(dist[node, j]) / speed
        start = max(eta, float(tw_start[j]))
        return float(tw_end[j]) - start

    n_pending = len(cs.customer_order)
    n_active = K

    for j in cs.customer_order:
        jidx = int(j)
        feas = feasible_vehicles[j]
        travels = sorted((_travel(vid, j) for vid in feas), reverse=False)
        slacks = sorted((_twslack(vid, j) for vid in feas), reverse=True)  # slack 越大越好
        travel_regret = (travels[1] - travels[0]) if len(travels) >= 2 else 0.0
        slack_regret = (slacks[0] - slacks[1]) if len(slacks) >= 2 else 0.0
        a_j = float(tw_start[j]); b_j = float(tw_end[j]); s_j = float(service[j])
        tw_width = max(b_j - a_j, eps)

        for vid in cs.active_vehicle_ids:
            i = vid_to_row[vid]
            node, t_k, l_k = cs.anchor_info[vid]
            travel = float(dist[node, j]) / speed
            eta = t_k + travel
            wait = max(0.0, a_j - eta)
            start = max(eta, a_j)
            tw_slack = b_j - start
            rel_urgency = tw_slack / tw_width
            ready = start + s_j
            return_slack = float(tw_end[0]) - (ready + float(dist[j, 0]) / speed)
            cap_slack = Q - (l_k + float(demand[j]))

            pair_feat[i, jidx, 0] = travel / T
            pair_feat[i, jidx, 1] = eta / T
            pair_feat[i, jidx, 2] = wait / T
            pair_feat[i, jidx, 3] = tw_slack / T
            pair_feat[i, jidx, 4] = rel_urgency
            pair_feat[i, jidx, 5] = return_slack / T
            pair_feat[i, jidx, 6] = cap_slack / Q
            pair_feat[i, jidx, 7] = m_j[j] / max(1, K)
            pair_feat[i, jidx, 8] = n_k[vid] / max(1, N)
            pair_feat[i, jidx, 9] = travel_regret / T
            pair_feat[i, jidx, 10] = slack_regret / T
            pair_feat[i, jidx, 11] = n_pending / max(1, N)

    # base_score [K, N] + candidate_mask [K, N]
    base_score = np.full((K, N), -1e9, dtype=np.float32)
    candidate_mask = np.zeros((K, N), dtype=bool)
    for j in cs.customer_order:
        jidx = int(j)
        for vid in cs.feasible_vehicles[j]:
            i = vid_to_row[vid]
            node, _, _ = cs.anchor_info[vid]
            base_score[i, jidx] = -float(dist[node, j]) / speed
            candidate_mask[i, jidx] = True

    return JF2Features(inst_idx=int(inst_idx), clock=float(clock), candidate=cs,
                       active_vehicle_ids=cs.active_vehicle_ids,
                       customer_order=cs.customer_order, anchor_ids=anchor_ids,
                       veh_feat=veh_feat, pair_feat=pair_feat,
                       base_score=base_score, candidate_mask=candidate_mask)
