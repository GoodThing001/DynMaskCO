"""repair_state_v1：M-trained 的统一输入 schema 与单一提取函数。

训练、部署、回放共用 `extract_repair_state_v1`。输入只接收公开状态、部分计划、待修复集合、
合法动作与**允许车辆集合**；输出模型张量 + 索引映射。完整环境快照只用于恢复/评价。

原则：
  - 512 维 encoder 只吃 coord+demand（raw_3d）；TW/冷链/车队经 c_adapter 的 node_feats 进入。
  - 车辆 ID/订单 ID 只作关联索引，不作可学习身份特征。
  - 尺度来自公开合同或 TRAIN 固定统计，不随隐藏未来变化。
  - 缺失字段用有效位区分，不与 depot=0 混同。
  - 枚举、特征提取、日志、动作应用共用同一 allowed_vehicle_ids 与目标车辆解析。

字段 schema（v1）：

  nodes（每可见节点，F_NODE=10，归一化）：
    [0] tw_start/tw_max  [1] tw_end/tw_max  [2] service_time/tw_max  [3] temp_class(0/1/2)
    [4] initial_quality  [5] is_depot  [6] masked  [7] served  [8] committed  [9] mutable

  action_feats（每合法动作，F_ACTION_EXPLICIT=20）：
    [0:13] 目标车辆 X_state（anchor_time, remaining_capacity, temp3, zone_load3, zone_quality3,
           state_valid, has_cargo）
    [13] incremental_distance
    [14] tw_slack（+[17] valid）  [15] cap_slack（+[18] valid）  [16] return_slack（+[19] valid）

  cargo（订单级，[C_max, F_CARGO=6] + cargo_valid [C_max]）：
    [0] vehicle_id（关联索引）[1] quantity  [2] temp_class  [3] quality_remaining
    [4] initial_quality  [5] exposure_time（取货后）
"""
import copy

import numpy as np
import jax.numpy as jnp

from cvrptw_utils import coord_normalize_visible


def freeze_vehicles(vehicles):
    """深拷贝车辆状态列表（含 cargo/coldchain 等可变字段），隔离模拟器后续修改。

    训练采集器必须在决策时刻保存独立快照，否则模拟器推进后会原地修改 VehicleState 的
    committed_next / current_node / coldchain_state 等，导致网络输入与奖励评价状态错位。
    """
    return copy.deepcopy(vehicles)

F_NODE = 10
F_ACTION_EXPLICIT = 20
F_CARGO = 6
MAX_LOTS = 8


def _target_vehicle_state(vehicle, plan, capacity, tw_max):
    """目标车辆 X_state（13 维，语义同 extract_vehicle_cargo_state）。"""
    anchor_time = float(plan.anchor_time) / max(tw_max, 1e-6)
    rem_cap = (capacity - float(plan.anchor_load)) / max(capacity, 1e-6)
    cc = getattr(vehicle, 'coldchain_state', None)
    if cc is None:
        return np.concatenate([[anchor_time, rem_cap], np.zeros(9, np.float32),
                               [0.0, 0.0]]).astype(np.float32)
    temp = [float(x) / 40.0 for x in (getattr(cc, 'compartment_temperature_c', None) or (0.0, 0.0, 0.0))]
    zload = [float(x) / max(capacity, 1e-6) for x in (getattr(cc, 'zone_load', None) or (0.0, 0.0, 0.0))]
    zqual = [0.0, 0.0, 0.0]
    has_cargo = 0.0
    for lot in (getattr(cc, 'cargo_manifest', None) or ()):
        z = int(getattr(lot, 'temp_class', 0))
        init = max(float(getattr(lot, 'initial_quality', 1.0)), 1e-9)
        zqual[z] += float(getattr(lot, 'quantity', 0.0)) * (float(getattr(lot, 'quality_remaining', 0.0)) / init)
        has_cargo = 1.0
    zqual = [q / max(capacity, 1e-6) for q in zqual]
    return np.asarray([anchor_time, rem_cap] + temp + zload + zqual
                      + [1.0, has_cargo], np.float32)


def _cargo_table(vehicle):
    """订单级 cargo 表：[MAX_LOTS, F_CARGO] + valid。"""
    rows = np.zeros((MAX_LOTS, F_CARGO), np.float32)
    valid = np.zeros(MAX_LOTS, bool)
    cc = getattr(vehicle, 'coldchain_state', None)
    for i, lot in enumerate(getattr(cc, 'cargo_manifest', None) or ()):
        if i >= MAX_LOTS:
            break
        rows[i] = [float(vehicle.vehicle_id),
                   float(getattr(lot, 'quantity', 0.0)),
                   float(getattr(lot, 'temp_class', 0)),
                   float(getattr(lot, 'quality_remaining', 0.0)),
                   float(getattr(lot, 'initial_quality', 0.0)),
                   float(getattr(lot, 'exposure_time', 0.0))]
        valid[i] = True
    return rows, valid


def extract_repair_state_v1(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                            plans, mask_set, legal_actions, capacity, tw_max,
                            allowed_vehicle_ids=None,
                            node_to_local=None, local_to_node=None, Nv=None):
    """生成 repair_state_v1 张量。allowed_vehicle_ids 与 apply_action 的 mutable_ids 一致，
    目标车辆解析用它，保证「特征读哪辆车 = 动作应用到哪辆车」。"""
    from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency, resolve_action_endpoints
    from coldchain_visible_features import resolve_target_vid

    if node_to_local is None:
        node_to_local, local_to_node, Nv = build_visible_index(visible_ids)

    # raw_3d（encoder 输入）
    coords = env.coords[inst_idx][np.asarray(local_to_node, dtype=np.int32)]
    demands = env.demands[inst_idx][np.asarray(local_to_node, dtype=np.int32)]
    coords_n = np.asarray(coord_normalize_visible(jnp.array(coords, dtype=jnp.float32), None))
    raw_3d = np.concatenate([coords_n, (demands / max(capacity, 1e-6))[..., None]], axis=-1)[None]

    # node_feats（c_adapter 输入）
    committed = {int(v.committed_next) for v in vehicles
                 if getattr(v, 'status', None) == 'committed' and v.committed_next not in (None, 0)}
    mutable_cust = {int(c) for c in visible_ids
                    if int(c) != 0 and not served_mask[int(c)] and int(c) not in committed}
    node_feats = np.zeros((1, Nv, F_NODE), np.float32)
    for li, nid in enumerate(local_to_node):
        if nid == 0:
            node_feats[0, li, 5] = 1.0
            continue
        node_feats[0, li, 0] = (float(env.tw_start[inst_idx, nid]) - clock) / max(tw_max, 1e-6)
        node_feats[0, li, 1] = (float(env.tw_end[inst_idx, nid]) - clock) / max(tw_max, 1e-6)
        node_feats[0, li, 2] = float(env.service_time[inst_idx, nid]) / max(tw_max, 1e-6)
        node_feats[0, li, 3] = float(env.temp_class[inst_idx, nid])
        node_feats[0, li, 4] = float(env.initial_quality[inst_idx, nid])
        if int(nid) in mask_set:
            node_feats[0, li, 6] = 1.0
        if served_mask[int(nid)]:
            node_feats[0, li, 7] = 1.0
        if int(nid) in committed:
            node_feats[0, li, 8] = 1.0
        if int(nid) in mutable_cust:
            node_feats[0, li, 9] = 1.0

    node_valid = np.ones((1, Nv), bool)
    adjmat = build_plan_adjacency(plans, node_to_local, Nv)[None]

    # action tensors
    M = len(legal_actions)
    cust = np.zeros((1, M), np.int32)
    pred = np.zeros((1, M), np.int32)
    succ = np.zeros((1, M), np.int32)
    action_valid = np.zeros((1, M), bool)
    action_feats = np.zeros((1, M, F_ACTION_EXPLICIT), np.float32)
    vehicle_target_vid = [None] * M
    for m, (c, a, cand) in enumerate(legal_actions):
        cust[0, m] = node_to_local[int(c)]
        e, v = resolve_action_endpoints(a.customer, a.slot, a.predecessor, a.successor,
                                        plans, node_to_local)
        pred[0, m] = int(e[2])
        succ[0, m] = int(e[3])
        action_valid[0, m] = True
        vid = resolve_target_vid({'customer': int(c), 'slot_kind': a.slot.kind,
                                  'slot_anchor': int(a.slot.anchor)}, plans,
                                 allowed_vehicle_ids=allowed_vehicle_ids)
        vehicle_target_vid[m] = vid
        if vid is not None:
            vp = plans.get(vid)
            vobj = vehicles[vid] if vid < len(vehicles) else None
            action_feats[0, m, :13] = _target_vehicle_state(vobj, vp, capacity, tw_max)
        action_feats[0, m, 13] = float(getattr(cand, 'incremental_distance', 0.0) or 0.0)
        action_feats[0, m, 14] = 0.0 if getattr(cand, 'tw_slack', None) is None else float(cand.tw_slack)
        action_feats[0, m, 15] = 0.0 if getattr(cand, 'cap_slack', None) is None else float(cand.cap_slack)
        action_feats[0, m, 16] = 0.0 if getattr(cand, 'return_slack', None) is None else float(cand.return_slack)
        action_feats[0, m, 17] = 0.0 if getattr(cand, 'tw_slack', None) is None else 1.0
        action_feats[0, m, 18] = 0.0 if getattr(cand, 'cap_slack', None) is None else 1.0
        action_feats[0, m, 19] = 0.0 if getattr(cand, 'return_slack', None) is None else 1.0

    # cargo 表（订单级，每车 MAX_LOTS 笔 + 有效位）
    cargo = np.zeros((len(vehicles), MAX_LOTS, F_CARGO), np.float32)
    cargo_valid = np.zeros((len(vehicles), MAX_LOTS), bool)
    for vi, v in enumerate(vehicles):
        rows, valid = _cargo_table(v)
        cargo[vi] = rows
        cargo_valid[vi] = valid

    return {
        'raw_3d': raw_3d.astype(np.float32),
        'node_feats': node_feats.astype(np.float32),
        'node_valid': node_valid,
        'adjmat': adjmat.astype(np.float32),
        'node_to_local': node_to_local,
        'local_to_node': local_to_node,
        'Nv': Nv,
        'clock': float(clock),
        'cust': cust, 'pred': pred, 'succ': succ,
        'action_valid': action_valid, 'action_feats': action_feats.astype(np.float32),
        'vehicle_target_vid': vehicle_target_vid,
        'cargo': cargo, 'cargo_valid': cargo_valid,
    }
