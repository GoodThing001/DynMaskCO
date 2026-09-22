"""M0/M1 可见特征提取器（路线 B / 工作包 A）。

按信息边界构造模型输入张量，只使用决策时刻已揭示的信息；完整 snapshot 不整体展开送入
网络。三类特征：

  - 订单：已揭示订单的坐标/需求/时间窗/服务/温区/初始品质，未揭示置 0 并由 node_visible
    区分（不泄露未来节点数量或坐标）。
  - 车队：当前车辆 anchor/ready/load/status/needs_replan/committed leg/mutable suffix/
    舱温/zone load 汇总（不带未来轨迹）。
  - 动作：customer/slot 类型与 anchor/position/predecessor/successor/incumbent/KEEP-DEFER。

缺失字段通过有效位区分，不与 depot=0 混同。终局 D/Q/E/J、service_ok、reject、selected
绝不进入输入。坐标统计由可见节点计算（不引入全局统计泄露）。

用法：
    from coldchain_visible_features import extract_context_features, extract_action_features
"""
import numpy as np


# 每节点订单特征维：coords(2) + demand/tw_start/tw_end/service/temp_class/initial_quality(6)
# + is_depot(1) = 9
ORDER_FEAT_DIM = 9
# 每车车队特征维：node/ready/load/needs/committed_next(+valid)/committed_arrive(+valid)/
# committed_finish(+valid)/舱温与 zone load(6) = 16
FLEET_FEAT_DIM = 16
# 每候选动作特征维：customer/slot_kind/slot_anchor/position/predecessor/successor/incumbent/
# is_pseudo = 8（另有 8 个有效位）
ACTION_FEAT_DIM = 8
# 候选局部特征：4 端点(customer/anchor/pred/succ) × 6(coord2+demand+tw2+temp) + 增量 = 25，
# 另有 4 个端点有效位
LOCAL_VAL_DIM = 25
LOCAL_VALID_DIM = 4

# 去编号（de-index）消融：把「任意节点身份编号」作为连续数值的输入通道屏蔽（置 0），
# 保持特征维度与 head 形状不变。有效位保留，因此「缺失 vs depot」仍由有效位 + 局部几何表达。
# 编号仍用于内部查找/端点解析/动作执行，只是不作为数值喂给模型。
#   动作 8 维：customer(0) / slot_anchor(2) / predecessor(4) / successor(5)
#   车队 16 维：vehicle_node(0) / committed_next(4)
ACTION_ID_CHANNELS = (0, 2, 4, 5)
FLEET_ID_CHANNELS = (0, 4)


def _mask_id_channels(arr, channels):
    """屏蔽编号通道：返回拷贝并把指定列置 0（不就地修改输入）。"""
    out = np.array(arr, copy=True)
    for c in channels:
        out[..., c] = 0.0
    return out


def extract_candidate_local_features(dataset, inst_idx, action, incremental_distance=None,
                                     capacity=50.0):
    """候选局部特征：customer/anchor/predecessor/successor 的坐标/需求/时间窗/温区 + 插入增量。

    这是把「节点编号」换成「节点几何与冷链属性」的关键输入；数字编号不作为连续数值依据。
    anchor_node = slot_anchor if >0 else 0（depot-route/new_route 均指 depot）；DEFER 只有
    customer 有效，anchor/pred/succ 零向量 + 有效位区分。

    返回 (vals [25] float32, valid [4] bool)。
    """
    tw_max = float(dataset['tw_end'][inst_idx].max())
    a = action or {}
    customer = a.get('customer')
    slot_kind = a.get('slot_kind') or a.get('kind')
    slot_anchor = a.get('slot_anchor')
    predecessor = a.get('predecessor')
    successor = a.get('successor')

    if slot_kind == 'defer':
        anchor = None
        pred = None
        succ = None
    else:
        anchor = (int(slot_anchor) if slot_anchor is not None and int(slot_anchor) > 0 else 0)
        pred = predecessor
        succ = successor

    def _node(n):
        if n is None:
            return np.zeros(6, np.float32), False
        n = int(n)
        v = np.array([
            float(dataset['coords'][inst_idx, n, 0]),
            float(dataset['coords'][inst_idx, n, 1]),
            float(dataset['demands'][inst_idx, n]) / max(capacity, 1e-6),
            float(dataset['tw_start'][inst_idx, n]) / tw_max,
            float(dataset['tw_end'][inst_idx, n]) / tw_max,
            float(dataset['temp_class'][inst_idx, n]) / 2.0,
        ], np.float32)
        return v, True

    vals = []
    valid = []
    for n in (customer, anchor, pred, succ):
        v, ok = _node(n)
        vals.append(v)
        valid.append(ok)
    vals = np.concatenate(vals)
    incr = float(incremental_distance) if incremental_distance is not None else 0.0
    vals = np.concatenate([vals, np.array([incr], np.float32)])
    return vals, np.asarray(valid, bool)


def extract_order_features(dataset, inst_idx, visible_mask):
    """已揭示订单特征 + node_visible 掩码。未揭示订单的特征置 0。"""
    N = int(dataset['coords'].shape[1])
    vis = np.asarray(visible_mask, dtype=bool).reshape(N)
    cols = []
    coords = dataset['coords'][inst_idx].astype(np.float32)  # (N, 2)
    cols.append(coords * vis[:, None])
    for key in ('demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
                'initial_quality'):
        v = dataset[key][inst_idx].astype(np.float32).reshape(N)
        cols.append((v * vis)[:, None])
    is_depot = (np.arange(N) == 0).astype(np.float32)[:, None]
    cols.append(is_depot)
    order = np.concatenate(cols, axis=1).astype(np.float32)  # (N, ORDER_FEAT_DIM)
    return order, vis


def _coldchain_state_summary(state):
    """从序列化 VehicleColdChainState 取舱温/zone load 汇总（缺省 → 0 + 有效位）。"""
    if not isinstance(state, dict):
        return np.zeros(6, np.float32), np.zeros(6, bool)
    temp = state.get('compartment_temperature_c') or (0.0, 0.0, 0.0)
    load = state.get('zone_load') or (0.0, 0.0, 0.0)
    vals = np.asarray([float(x) for x in (list(temp) + list(load))], np.float32)
    valid = np.ones(6, bool)
    return vals, valid


def extract_fleet_features(snapshot, deindex=False):
    """车队特征（每车固定维）+ vehicle_valid 掩码。

    deindex=True 时把 vehicle_node / committed_next 的原始编号屏蔽（置 0），
    但 committed_next_valid 有效位保留，以区分「无 committed」与「有 committed」。
    """
    K = int(snapshot['num_vehicles'])
    sn = snapshot
    node = np.asarray(sn['vehicle_node'], np.float32).reshape(K, 1)
    ready = np.asarray(sn['vehicle_ready'], np.float32).reshape(K, 1)
    load = np.asarray(sn['vehicle_load'], np.float32).reshape(K, 1)
    needs = np.asarray(sn['needs_replan'], np.float32).reshape(K, 1)
    # committed_next：-1 = PAD（无 committed），-1 本身作为哨兵 + 有效位区分。
    committed_next = np.asarray(sn['committed_next'], np.float32).reshape(K, 1)
    committed_next_valid = (committed_next[:, 0] >= 0).astype(np.float32).reshape(K, 1)

    def _nan2zero(a):
        a = np.asarray(a, np.float64).reshape(K)
        valid = (~np.isnan(a)).astype(np.float32).reshape(K, 1)
        v = np.nan_to_num(a, nan=0.0).astype(np.float32).reshape(K, 1)
        return v, valid

    committed_arrive, ca_valid = _nan2zero(sn['committed_arrive'])
    committed_finish, cf_valid = _nan2zero(sn['committed_finish'])

    # 舱温/zone load 汇总（每车 6 维 + 有效位）。
    states = sn.get('vehicle_coldchain_state', [None] * K)
    cc = np.zeros((K, 6), np.float32)
    cc_valid = np.zeros((K, 6), np.float32)
    for k in range(K):
        cc[k], cc_valid[k] = _coldchain_state_summary(states[k] if k < len(states) else None)

    fleet = np.concatenate([
        node, ready, load, needs, committed_next, committed_next_valid,
        committed_arrive, ca_valid, committed_finish, cf_valid, cc,
    ], axis=1).astype(np.float32)
    if deindex:
        fleet = _mask_id_channels(fleet, FLEET_ID_CHANNELS)
    vehicle_valid = np.ones(K, bool)
    return fleet, vehicle_valid


def extract_action_features(candidate, deindex=False):
    """动作特征 + action_valid 掩码（缺失字段不当作 depot=0）。

    deindex=True 时把 customer/slot_anchor/predecessor/successor 的原始编号屏蔽（置 0），
    有效位保留（缺失 vs 存在的区别不变）。
    """
    a = candidate.get('action') or {}
    is_pseudo = bool(candidate.get('is_pseudo', False))
    pseudo = candidate.get('pseudo')
    customer = a.get('customer')
    # 普通候选用 slot_kind（'anchored'|'new_route'）；DEFER 伪动作在导出器里写的是 kind='defer'。
    slot_kind = a.get('slot_kind') or a.get('kind')
    slot_anchor = a.get('slot_anchor')
    position = a.get('position')
    predecessor = a.get('predecessor')
    successor = a.get('successor')
    incumbent = bool(a.get('incumbent', False))

    def _val(x):
        return 0.0 if x is None else float(x)

    def _ok(x):
        return x is not None

    # 类别 slot_kind → 数值编码（anchored=0, new_route=1, defer=2），缺失用有效位。
    kind_map = {'anchored': 0.0, 'new_route': 1.0, 'defer': 2.0}
    kind_val = kind_map.get(slot_kind, 0.0)
    kind_ok = slot_kind in kind_map

    vals = np.asarray([
        _val(customer), kind_val, _val(slot_anchor), _val(position),
        _val(predecessor), _val(successor), float(incumbent), float(is_pseudo),
    ], np.float32)
    valid = np.asarray([
        _ok(customer), kind_ok, _ok(slot_anchor), _ok(position),
        _ok(predecessor), _ok(successor), True, True,
    ], bool)
    if deindex:
        vals = _mask_id_channels(vals, ACTION_ID_CHANNELS)
    return vals, valid


def extract_context_features(dataset, inst_idx, snapshot, deindex=False):
    """context 级（订单 + 车队）可见特征；不含任何候选/终局信息。

    deindex=True 时车队编号通道被屏蔽（订单特征本就是按节点排列的几何/属性，
    经掩码平均后对节点重编号不变）。
    """
    order, node_visible = extract_order_features(dataset, inst_idx, snapshot['visible_mask'])
    fleet, vehicle_valid = extract_fleet_features(snapshot, deindex=deindex)
    return {
        'order_feats': order,
        'node_visible': node_visible,
        'fleet_feats': fleet,
        'vehicle_valid': vehicle_valid,
    }


# --------------------------------------------------------------------------- #
# 候选相关冷链状态 X_state（候选—车辆—货物对应关系）
# --------------------------------------------------------------------------- #
def _zone_cargo_quality(state, zones=3):
    """按温区汇总在车货物的「剩余品质加权载货量」= Σ quantity × quality_remaining/initial_quality。

    用剩余品质比例加权，使模型同时看到数量与退化程度；与 zone_load（纯数量）区分。
    """
    q = [0.0] * zones
    for lot in state.get('cargo_manifest', ()):
        z = int(lot['temp_class'])
        initial = max(float(lot['initial_quality']), 1e-9)
        q[z] += float(lot['quantity']) * (float(lot['quality_remaining']) / initial)
    return q


def extract_vehicle_cargo_state(snapshot, vid, plan, capacity=50.0, tw_max=24.0):
    """目标车辆 vid 的候选相关冷链状态（13 维）。

    用 plan.anchor_time / plan.anchor_load（**可执行插入起点**：committed 车是 committed_finish /
    取货后载重），不用 snapshot 的 vehicle_ready / vehicle_load（当前时刻状态，语义不同）。

    [anchor_time, remaining_capacity, temp(3), zone_load(3), zone_quality(3), state_valid, has_cargo]
    state_valid=1 表示该车有有效冷链状态字典（即使 manifest 为空）；has_cargo=1 仅在
    cargo_manifest 非空时置 1。二者不混称。
    """
    anchor_time = float(plan.anchor_time) / max(tw_max, 1e-6)
    rem_cap = (capacity - float(plan.anchor_load)) / max(capacity, 1e-6)
    states = snapshot.get('vehicle_coldchain_state')
    cc = states[vid] if (states is not None and vid < len(states)) else None
    if isinstance(cc, dict):
        temp = [float(x) / 40.0 for x in (cc.get('compartment_temperature_c') or (0.0, 0.0, 0.0))]
        zload = [float(x) / max(capacity, 1e-6) for x in (cc.get('zone_load') or (0.0, 0.0, 0.0))]
        zqual = [q / max(capacity, 1e-6) for q in _zone_cargo_quality(cc)]
        state_valid = 1.0
        has_cargo = 1.0 if cc.get('cargo_manifest') else 0.0
    else:
        temp = [0.0, 0.0, 0.0]
        zload = [0.0, 0.0, 0.0]
        zqual = [0.0, 0.0, 0.0]
        state_valid = 0.0
        has_cargo = 0.0
    return np.asarray([anchor_time, rem_cap] + temp + zload + zqual
                      + [state_valid, has_cargo], np.float32)


X_STATE_DIM = 13


def resolve_target_vid(action, plans, allowed_vehicle_ids=None):
    """候选动作的目标车辆 vid；DEFER 返回 None（显式「无目标车辆」编码）。

    与 apply_action 同一匹配规则：anchored → _match_anchored；new_route → _match_new_route
    （min 空 idle@depot 车，限定 allowed_vehicle_ids）。空车仍有容量/可用时间/温度，不整段填零；
    只有 DEFER 用明确的「无目标」编码。
    """
    slot_kind = action.get('slot_kind') or action.get('kind')
    if slot_kind == 'defer':
        return None
    from action_contract import _match_anchored, _match_new_route
    if slot_kind == 'new_route':
        return int(_match_new_route(plans, allowed_vehicle_ids))
    return int(_match_anchored(plans, int(action.get('slot_anchor'))))
