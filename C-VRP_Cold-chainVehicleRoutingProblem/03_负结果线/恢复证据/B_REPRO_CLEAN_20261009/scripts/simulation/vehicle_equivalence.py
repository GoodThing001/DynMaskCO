"""
VehicleEquivalenceClass —— 车辆等价类（导师 02 文档 §7）。

动态多车问题里，idle@depot 同质车辆（同 anchor/time/load/status/无 committed）互相可置换，
vehicle ID 无语义。本模块把「物理车辆」折叠成「等价类」，供 FleetSlot / partition 监督使用。

关键原则（02 文档 §7.1）：只有 anchor node、time、load、committed/mutable context 都等价才算
等价；不能只写 `if status=='idle': equivalent`（后续 heterogeneous fleet 还要加 capacity/type 等）。
"""
from typing import Tuple, List


def vehicle_anchor(v, clock):
    """车辆 v 的规划锚点 (node, time, load)。与 joint_fleet.compute_anchor_info 语义一致。

    idle/ready：anchor = (current_node, ready_time, current_load)。
    committed：anchor = (committed_next, committed_finish, current_load + demand[committed_next])。
    注意：committed 的 demand[committed_next] 由调用方通过 load 参数传入（本函数不读 env）。
    """
    if v.status == 'committed' and v.committed_next not in (None, 0):
        return (int(v.committed_next), float(v.committed_finish), None)  # load 由调用方补
    return (int(v.current_node), float(v.ready_time), float(v.current_load))


def vehicle_equiv_key(v, clock, anchor_load, tol=1e-6):
    """等价判定 key。等价 ⟺ 同 key。anchor_load 为「完成 committed 后」的负载（含 demand[committed_next]）。"""
    def q(x):
        return round(float(x) / tol) * tol

    if v.status == 'idle':
        return ('idle', 0, q(clock), 0.0, ())
    if v.status == 'ready':
        return ('ready', int(v.current_node), q(v.ready_time), q(anchor_load), tuple(v.mutable_suffix))
    if v.status == 'committed':
        return ('committed', int(v.committed_next), q(v.committed_finish), q(anchor_load), tuple(v.mutable_suffix))
    # returning / closed：各车独立（不可接新客户，不参与等价折叠）
    return (v.status, int(v.vehicle_id))


def build_equivalence_classes(vehicles, clock, anchor_loads, tol=1e-6):
    """把 vehicles 折叠成等价类。返回 {class_id: [vehicle_ids]}。

    anchor_loads: dict[vehicle_id -> anchor_load]（含 committed demand），由调用方用
    compute_anchor_info 提供。等价车辆（同 key）归一类，class_id 按首次出现顺序编号。
    """
    classes = {}
    order = []
    for v in vehicles:
        key = vehicle_equiv_key(v, clock, anchor_loads.get(int(v.vehicle_id), float(v.current_load)), tol)
        if key not in classes:
            classes[key] = []
            order.append(key)
        classes[key].append(int(v.vehicle_id))
    return {i: classes[k] for i, k in enumerate(order)}
