"""
FleetSlot —— route slot 抽象（导师 02 文档 §14）。

把「K 辆物理车」降维成「route slot」，让 vehicle symmetry 直接在 action space 消失：

  - anchored slot：active/committed 车辆，各自对应一个 slot（state 有语义，不可折叠）。
  - anonymous_idle slot：等价 idle@depot 车辆折叠成一个「新路线」slot（vehicle ID 无语义）。

第一版不搞 GNN / Sinkhorn / slot attention，只建立正确的 action space。后续 FleetAssignmentHead
可以从 [K physical vehicles, N customers] 升级为 [S fleet slots, N customers]（S 可小于 K）。
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class FleetSlot:
    slot_id: int
    kind: str                       # 'anchored' | 'anonymous_idle'
    vehicle_ids: Tuple[int, ...]    # anchored = (单辆车,)；anonymous_idle = 等价 idle 车集合
    anchor_node: int
    anchor_time: float
    load: float
    remaining_capacity: float       # 后续：capacity / type / cold compartment 等 heterogeneous 特征
    current_members: List[int] = field(default_factory=list)   # 已分配进该 slot 的客户


def build_fleet_slots(vehicles, clock, capacity, anchor_loads, tol=1e-6):
    """把 vehicles 折叠成 fleet slots。

    anchor_loads: dict[vehicle_id -> anchor_load]（含 committed demand）。
    active/committed 车 → 各一个 anchored slot；等价 idle 车 → 一个 anonymous_idle slot（or 多个
    new-route slot，后续按需展开）。
    """
    from vehicle_equivalence import build_equivalence_classes

    slots = []
    # 等价类：idle 车折叠，ready/committed 车各自独立（不同 anchor）
    eq_classes = build_equivalence_classes(vehicles, clock, anchor_loads, tol)
    for _cid, vids in eq_classes.items():
        # 取该等价类第一辆车作为代表（等价类内状态相同）
        rep = next(v for v in vehicles if int(v.vehicle_id) == vids[0])
        if rep.status == 'idle':
            kind = 'anonymous_idle'
            node, t, ld = 0, float(clock), 0.0
        elif rep.status == 'ready':
            kind = 'anchored'
            node, t, ld = int(rep.current_node), float(rep.ready_time), float(rep.current_load)
        elif rep.status == 'committed':
            kind = 'anchored'
            node, t = int(rep.committed_next), float(rep.committed_finish)
            ld = float(anchor_loads.get(int(rep.vehicle_id), float(rep.current_load)))
        else:
            continue  # returning/closed 不进入 slot
        slots.append(FleetSlot(
            slot_id=len(slots), kind=kind, vehicle_ids=tuple(vids),
            anchor_node=node, anchor_time=t, load=ld,
            remaining_capacity=max(0.0, capacity - ld),
        ))
    return slots
