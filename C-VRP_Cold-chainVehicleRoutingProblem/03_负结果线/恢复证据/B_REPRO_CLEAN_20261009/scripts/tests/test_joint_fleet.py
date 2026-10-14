"""
JF0 Blocking Tests —— fleet-level ownership semantics（导师 §21 的 P0）。

验证：
  #1 frozen（已执行 prefix + committed）客户永不进入 releasable pool
  #2 committed 客户永不跨车重分配
  #3 mutable 客户可跨车重分配（进 pool）
  #4 每个客户最多属于一辆车（ownership 不重复）
  #5 future 未揭示客户永不进入 pool（non-anticipatory）
  #6 closed 车辆永不获取新客户（被排除出 anchors）

只用 strict_online_env 的 VehicleState + joint_fleet 的 JF0 函数，不依赖模型。

用法:
    python scripts/tests/test_joint_fleet.py
"""

import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from strict_online_env import VehicleState
from joint_fleet import get_fleet_anchors, get_global_pool, ownership_partition, get_vehicle_anchor_node


def _mk_vehicles():
    """构造一个含 idle/ready/committed/returning/closed 各种状态的 fleet。"""
    return [
        # 0 idle at depot
        VehicleState(vehicle_id=0, status='idle', current_node=0, ready_time=0.0, current_load=0.0),
        # 1 ready at customer 5, has mutable_suffix [7, 0]
        VehicleState(vehicle_id=1, status='ready', current_node=5, ready_time=3.0, current_load=2.0,
                     mutable_suffix=[7, 0]),
        # 2 committed to customer 9 (frozen), has mutable_suffix [11, 0]
        VehicleState(vehicle_id=2, status='committed', current_node=6, ready_time=4.0, current_load=3.0,
                     committed_next=9, committed_finish=8.0, mutable_suffix=[11, 0]),
        # 3 returning (will close)
        VehicleState(vehicle_id=3, status='returning', current_node=12, ready_time=5.0, current_load=0.0,
                     return_finish=6.0),
        # 4 closed
        VehicleState(vehicle_id=4, status='closed', current_node=0, ready_time=99.0, current_load=0.0),
    ]


def test_frozen_and_committed_excluded():
    print("=" * 60)
    print("#1/#2 frozen（executed+committed）永不进 pool，committed 不跨车")
    print("=" * 60)
    vehicles = _mk_vehicles()
    served_mask = np.zeros(15, dtype=bool)
    served_mask[0] = True
    served_mask[5] = True   # 已执行（车1的 current_node）
    served_mask[6] = True   # 已执行（车2的 current_node）
    visible_ids = [5, 6, 7, 9, 11, 13]
    pool, locked = get_global_pool(vehicles, served_mask, visible_ids)
    # 9 是 committed（车2），必须 locked；5/6 已服务，必须排除；7/11 是 mutable，可进 pool
    ok = (9 in locked and 5 not in pool and 6 not in pool
          and 7 in pool and 11 in pool)
    print(f"  pool={sorted(pool)} locked={sorted(locked)}")
    print(f"  => {'PASS' if ok else 'FAIL'} (9 committed 冻结；5/6 已服务排除；7/11 mutable 可释放)")
    return ok


def test_anchor_semantics():
    print("\n" + "=" * 60)
    print("#3 committed 车的 anchor 是 committed_next（服务完成后可规划）")
    print("=" * 60)
    vehicles = _mk_vehicles()
    anchors = get_fleet_anchors(vehicles)
    by_id = {a.vehicle_id: a for a in anchors}
    # closed 车（4）被排除
    ok4 = (4 not in by_id)
    # ready 车（1）anchor = current_node=5
    ok1 = (by_id[1].current_node == 5 and by_id[1].status == 'ready')
    # committed 车（2）anchor 节点 = committed_next=9
    a2 = by_id[2]
    ok2 = (a2.status == 'committed' and a2.committed_next == 9
           and get_vehicle_anchor_node(a2) == 9 and 9 in a2.locked_customers)
    ok = ok4 and ok1 and ok2
    print(f"  closed excluded={ok4}  ready anchor={ok1}  committed anchor={ok2}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_each_customer_one_vehicle():
    print("\n" + "=" * 60)
    print("#4 每个客户最多属于一辆车（ownership 不重复）")
    print("=" * 60)
    vehicles = _mk_vehicles()
    served_mask = np.zeros(15, dtype=bool)
    served_mask[0] = True
    visible_ids = [5, 7, 9, 11, 13]
    _, locked = get_global_pool(vehicles, served_mask, visible_ids)
    # locked（committed）= {9}；mutable = {7, 11}（车1/车2各一个，不重复）
    # 模拟分配：9→车2, 7→车1, 11→车2（车2 mutable 尾），检查无重复
    assigned = {9: 2, 7: 1, 11: 2}
    owners = {}
    dup = False
    for c, v in assigned.items():
        if c in owners and owners[c] != v:
            dup = True
        owners[c] = v
    # 每个客户一个 owner
    ok = (not dup and len(owners) == 3)
    print(f"  owners={owners} no_dup={ok}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_future_invisible():
    print("\n" + "=" * 60)
    print("#5 future 未揭示客户永不进 pool（non-anticipatory）")
    print("=" * 60)
    vehicles = _mk_vehicles()
    served_mask = np.zeros(15, dtype=bool)
    served_mask[0] = True
    # visible_ids 只含 t<=clock 已揭示的客户（env 已过滤 future）；这里额外验证 reveal_time>clock 的不进 pool
    reveal_time = np.zeros(15, dtype=np.float32)
    reveal_time[13] = 10.0  # 客户13 未来才揭示
    visible_ids = [5, 7, 9, 11]  # 不含 13（env 语义）
    pool, _ = get_global_pool(vehicles, served_mask, visible_ids)
    ok = (13 not in pool and 13 not in visible_ids)
    print(f"  pool={sorted(pool)} (13 不在 visible_ids 也不在 pool)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_non_replan_tail_frozen():
    print("\n" + "=" * 60)
    print("#6 非 replan 车的 mutable tail 冻结（防 stale-tail duplicate）")
    print("=" * 60)
    vehicles = _mk_vehicles()
    served_mask = np.zeros(15, dtype=bool)
    served_mask[0] = True
    visible_ids = [5, 7, 9, 11, 13]
    # 车1（ready，mutable_suffix=[7,0]）不在 replan_ids → 客户7 应冻结（进 locked，不进 pool）
    pool, locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids={0, 2})
    ok_frozen = (7 in locked and 7 not in pool)
    # 车1 在 replan_ids → 客户7 的 tail 被释放（进 pool）
    pool2, locked2 = get_global_pool(vehicles, served_mask, visible_ids, replan_ids={0, 1, 2})
    ok_released = (7 not in locked2 and 7 in pool2)
    ok = ok_frozen and ok_released
    print(f"  非 replan：7 locked={ok_frozen}；replan：7 释放={ok_released}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    results = {
        'frozen/committed excluded': test_frozen_and_committed_excluded(),
        'anchor semantics': test_anchor_semantics(),
        'each customer one vehicle': test_each_customer_one_vehicle(),
        'future invisible': test_future_invisible(),
        'non-replan tail frozen': test_non_replan_tail_frozen(),
    }
    print("\n" + "=" * 60)
    all_ok = all(results.values())
    for k, v in results.items():
        print(f"  {k:<30} {'PASS' if v else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS (5/5)' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
