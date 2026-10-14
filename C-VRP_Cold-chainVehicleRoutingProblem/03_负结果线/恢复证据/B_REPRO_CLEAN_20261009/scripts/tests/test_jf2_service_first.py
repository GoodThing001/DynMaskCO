"""
JF2 Step 4（no silent drop）原语测试。

验证（供 JF2 skeleton 的 greedy assigner 使用）：
  1. `_greedy_sequence_report` 显式报告 sequencing 丢弃的客户（不再静默丢）。
  2. `_repair_unresolved` 能把丢弃客户移到能服务它的车辆。

注：no-silent-drop 的 report/repair 原语**不** retro-fit 到已判空的 safe_pair beam
（safe_pair 早期决策发散会导致 fallback 无法恢复且引入新 drop），而是留给 JF2 skeleton
（Phase 1）从零接进 greedy assigner + shadow-incumbent guard。

纯 NumPy，无 JAX / model，可本地跑：

    python scripts/tests/test_jf2_service_first.py
"""
import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(os.path.dirname(_BASE), 'simulation')
sys.path.insert(0, _SIM)

from strict_online_env import StrictOnlineEnv, VehicleState
from joint_fleet import (JointAssignmentReplanner,
                         get_fleet_anchors, compute_anchor_info)


def _make_tight_dataset():
    """3 节点：0=depot(0.5,0.5)，1=客户 j(0.9,0.5) 窄 TW，2=近 j 的 dummy(0.88,0.5,demand=0)。"""
    coords = np.array([[[0.5, 0.5], [0.9, 0.5], [0.88, 0.5]]], dtype=np.float32)
    demands = np.array([[0., 5., 0.]], dtype=np.float32)       # node2 是 dummy
    tw_start = np.array([[0., 0., 0.]], dtype=np.float32)
    tw_end = np.array([[100., 0.1, 100.]], dtype=np.float32)   # j 的 TW 极窄
    service_time = np.array([[0., 0.1, 0.]], dtype=np.float32)
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
            'service_time': service_time, 'temp_class': np.zeros((1, 3), np.int32),
            'reveal_time': np.zeros((1, 3), np.float32)}


def test_sequence_report_drops():
    dataset = _make_tight_dataset()
    env = StrictOnlineEnv(dataset, capacity=50)
    # 车 A 在 depot（距 j 0.4），ready_time=0；直接 anchor→j 到达 0.4 > tw_end[j]=0.1 → 丢弃
    A = VehicleState(vehicle_id=0, status='ready', current_node=0, ready_time=0.0)
    rp = JointAssignmentReplanner('heuristic')
    route, dropped = rp._greedy_sequence_report(env, 0, A, [1])
    assert dropped == [1], f"期望丢弃客户 1，实际 dropped={dropped}, route={route}"
    assert route == [0], f"A 无法服务 j，route 应只有 depot，实际 {route}"


def test_repair_moves_dropped():
    dataset = _make_tight_dataset()
    env = StrictOnlineEnv(dataset, capacity=50)
    A = VehicleState(vehicle_id=0, status='ready', current_node=0, ready_time=0.0)    # 远，服务不了 j
    B = VehicleState(vehicle_id=1, status='ready', current_node=2, ready_time=0.0)    # 近，能服务 j
    anchors = get_fleet_anchors([A, B])
    anchor_info = compute_anchor_info(env, 0, anchors)
    rp = JointAssignmentReplanner('heuristic')

    assignment = {0: [], 1: []}
    still = rp._repair_unresolved(env, 0, [A, B], anchors, anchor_info,
                                  assignment, unresolved=[1], active_ids={0, 1}, capacity=50)
    assert still == [], f"repair 应把 j 放到 B，still={still}"
    assert 1 in assignment.get(1, []), f"j 应被放到车辆 1（B），assignment={assignment}"
    assert 1 not in assignment.get(0, []), "j 不应留在车辆 0（A）"


if __name__ == '__main__':
    test_sequence_report_drops()
    test_repair_moves_dropped()
    print("PASS: test_jf2_service_first — no silent drop report/repair 原语")
