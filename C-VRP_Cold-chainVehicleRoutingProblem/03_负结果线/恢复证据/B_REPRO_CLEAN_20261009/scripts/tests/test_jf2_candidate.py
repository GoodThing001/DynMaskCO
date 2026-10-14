"""
JF2 Step 5（冻结 jf2_sound_v1 candidate semantics）测试。

验证 `build_sound_candidate_set` 原语：
  1. 确定性：同一 FleetState 两次构造结果 identical。
  2. sound hard mask：direct-arrival TW infeasible 的 (vehicle, customer) 被 mask；capacity+direct-TW
     才是 hard mask，return-depot 只作 soft feature（return_slack 记录但不 mask）。

注：sound mask 与 JF1-H 的等价性（min-travel 贪心从不把客户分给 direct-TW-infeasible 车）
已在 VAL128 上指纹级验证（128 实例 exact 一致），见 `JF2_执行进度表.md` Step 5 结论。

纯 NumPy，无 JAX / model，可本地跑：

    python scripts/tests/test_jf2_candidate.py
"""
import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(os.path.dirname(_BASE), 'simulation')
sys.path.insert(0, _SIM)

from strict_online_env import StrictOnlineEnv, VehicleState
from joint_fleet import build_sound_candidate_set


def _make_dataset():
    """3 节点：0=depot(0.5,0.5)，1=客户 j(0.9,0.5) 窄 TW [0,0.1]，2=近 j 的 dummy(0.88,0.5,demand=0)。"""
    coords = np.array([[[0.5, 0.5], [0.9, 0.5], [0.88, 0.5]]], dtype=np.float32)
    demands = np.array([[0., 5., 0.]], dtype=np.float32)
    tw_start = np.array([[0., 0., 0.]], dtype=np.float32)
    tw_end = np.array([[100., 0.1, 100.]], dtype=np.float32)
    service_time = np.array([[0., 0.1, 0.]], dtype=np.float32)
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
            'service_time': service_time, 'temp_class': np.zeros((1, 3), np.int32),
            'reveal_time': np.zeros((1, 3), np.float32)}


def _state():
    dataset = _make_dataset()
    env = StrictOnlineEnv(dataset, capacity=50)
    A = VehicleState(vehicle_id=0, status='ready', current_node=0, ready_time=0.0)  # depot，距 j 0.4
    B = VehicleState(vehicle_id=1, status='ready', current_node=2, ready_time=0.0)  # 距 j 0.02
    served_mask = np.zeros(3, dtype=bool)
    served_mask[0] = True
    visible_ids = [1]
    return env, [A, B], served_mask, visible_ids


def test_candidate_set_deterministic():
    env, vehicles, served_mask, visible_ids = _state()
    cs1 = build_sound_candidate_set(env, 0, 0.0, vehicles, served_mask, visible_ids)
    cs2 = build_sound_candidate_set(env, 0, 0.0, vehicles, served_mask, visible_ids)
    assert cs1.customer_order == cs2.customer_order
    assert cs1.active_vehicle_ids == cs2.active_vehicle_ids
    assert cs1.feasible_vehicles == cs2.feasible_vehicles
    assert cs1.return_slack == cs2.return_slack


def test_candidate_set_direct_tw_mask():
    env, vehicles, served_mask, visible_ids = _state()
    cs = build_sound_candidate_set(env, 0, 0.0, vehicles, served_mask, visible_ids)
    # customer order = pool 顺序
    assert cs.customer_order == [1]
    # 车 A 距 j 0.4（> tw_end 0.1）→ direct-TW mask；车 B 距 j 0.02 → feasible
    assert cs.feasible_vehicles[1] == [1], f"车 A 应被 direct-TW mask，实际 {cs.feasible_vehicles[1]}"
    # return_slack 作为 soft feature 记录（B 的 return_slack 存在且为 float）
    assert (1, 1) in cs.return_slack
    assert isinstance(cs.return_slack[(1, 1)], float)


if __name__ == '__main__':
    test_candidate_set_deterministic()
    test_candidate_set_direct_tw_mask()
    print("PASS: test_jf2_candidate — jf2_sound_v1 candidate set（确定性 + sound hard mask）")
