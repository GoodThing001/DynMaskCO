"""
JF2 Step 8（fleet_features）测试。

验证 build_jf2_features 的 shape、base_score、candidate_mask、无 NaN。纯 NumPy，可本地/服务器跑：

    python scripts/tests/test_jf2_features.py
"""
import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_BASE), 'simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(_BASE), 'models'))

from strict_online_env import StrictOnlineEnv, VehicleState
from fleet_features import build_jf2_features, Fv, Fp


def _state():
    # 3 节点：0=depot(0.5,0.5)，1=客户 j(0.9,0.5) 窄 TW [0,0.1]，2=近 j 的 dummy(0.88,0.5)
    coords = np.array([[[0.5, 0.5], [0.9, 0.5], [0.88, 0.5]]], dtype=np.float32)
    demands = np.array([[0., 5., 0.]], dtype=np.float32)
    tw_start = np.array([[0., 0., 0.]], dtype=np.float32)
    tw_end = np.array([[100., 0.1, 100.]], dtype=np.float32)
    service_time = np.array([[0., 0.1, 0.]], dtype=np.float32)
    ds = {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
          'service_time': service_time, 'temp_class': np.zeros((1, 3), np.int32),
          'reveal_time': np.zeros((1, 3), np.float32)}
    env = StrictOnlineEnv(ds, 50)
    A = VehicleState(vehicle_id=0, status='ready', current_node=0, ready_time=0.0)  # depot，距 j 0.4
    B = VehicleState(vehicle_id=1, status='ready', current_node=2, ready_time=0.0)  # 距 j 0.02
    served = np.zeros(3, dtype=bool); served[0] = True
    return env, [A, B], served, [1]


def test_shapes_and_no_nan():
    env, vehicles, served, visible = _state()
    f = build_jf2_features(env, 0, 0.0, vehicles, served, visible)
    K = len(f.active_vehicle_ids)
    N = env.num_nodes
    assert f.veh_feat.shape == (K, Fv)
    assert f.pair_feat.shape == (K, N, Fp)
    assert f.base_score.shape == (K, N)
    assert f.candidate_mask.shape == (K, N)
    assert not np.isnan(f.pair_feat).any(), "pair_feat 不应有 NaN"
    assert not np.isnan(f.veh_feat).any(), "veh_feat 不应有 NaN"


def test_sound_mask_and_base_score():
    env, vehicles, served, visible = _state()
    f = build_jf2_features(env, 0, 0.0, vehicles, served, visible)
    vid_to_row = {vid: i for i, vid in enumerate(f.active_vehicle_ids)}
    # 车 A 距 j 0.4（> tw_end 0.1）→ direct-TW mask；车 B 距 j 0.02 → feasible
    assert not f.candidate_mask[vid_to_row[0], 1], "车 A 应被 direct-TW mask"
    assert f.candidate_mask[vid_to_row[1], 1], "车 B 应 feasible"
    assert f.base_score[vid_to_row[0], 1] == -1e9, "masked 处 base_score 应 == -1e9"
    assert abs(f.base_score[vid_to_row[1], 1] - (-0.02)) < 1e-4, "B 的 base_score 应 = -travel ≈ -0.02"


if __name__ == '__main__':
    test_shapes_and_no_nan()
    test_sound_mask_and_base_score()
    print("PASS: test_jf2_features — shape / base_score / sound mask / no NaN")
