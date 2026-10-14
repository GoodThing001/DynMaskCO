"""
JF2 Gate-0 regression（Phase 0 Closure Step 1-2）。

验证 `JointAssignmentBeamReplanner(candidate_mode='legacy_jf1h', beam_width=1)` 与冻结的
`JointAssignmentReplanner('heuristic')`（JF1-H）在 strict-online 环境下**逐实例 exact 一致**
（同 assignment + 同 route sequence + 同 cost）。这是 JF2 能否把 JF1-H 当作 residual baseline
的前置 Gate（master doc Part III Step 1-2）。

纯 NumPy，不依赖 JAX / model，可直接本地跑：

    python scripts/tests/test_jf2_regression.py

"""
import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(os.path.dirname(_BASE), 'simulation')
sys.path.insert(0, _SIM)

from strict_online_env import StrictOnlineEnv
from joint_fleet import JointAssignmentReplanner, JointAssignmentBeamReplanner


def _make_toy_dataset():
    """构造一个含动态 reveal 的 toy 实例（8 节点 = depot + 7 客户，3 车，容量 50）。"""
    n = 8
    coords = np.array([[
        [0.5, 0.5],   # 0 depot
        [0.1, 0.2], [0.9, 0.1], [0.2, 0.9], [0.8, 0.8],
        [0.3, 0.4], [0.6, 0.7], [0.4, 0.1],
    ]], dtype=np.float32)
    demands = np.array([[0., 12., 14., 10., 15., 11., 13., 9.]], dtype=np.float32)
    tw_start = np.array([[0.] * n], dtype=np.float32)
    # depot tw_end = horizon（宽裕），客户 TW 足够宽以保持可行
    tw_end = np.array([[100., 60., 60., 60., 60., 60., 60., 60.]], dtype=np.float32)
    service_time = np.full((1, n), 0.5, dtype=np.float32)
    service_time[0, 0] = 0.0
    temp_class = np.zeros((1, n), dtype=np.int32)
    reveal_time = np.array([[0., 0., 0., 0., 8., 12., 0., 20.]], dtype=np.float32)
    return {
        'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
        'service_time': service_time, 'temp_class': temp_class, 'reveal_time': reveal_time,
    }


def _fingerprint(traces):
    """从 execution trace 抽取 canonical 指纹：按 vehicle_id 排序的 (services, return_arrival)。"""
    fp = []
    for tr in sorted(traces, key=lambda t: t.vehicle_id):
        services = tuple(int(sr.node) for sr in tr.services)
        ret = None if tr.return_arrival is None else round(float(tr.return_arrival), 6)
        fp.append((tr.vehicle_id, services, ret))
    return tuple(fp)


def test_legacy_jf1h_parity():
    dataset = _make_toy_dataset()
    capacity = 50
    tw_speed = 1.0
    num_vehicles = 3

    # 参考：冻结 JF1-H（heuristic joint）
    env_ref = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles,
                              replanner=JointAssignmentReplanner('heuristic'))
    traces_ref, served_ref = env_ref.run(0)

    # legacy_jf1h：必须走同一路径
    env_legacy = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles,
                                 replanner=JointAssignmentBeamReplanner(
                                     beam_width=1, candidate_mode='legacy_jf1h'))
    traces_legacy, served_legacy = env_legacy.run(0)

    fp_ref = _fingerprint(traces_ref)
    fp_legacy = _fingerprint(traces_legacy)

    assert fp_ref == fp_legacy, (
        "legacy_jf1h 与 JF1-H trace 不一致：\n"
        f"  JF1-H     = {fp_ref}\n"
        f"  legacy    = {fp_legacy}")
    assert np.array_equal(served_ref, served_legacy), "served_mask 不一致"


def test_legacy_jf1h_ignores_beam_width():
    """legacy 模式下 beam_width 无意义：B=1 与 B=16 应完全一致（不再进入 beam search）。"""
    dataset = _make_toy_dataset()
    fp = {}
    for B in (1, 16):
        env = StrictOnlineEnv(dataset, 50, 1.0, 3,
                              replanner=JointAssignmentBeamReplanner(
                                  beam_width=B, candidate_mode='legacy_jf1h'))
        traces, _ = env.run(0)
        fp[B] = _fingerprint(traces)
    assert fp[1] == fp[16], f"legacy 模式下 B 不应改变结果：\n  B=1 ={fp[1]}\n  B=16={fp[16]}"


if __name__ == '__main__':
    test_legacy_jf1h_parity()
    test_legacy_jf1h_ignores_beam_width()
    print("PASS: test_jf2_regression — legacy_jf1h == 冻结 JF1-H（exact parity）")
