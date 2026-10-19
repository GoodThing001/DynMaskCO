"""滚动优化器单元测试：接受/拒绝联合优化 + 预算约束 + 路线 + A-v1 评价。

2026-10-18（P0 收口）修订说明
============================
本文件此前有 1 个用例长期为红：`test_in_transit_no_double_count`。
根因不是"在途重复计数"缺陷复发，而是该用例把两单需求设为 25/25，
载重**恰好等于容量 50**；而接单侧容量判据自 2026-09-28 起是

    if l > self.capacity - 1e-4:  return False      # run_exp_energy_c0.C0ReserveReplanner

即"必须严格小于 capacity-1e-4"（保守余量，防止 float32 边界载重导致执行期崩溃）。
所以"恰好满容量"被拒绝是**设计行为**，旧断言与当前合同不一致。

本次只改测试侧（未改 `_route_feasible`、未改物理合同、未改任何生产算法）：
  * 原用例夹具改为 24/24（余量 2），并**新增决策点断言**：t=1.0 时起始载重必须恰为 24
    （而不是 48），且已服务的单 1 不得再出现在剩余路线里——不再只看最终 accepted 集合；
  * 新增负控 `test_capacity_boundary_rejected_by_conservative_margin`（25/25 必须被拒），
    把 `capacity - 1e-4` 这条口径钉住，防止被误改回 `+1e-6`。

**旧 FAIL 保留登记**：本文件历史状态为 `test_in_transit_no_double_count` FAIL
（`AssertionError: ({1}, {2})`），该结论**不因本次修订而被追认为 PASS**；
修订是在新口径下重写夹具，历史红记录见 `docs/仓库治理/12_第二阶段_任务D_既有FAIL诊断.md`。
"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_contract import ColdChainContractV2
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from rolling_optimizer import RollingOptimizeReplanner
from strict_online_env import StrictOnlineEnv

# 接单侧容量判据的保守余量（与 run_exp_energy_c0.C0ReserveReplanner._route_feasible 一致）。
# 这里只作为断言里的常量引用，不复制、不替代生产实现。
CAPACITY = 50.0
CAPACITY_MARGIN = 1e-4


def _small_dataset():
    return {
        'coords': np.array([[[0.5, 0.5], [0.6, 0.5], [0.4, 0.5]]], dtype=np.float32),
        'demands': np.array([[0.0, 2.0, 2.0]], dtype=np.float32),
        'tw_start': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        'tw_end': np.array([[22.0, 22.0, 22.0]], dtype=np.float32),
        'service_time': np.array([[0.0, 0.05, 0.05]], dtype=np.float32),
        'reveal_time': np.array([[1e6, 0.5, 0.5]], dtype=np.float32),
        'temp_class': np.array([[0, 1, 0]], dtype=np.int32),
    }


def _single(ds):
    return {k: v[0] for k, v in ds.items()}


def _run(budget, p_c):
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_small_dataset(), c)
    rp = RollingOptimizeReplanner(budget=budget, capacity=CAPACITY, booking_horizon=16.0,
                                  contract=c, cooling_share=2.0,
                                  reject_penalty=p_c, time_limit=10.0)
    env = StrictOnlineEnv(ds, capacity=CAPACITY, num_vehicles=2, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    return traces, set(rp._accepted), set(rp._rejected), ds, c


def _run_double_count(demand, recorder=None):
    """单车、容量 50、两单各 `demand`，先 0.5 后 1.0 揭示。

    `demand=24` 时载荷 48，余量 2 → 两单都应被接受。
    `demand=25` 时载荷 50，恰好落在 `capacity - 1e-4` 边界 → 必须被拒绝。

    `recorder` 若非 None，会在**测试侧**包装 `_start_states`（不触碰生产源码），
    记录每个决策点的 (clock, 每车 cur/cur_time/load/route)，用于验证在途载重不被重复计数。
    """
    ds = {
        'coords': np.array([[[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]]], dtype=np.float32),
        'demands': np.array([[0.0, demand, demand]], dtype=np.float32),
        'tw_start': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        'tw_end': np.array([[22.0, 22.0, 22.0]], dtype=np.float32),
        'service_time': np.array([[0.0, 0.05, 0.05]], dtype=np.float32),
        'reveal_time': np.array([[1e6, 0.5, 1.0]], dtype=np.float32),   # 单1先、单2后
        'temp_class': np.array([[0, 1, 1]], dtype=np.int32),
    }
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(ds, c)
    rp = RollingOptimizeReplanner(budget=1e9, capacity=CAPACITY, booking_horizon=16.0,
                                  contract=c, cooling_share=2.0,
                                  reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, time_limit=10.0)
    if recorder is not None:
        _orig_start = rp._start_states                     # 实例级包装，仅在测试进程内生效

        def _traced(env, inst_idx, vehicles, served_mask, clock, plan):
            st = _orig_start(env, inst_idx, vehicles, served_mask, clock, plan)
            recorder.append((float(clock),
                             {vid: dict(cur=s['cur'], cur_time=float(s['cur_time']),
                                        load=float(s['load']), route=list(s['route']))
                              for vid, s in st.items()}))
            return st

        rp._start_states = _traced
    env = StrictOnlineEnv(ds, capacity=CAPACITY, num_vehicles=1, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    return rp, env, traces, ds, c


def test_accept_profitable():
    # 预算宽松，p_c=(5,10,15)：鱼收入20>p10、番茄收入10>p5，都接受（拒绝有代价）
    traces, accepted, rejected, ds, c = _run(1e9, {0: 5.0, 1: 10.0, 2: 15.0})
    assert accepted == {1, 2}, (accepted, rejected)
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 10.0, 2: 15.0})
    assert result['hard_feasible'], result['failures']
    assert abs(result['revenue'] - 30.0) < 1e-6
    print(f"[PASS] accept_profitable: accepted={sorted(accepted)}, utility={result['utility']:.2f}")
    return True


def test_certify_rejects_over_budget():
    # 硬预算认证：B=10 远小于完成任一订单的真实 C0 能耗（~91，含 dispatch+WAIT 制冷+预冷），
    # 应拒绝所有订单（certify_plan 硬认证，不是逐单 est）。
    traces, accepted, rejected, ds, c = _run(10.0, {0: 5.0, 1: 10.0, 2: 15.0})
    assert accepted == set() and rejected == {1, 2}, (accepted, rejected)
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=10.0,
                               reject_penalty_by_class={0: 5.0, 1: 10.0, 2: 15.0})
    assert result['hard_feasible'], result['failures']
    print(f"[PASS] certify_rejects_over_budget: rejected={sorted(rejected)}")
    return True


def test_in_transit_no_double_count():
    """在途载重只计一次（夹具 24/24，余量 2，避开容量边界）。

    三重断言，不只验证最终 accepted 集合：
      1. 两单都被接受；
      2. t=1.0 决策点的**起始载重恰为 24**（= 已取到的单 1 一份，不是 48）；
      3. 已服务的单 1 **不出现在剩余路线里**（committed/已服务节点不得重复计入后缀）。
    """
    rec = []
    rp, env, traces, ds, c = _run_double_count(24.0, recorder=rec)
    accepted, rejected = set(rp._accepted), set(rp._rejected)

    # 同一条 [1,2] 在 48 载荷下必须可行（对照组，证明夹具本身合法）
    # 注意：本方法末行返回 numpy.bool_，早退返回 Python False，故用真值判断而非 `is True`。
    assert rp._route_feasible(env, 0, 0, 0.0, 0.0, [1, 2]), \
        'fixture precondition: route [1,2] with load 48 must be feasible'

    assert accepted == {1, 2}, (accepted, rejected)

    second = [(t, st) for (t, st) in rec if abs(t - 1.0) < 1e-9]
    assert second, ('no decision captured at t=1.0', sorted({t for t, _ in rec}))
    for _, st in second:
        for vid, s in st.items():
            assert abs(s['load'] - 24.0) < 1e-6, \
                ('in-transit load double-counted (expected exactly one order, 24.0)', vid, s)
            assert 1 not in s['route'], \
                ('already-served order 1 re-appears in the remaining route', vid, s)

    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 10.0, 2: 15.0})
    assert result['hard_feasible'], result['failures']
    assert abs(result['revenue'] - 40.0) < 1e-6, result['revenue']
    print(f"[PASS] in_transit_no_double_count: accepted={sorted(accepted)}, "
          f"t=1.0 start_load={second[0][1]}")
    return True


def test_capacity_boundary_rejected_by_conservative_margin():
    """容量边界负控：载荷恰好等于容量时必须被拒（`l > capacity - 1e-4`）。

    这条口径是 2026-09-28 有意收紧的（接单侧检查必须严格于 float32 物理 transition，
    `_EPS=1e-9`），目的是杜绝边界载重在执行期崩溃。此用例把它钉住：
    若有人把判据改回 `capacity + 1e-6`，本用例会立即变红。
    """
    rp, env, traces, ds, c = _run_double_count(25.0)
    accepted, rejected = set(rp._accepted), set(rp._rejected)

    # 机制对照：同一条 [1,2]，载荷 50，必须不可行
    assert not rp._route_feasible(env, 0, 0, 0.0, 0.0, [1, 2]), \
        'boundary-exact route (load==capacity) must be rejected by the conservative margin'
    # 对照：单 2 单独走一条路线（空车从仓出发，载荷 25）是可行的
    # —— 说明被拒的原因只可能是"合计压线"，而不是路线本身不可行。
    assert rp._route_feasible(env, 0, 0, 0.0, 0.0, [2]), \
        'control: order 2 alone from an empty vehicle (load 25) must be feasible'

    assert accepted == {1}, ('boundary-exact route must not be accepted', accepted, rejected)
    assert rejected == {2}, (accepted, rejected)

    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 10.0, 2: 15.0})
    assert result['hard_feasible'], result['failures']
    print(f"[PASS] capacity_boundary_rejected_by_conservative_margin: "
          f"accepted={sorted(accepted)}, rejected={sorted(rejected)}, "
          f"margin={CAPACITY_MARGIN:g}")
    return True


if __name__ == '__main__':
    r = {'accept_profitable': test_accept_profitable(),
         'certify_rejects_over_budget': test_certify_rejects_over_budget(),
         'in_transit_no_double_count': test_in_transit_no_double_count(),
         'capacity_boundary_rejected_by_conservative_margin':
             test_capacity_boundary_rejected_by_conservative_margin()}
    ok = all(r.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
