"""滚动优化器单元测试：接受/拒绝联合优化 + 预算约束 + 路线 + A-v1 评价。"""
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
    rp = RollingOptimizeReplanner(budget=budget, capacity=50.0, booking_horizon=16.0,
                                  contract=c, cooling_share=2.0,
                                  reject_penalty=p_c, time_limit=10.0)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    return traces, set(rp._accepted), set(rp._rejected), ds, c


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
    # 单车容量50，两单各需25，先后揭示：路线[1,2]可行(25+25=50)，应都接受（修在途重复计数）
    ds = {
        'coords': np.array([[[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]]], dtype=np.float32),
        'demands': np.array([[0.0, 25.0, 25.0]], dtype=np.float32),
        'tw_start': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        'tw_end': np.array([[22.0, 22.0, 22.0]], dtype=np.float32),
        'service_time': np.array([[0.0, 0.05, 0.05]], dtype=np.float32),
        'reveal_time': np.array([[1e6, 0.5, 1.0]], dtype=np.float32),  # 单1先、单2后
        'temp_class': np.array([[0, 1, 1]], dtype=np.int32),
    }
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(ds, c)
    rp = RollingOptimizeReplanner(budget=1e9, capacity=50.0, booking_horizon=16.0,
                                  contract=c, cooling_share=2.0,
                                  reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, time_limit=10.0)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=1, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    accepted, rejected = set(rp._accepted), set(rp._rejected)
    assert accepted == {1, 2}, (accepted, rejected)
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 10.0, 2: 15.0})
    assert result['hard_feasible'], result['failures']
    print(f"[PASS] in_transit_no_double_count: accepted={sorted(accepted)}")
    return True


if __name__ == '__main__':
    r = {'accept_profitable': test_accept_profitable(),
         'certify_rejects_over_budget': test_certify_rejects_over_budget(),
         'in_transit_no_double_count': test_in_transit_no_double_count()}
    ok = all(r.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
