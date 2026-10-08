"""A-v1 统一接受硬认证测试：uncond_dynamic 走 certify_plan 接单 + rolling 接入无条件未来预算门槛。"""
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
from run_exp_rule_control import UncondDynamicReplanner
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}


def _dataset(class_of_1, class_of_2=0, with_2=True):
    """depot(0,0) + node1(class_of_1, 0.6,0.5) + 可选 node2(class_of_2, 0.4,0.5)，均在 0.5 揭示。"""
    n = 3 if with_2 else 2
    coords = np.zeros((1, n, 2), dtype=np.float32)
    coords[0, 0] = [0.5, 0.5]
    coords[0, 1] = [0.6, 0.5]
    if with_2:
        coords[0, 2] = [0.4, 0.5]
    demands = np.zeros((1, n), dtype=np.float32)
    demands[0, 1] = 2.0
    if with_2:
        demands[0, 2] = 2.0
    tw_end = np.full((1, n), 22.0, dtype=np.float32)
    service = np.full((1, n), 0.05, dtype=np.float32)
    service[0, 0] = 0.0
    reveal = np.full((1, n), 1e6, dtype=np.float32)
    reveal[0, 1] = 0.5
    if with_2:
        reveal[0, 2] = 0.5
    tc = np.zeros((1, n), dtype=np.int32)
    tc[0, 1] = class_of_1
    if with_2:
        tc[0, 2] = class_of_2
    return {'coords': coords, 'demands': demands,
            'tw_start': np.zeros((1, n), dtype=np.float32),
            'tw_end': tw_end, 'service_time': service,
            'reveal_time': reveal, 'temp_class': tc}


def _future_hist(n=100, t=1.0, c=0):
    """一条训练日：n 个 class c 的未来订单（时间 t > 0.5）。n=0 表示无未来订单。"""
    return [(np.full(n, float(t)), np.full(n, int(c), dtype=int))]


def _single(ds):
    return {k: v[0] for k, v in ds.items()}


def _run_uncond(ds, c, budget, hist):
    rp = UncondDynamicReplanner(budget=budget, capacity=50.0, booking_horizon=16.0,
                                contract=c, cooling_share=2.0, hist=hist, k=10)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    return traces, set(rp._accepted), set(rp._rejected)


def _run_rolling(ds, c, budget, p_c, hist):
    rp = RollingOptimizeReplanner(budget=budget, capacity=50.0, booking_horizon=16.0,
                                  contract=c, cooling_share=2.0, reject_penalty=p_c,
                                  time_limit=5.0, hist=hist, k=10)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2, tw_speed=1.5,
                          replanner=rp, coldchain_contract=c, booking_horizon=16.0)
    traces, _ = env.run(0)
    return traces, set(rp._accepted), set(rp._rejected)


def test_uncond_certified_rejects_over_budget():
    # B=10：逐单 est（e≈8.3/4.1）会误收，真实 C0 certify（~90 kWh）必须全部拒绝。
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_dataset(1, 0), c)
    traces, accepted, rejected = _run_uncond(ds, c, 10.0, _future_hist(0))
    assert accepted == set() and rejected == {1, 2}, (accepted, rejected)
    r = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, 10.0, P_C)
    assert r['hard_feasible'], r['failures']
    print(f"[PASS] uncond_certified_rejects_over_budget: rejected={sorted(rejected)}")
    return True


def test_uncond_certified_accepts_with_budget():
    # B=1e9：certify 通过，两单都接受。
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_dataset(1, 0), c)
    traces, accepted, rejected = _run_uncond(ds, c, 1e9, _future_hist(0))
    assert accepted == {1, 2} and rejected == set(), (accepted, rejected)
    r = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, 1e9, P_C)
    assert r['hard_feasible'], r['failures']
    print(f"[PASS] uncond_certified_accepts_with_budget: accepted={sorted(accepted)}")
    return True


def test_uncond_lam_rejects_low_ratio():
    # 单条鱼（REV/e≈2.41）。hist=100 个未来 class0（REV/e≈2.44）→ lam≈2.44 → 门槛拒绝。
    # 对照（无未来订单，lam=0）：同一 B=150 下 certify 通过 → 接受。
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_dataset(1, with_2=False), c)
    traces, accepted, rejected = _run_uncond(ds, c, 150.0, _future_hist(100, 1.0, 0))
    assert accepted == set() and rejected == {1}, (accepted, rejected)
    ds2 = add_v2_initial_quality(_dataset(1, with_2=False), c)
    traces2, accepted2, rejected2 = _run_uncond(ds2, c, 150.0, _future_hist(0))
    assert accepted2 == {1} and rejected2 == set(), (accepted2, rejected2)
    print(f"[PASS] uncond_lam_rejects_low_ratio: with_hist rejected={sorted(rejected)}, "
          f"control accepted={sorted(accepted2)}")
    return True


def test_rolling_reserve_threshold():
    # 单条 class2 鱼（REV/e≈2.01）。hist=100 个未来 class0（lam≈2.44）→ 预拒绝留预算。
    # 对照（hist=None，lam=0）：同一 B=150 下 certify 通过 → 搜索接受（拒绝确实来自门槛）。
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_dataset(2, with_2=False), c)
    traces, accepted, rejected = _run_rolling(ds, c, 150.0, P_C, _future_hist(100, 1.0, 0))
    assert accepted == set() and rejected == {1}, (accepted, rejected)
    ds2 = add_v2_initial_quality(_dataset(2, with_2=False), c)
    traces2, accepted2, rejected2 = _run_rolling(ds2, c, 150.0, P_C, None)
    assert accepted2 == {1} and rejected2 == set(), (accepted2, rejected2)
    print(f"[PASS] rolling_reserve_threshold: with_hist rejected={sorted(rejected)}, "
          f"control accepted={sorted(accepted2)}")
    return True


if __name__ == '__main__':
    r = {'uncond_certified_rejects_over_budget': test_uncond_certified_rejects_over_budget(),
         'uncond_certified_accepts_with_budget': test_uncond_certified_accepts_with_budget(),
         'uncond_lam_rejects_low_ratio': test_uncond_lam_rejects_low_ratio(),
         'rolling_reserve_threshold': test_rolling_reserve_threshold()}
    ok = all(r.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
