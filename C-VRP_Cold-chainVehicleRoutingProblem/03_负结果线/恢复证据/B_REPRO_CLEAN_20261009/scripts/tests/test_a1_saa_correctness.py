"""A-v1 共用 SAA 修正回归测试（2026-09-26 审计 T1–T5）。

T1 未来接单收入恰好一次（无 served_rev 双计）
T2 未来揭示等待造成的截止/返仓冲突被识别（预检 + 完成阶段复查）
T3 影子完成与真实 C0 certify_plan 逐段对账（能耗一致，含在途 committed 车）
T4 场景失败不使效用变 −inf（有限 INFEASIBLE_SCENARIO_PENALTY）
T5 采样耗时计入每决策 10s 时钟（超时 = 保计划 + 拒单 + 计次）
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1, REV
from coldchain_state import create_vehicle_state, dispatch_vehicle
from run_exp_energy_c0 import certify_plan
from run_exp_reserve import (generate_dataset, BOOKING_HORIZON, SPEED_KMH,
                             KM_PER_UNIT, FUEL_COST_PER_KM)
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, ScenarioOrder,
                          build_history, make_c0_contract_v2,
                          INFEASIBLE_SCENARIO_PENALTY)
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}


def _mk_day(revealed, hidden=()):
    """手工构造 1 实例数据集（宽 TW [0,22]，revealed 揭示早、hidden 揭示晚）。"""
    orders = list(revealed) + list(hidden)
    n = 1 + len(orders)
    coords = np.zeros((1, n, 2), dtype=np.float32)
    coords[0, 0] = [0.5, 0.5]
    demands = np.zeros((1, n), dtype=np.float32)
    tw_end = np.full((1, n), 22.0, dtype=np.float32)
    service = np.full((1, n), 0.05, dtype=np.float32)
    service[0, 0] = 0.0
    reveal = np.full((1, n), 1e6, dtype=np.float32)
    tc = np.zeros((1, n), dtype=np.int32)
    tw_start = np.zeros((1, n), dtype=np.float32)
    for j, (x, y, dem, c, rev, twe) in enumerate(orders, start=1):
        coords[0, j] = [x, y]
        demands[0, j] = dem
        tc[0, j] = c
        reveal[0, j] = rev
        tw_end[0, j] = twe
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
            'service_time': service, 'reveal_time': reveal, 'temp_class': tc}


class _FakeEnv(SimpleNamespace):
    """_sim_scenario/_build_space/_complete_u 所需的最小 env 界面（inst_idx=0）。"""

    def __init__(self, ds):
        super().__init__()
        self.coords = ds['coords']
        self.dataset = ds
        self.demands = ds['demands']
        self.tw_start = ds['tw_start']
        self.tw_end = ds['tw_end']
        self.service_time = ds['service_time']
        self.reveal_time = ds['reveal_time']
        self.dist_mat = np.sqrt(
            ((ds['coords'][:, :, None, :] - ds['coords'][:, None, :, :]) ** 2).sum(-1))
        self.booking_horizon = BOOKING_HORIZON
        self.tw_speed = SPEED_KMH / KM_PER_UNIT


def _mk_replanner(budget=1e9, cooling_share=2.0, K=10, time_limit=60.0, sampler=None):
    contract = make_c0_contract_v2()
    if sampler is None:
        sampler = UncondHistoricalSampler(build_history(generate_dataset(5, 30, 20260925)))
    rp = SaaReplanner(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=cooling_share, sampler=sampler,
                      reject_penalty=P_C, K=K, time_limit=time_limit, arm_seed=7001)
    return rp, contract


def test_t1_future_revenue_counted_once():
    """未来被模拟接受的订单收入只在 plan_rev 计一次（回归：served_rev 双计）。"""
    rp, contract = _mk_replanner(budget=1e9)
    # 真实订单 1（已计划），未来订单 A
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    space, scen_orders = rp._build_space(env, 0, [[fut]])
    assert len(scen_orders) == 1 and space.n_real == 2
    st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None)}
    u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
    # 期望：route=[fut,1]（贪心首个可行位）→ plan_rev = REV[1]+REV[fut] 各一次
    d01 = float(np.hypot(0.7 - 0.5, 0.0))
    d12 = float(np.hypot(0.6 - 0.7, 0.0))
    d20 = float(np.hypot(0.5 - 0.6, 0.0))
    fuel = (d01 + d12 + d20) * KM_PER_UNIT * FUEL_COST_PER_KM
    expected = REV[0] + REV[1] - fuel
    assert abs(u - expected) < 1e-6, f"u={u} expected={expected}（双计会使 u≈expected+REV[1]）"
    assert np.isfinite(u)
    print(f"[PASS] t1_future_revenue_counted_once (u={u:.4f}, expected={expected:.4f})")
    return True


def test_t2_reveal_wait_conflicts_detected():
    """揭示等待把后续订单推出 TW/返仓超截止 → 预检拒绝插入；完成阶段复查同样拦截。"""
    rp, contract = _mk_replanner(budget=1e9)
    # 真实订单 1 已计划（TW 紧 [0,2]）；未来 A 揭示 21.95（等待后返仓超 22 截止）
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 2.0)])
    futA = ScenarioOrder(oid=-1, reveal=21.95, x=0.7, y=0.5, demand=2.0, temp_class=0,
                         tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    space, scen_orders = rp._build_space(env, 0, [[futA]])
    assert space.n_real == 2
    st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None)}
    u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
    # A 无法插入（等待其揭示后返仓超 22）→ 拒绝损失 P_C[0]=5；计划只剩真实订单 1
    fuel1 = (float(np.hypot(0.1, 0.0)) * 2) * KM_PER_UNIT * FUEL_COST_PER_KM
    expected = REV[0] - fuel1 - P_C[0]
    assert abs(u - expected) < 1e-6, f"u={u} expected={expected}（旧实现会错误接受 A）"
    # 完成阶段复查：直接给一条 reveal 等待会破坏下游 TW 的路线 → (None, None)
    futB = ScenarioOrder(oid=-2, reveal=21.9, x=0.7, y=0.5, demand=2.0, temp_class=0,
                         tw_start=0.0, tw_end=22.0, service_time=0.05)
    space2, _ = rp._build_space(env, 0, [[futB]])
    st2 = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[space2.n_real, 1], cc=None)}
    e, d = rp._complete_u(space2, env, 0, 1.0, st2, BOOKING_HORIZON)
    assert e is None and d is None, "下游 TW 被揭示等待破坏必须判不可行"
    print("[PASS] t2_reveal_wait_conflicts_detected")
    return True


def _veh(vid, status, cur, nxt, load, cc):
    return SimpleNamespace(vehicle_id=vid, status=status, current_node=cur,
                           committed_next=nxt, current_load=load,
                           coldchain_state=cc, ready_time=0.0)


def test_t3_shadow_matches_certify_plan():
    """影子 _complete_u 与真实 certify_plan 在相同状态/计划下能耗逐段一致（含在途车）。"""
    rp, contract = _mk_replanner(budget=1e9)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0), (0.4, 0.5, 2.0, 1, 0.5, 22.0)])
    env = _FakeEnv(ds)
    space, _ = rp._build_space(env, 0, [])
    served = np.zeros(ds['demands'].shape[1], dtype=bool)
    clock = 1.0
    # (a) 未 dispatch 车（idle，有计划）
    v1 = _veh(1, 'idle', 0, None, 0.0, None)
    plan1 = {1: [1]}
    st1 = rp._sim_states(env, 0, clock, [v1], served, plan1)
    e_shadow, _ = rp._complete_u(space, env, 0, clock, st1, BOOKING_HORIZON)
    ok, e_cert = certify_plan(env, 0, clock, [v1], served, plan1, contract, 1e9)
    assert ok and abs(e_shadow - e_cert) < 1e-6, f"(a) shadow={e_shadow} cert={e_cert}"
    # (b) 在途 committed 车（committed_next=1，从 current_node=0 继续，plan 尾=[2]）
    cc = dispatch_vehicle(create_vehicle_state(contract), contract)
    v2 = _veh(2, 'committed', 0, 1, 0.0, cc)
    plan2 = {2: [2]}
    st2 = rp._sim_states(env, 0, clock, [v2], served, plan2)
    assert st2[2]['route'] == [1, 2], "committed_next 必须并入影子路线（旧实现从 committed_finish 起算会漏掉在途段能耗）"
    assert st2[2]['cur'] == 0 and abs(st2[2]['cur_time'] - clock) < 1e-12
    e_shadow2, _ = rp._complete_u(space, env, 0, clock, st2, BOOKING_HORIZON)
    ok2, e_cert2 = certify_plan(env, 0, clock, [v2], served, plan2, contract, 1e9)
    assert ok2 and abs(e_shadow2 - e_cert2) < 1e-6, f"(b) shadow={e_shadow2} cert={e_cert2}"
    print(f"[PASS] t3_shadow_matches_certify_plan (E={e_shadow:.4f}/{e_cert:.4f}, "
          f"committed E={e_shadow2:.4f}/{e_cert2:.4f})")
    return True


def test_t4_no_inf_contagion():
    """不可行场景返回有限 INFEASIBLE_SCENARIO_PENALTY（无 −inf 传染）。"""
    rp, contract = _mk_replanner(budget=0.5)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    space, scen_orders = rp._build_space(env, 0, [[fut]])
    st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None)}
    u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
    assert u == INFEASIBLE_SCENARIO_PENALTY and np.isfinite(u), u
    # 全失败场景的 SAA 累加保持有限（无 inf/nan 传播到比较）
    acc = INFEASIBLE_SCENARIO_PENALTY * 10.0
    rej = INFEASIBLE_SCENARIO_PENALTY * 10.0 - 5.0 * 10.0
    assert np.isfinite(acc) and np.isfinite(rej)
    print("[PASS] t4_no_inf_contagion")
    return True


def test_t5_sampling_counts_into_time_limit():
    """采样耗时计入 10s：慢采样器 → 决策超时降级（保计划+拒单+计次）；快采样器 → 0 超时。"""
    rp, contract = _mk_replanner(budget=1e9, time_limit=0.1)

    class Slow(UncondHistoricalSampler):
        def sample(self, snapshot, rng, k):
            time.sleep(0.5)
            return [[] for _ in range(k)]

    ds = add_v2_initial_quality(
        _mk_day([(0.6, 0.5, 2.0, 1, 0.5, 22.0), (0.4, 0.5, 2.0, 0, 0.5, 22.0)]), contract)
    hist = build_history(generate_dataset(5, 30, 20260925))
    rp_slow = SaaReplanner(budget=1e9, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                           contract=contract, cooling_share=2.0, sampler=Slow(hist),
                           reject_penalty=P_C, K=10, time_limit=0.1, arm_seed=7001)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp_slow,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    traces, _ = env.run(0)
    acc, rej = set(rp_slow._accepted), set(rp_slow._rejected)
    assert acc == set() and rej == {1, 2}, (acc, rej)
    assert rp_slow.timeouts >= 2, rp_slow.timeouts
    r = evaluate_trace_a1(traces, {k: v[0] for k, v in ds.items()}, contract,
                          acc, rej, 1e9, P_C)
    assert r['hard_feasible'], r['failures']
    # 快采样器对照：同一天、宽松时限 → 无超时
    rp_fast = SaaReplanner(budget=1e9, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                           contract=contract, cooling_share=2.0,
                           sampler=UncondHistoricalSampler(hist),
                           reject_penalty=P_C, K=10, time_limit=60.0, arm_seed=7001)
    env2 = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2,
                           tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp_fast,
                           coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env2.run(0)
    assert rp_fast.timeouts == 0
    print(f"[PASS] t5_sampling_counts_into_time_limit (slow timeouts={rp_slow.timeouts})")
    return True


if __name__ == '__main__':
    res = {
        't1_future_revenue_counted_once': test_t1_future_revenue_counted_once(),
        't2_reveal_wait_conflicts_detected': test_t2_reveal_wait_conflicts_detected(),
        't3_shadow_matches_certify_plan': test_t3_shadow_matches_certify_plan(),
        't4_no_inf_contagion': test_t4_no_inf_contagion(),
        't5_sampling_counts_into_time_limit': test_t5_sampling_counts_into_time_limit(),
    }
    ok = all(res.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
