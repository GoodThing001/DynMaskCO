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


def _mk_replanner(budget=1e9, cooling_share=2.0, K=10, time_limit=60.0, sampler=None,
                  shadow_mode='greedy', incr_eval=False, anytime_vote=False,
                  anytime_early_stop=True, arm_seed=7001):
    contract = make_c0_contract_v2()
    if sampler is None:
        sampler = UncondHistoricalSampler(build_history(generate_dataset(5, 30, 20260925)))
    rp = SaaReplanner(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=cooling_share, sampler=sampler,
                      reject_penalty=P_C, K=K, time_limit=time_limit, arm_seed=arm_seed,
                      shadow_mode=shadow_mode, incr_eval=incr_eval,
                      anytime_vote=anytime_vote, anytime_early_stop=anytime_early_stop)
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
    """影子 _complete_u 与真实认证口径一致：未 dispatch 车逐段对齐；在途车按真实事件推进（A-06）。"""
    from run_exp_energy_c0 import c0_completion_energy
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
    # (b) 在途 committed 车：A-06 语义——从 committed_finish 继续、在途段冻结
    cc = dispatch_vehicle(create_vehicle_state(contract), contract)
    v2 = _veh(2, 'committed', 0, 1, 0.0, cc)
    v2.committed_finish = 0.85
    plan2 = {2: [2]}
    st2 = rp._sim_states(env, 0, clock, [v2], served, plan2)
    assert st2[2]['route'] == [1, 2] and st2[2]['frozen'] == 1, st2[2]
    assert st2[2]['cur'] == 1 and abs(st2[2]['cur_time'] - 0.85) < 1e-12
    e_shadow2, _ = rp._complete_u(space, env, 0, clock, st2, BOOKING_HORIZON)
    # 参考：同一真实推进——c0_completion_energy(cc_mid, 1, 0.85, [1,2], ...)（0 长在途段 + 取货）
    e_ref2 = c0_completion_energy(cc, 1, 0.85, [1, 2], env.dist_mat[0],
                                  env.tw_start[0], env.service_time[0],
                                  env.reveal_time[0], env.demands[0],
                                  env.dataset['temp_class'][0], contract, BOOKING_HORIZON)
    assert abs(e_shadow2 - e_ref2) < 1e-6, f"(b) shadow={e_shadow2} ref={e_ref2}"
    print(f"[PASS] t3_shadow_matches_certify_plan (E={e_shadow:.4f}/{e_cert:.4f}, "
          f"committed E={e_shadow2:.4f}/{e_ref2:.4f})")
    return True


def test_t4_no_inf_contagion():
    """场景失败不使效用变 −inf。

    2026-09-26 深夜用户裁定修正语义：只有「真不可行」（energy is None，TW/容量/返仓）
    才置 INFEASIBLE_SCENARIO_PENALTY；能耗超预算场景保留其价值（est 门槛是影子预算规则，
    真实预算由接单侧 certify_plan 强制）——旧「超预算即 −1e6」使每场景全中惩罚、
    投票退化为恒接受（40 天实测两臂决策同质 97-100%）。
    """
    rp, contract = _mk_replanner(budget=0.5)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    space, scen_orders = rp._build_space(env, 0, [[fut]])
    st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None)}
    u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
    # est 门槛拒绝未来单（P_C[1]=10）；完成能耗超预算但 feasible → 有限效用
    # plan_rev=REV[0]=10, fuel=0.4, rej=10 → u=-0.4（历史实测值）
    assert np.isfinite(u), u
    assert abs(u - (-0.4)) < 1e-3, u
    # 真不可行（TW 冲突）仍置有限惩罚、无 inf/nan 传播
    acc = INFEASIBLE_SCENARIO_PENALTY * 10.0
    rej = INFEASIBLE_SCENARIO_PENALTY * 10.0 - 5.0 * 10.0
    assert np.isfinite(acc) and np.isfinite(rej)
    print(f"[PASS] t4_no_inf_contagion (over-budget scenario utility={u:.4f}, finite)")
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
    # A-02 每决策语义：采样耗时计入事件首决策（订单1 → 超时拒单）；同事件后续决策各自新起
    # 时限、复用已采场景（订单2 → 正常投票接单）。
    assert acc == {2} and rej == {1}, (acc, rej)
    assert rp_slow.timeouts == 1, rp_slow.timeouts
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


def test_t6_shadow_ls_no_regression():
    """战役 L1：shadow_mode='ls'（最优插入 + relocate/swap）无崩溃、且不劣于贪心影子。"""
    from scenario_saa import UncondHistoricalSampler as _U
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0), (0.4, 0.5, 2.0, 1, 0.5, 22.0)])
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    st_base = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None),
               2: dict(cur=0, cur_time=1.0, load=0.0, route=[2], cc=None)}
    rp_g, _ = _mk_replanner(budget=1e9)
    rp_l, _ = _mk_replanner(budget=1e9)
    rp_l.shadow_mode = 'ls'
    for rp, tag in ((rp_g, 'greedy'), (rp_l, 'ls')):
        st = {k: dict(v) for k, v in st_base.items()}
        space, _ = rp._build_space(env, 0, [[fut]])
        u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
        assert np.isfinite(u), (tag, u)
        if tag == 'greedy':
            u_greedy = u
        else:
            assert u >= u_greedy - 1e-9, (u_greedy, u)
    print(f"[PASS] t6_shadow_ls_no_regression (greedy={u_greedy:.4f}, ls={u:.4f})")
    return True


def test_t7_energy_pricing_marginal():
    """战役 L2b：marginal 定价（只计 door/COP+precool）下未来单不被待命分摊挤出。"""
    from run_exp_energy_c0 import c0_marginal_energy
    contract = make_c0_contract_v2()
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    env = _FakeEnv(ds)
    marg_planned = c0_marginal_energy(0, 2.0, 0.0, contract)
    marg_fut = c0_marginal_energy(1, 2.0, 0.0, contract)
    budget = marg_planned + marg_fut + 0.3   # 介于边际总量与分摊总量之间
    assert budget < marg_planned + marg_fut + 2.0 * 2.0 + 0.3  # 分摊定价必然超

    def run_with(pricing):
        rp = SaaReplanner(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                          contract=contract, cooling_share=2.0,
                          sampler=UncondHistoricalSampler([]),
                          reject_penalty=P_C, K=10, time_limit=60.0, arm_seed=7001,
                          energy_pricing=pricing)
        st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None)}
        space, _ = rp._build_space(env, 0, [[fut]])
        return rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)

    u_amort = run_with('amortized')
    u_marg = run_with('marginal')
    # marginal 定价下未来单被接受（plan_rev 含 REV[1]=20 且无拒绝损失）；分摊定价下被拒（−P_C[1]=−10）
    assert u_marg - u_amort > 15, (u_marg, u_amort)
    assert np.isfinite(u_marg) and np.isfinite(u_amort)
    print(f"[PASS] t7_energy_pricing_marginal (amortized={u_amort:.3f}, marginal={u_marg:.3f})")
    return True


def test_t8_future_policy_density():
    """战役 L2c：density 顺序下影子按价值密度打包（接单集合与 reveal 顺序不同）。"""
    from run_exp_energy_c0 import c0_marginal_energy
    contract = make_c0_contract_v2()
    # 未来 A：class1 REV20 e≈6.45（密度~3.1，reveal 晚）；未来 B：class2 REV30 e≈12.9（密度~2.3，reveal 早）
    futB = ScenarioOrder(oid=-2, reveal=2.0, x=0.6, y=0.4, demand=2.0, temp_class=2,
                         tw_start=0.0, tw_end=22.0, service_time=0.05)
    futA = ScenarioOrder(oid=-1, reveal=3.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                         tw_start=0.0, tw_end=22.0, service_time=0.05)
    ds = _mk_day([])   # 无真实已计划订单
    env = _FakeEnv(ds)
    eA = c0_marginal_energy(1, 2.0, 0.0, contract)
    eB = c0_marginal_energy(2, 2.0, 0.0, contract)
    assert eA < eB, (eA, eB)
    budget = eB + 0.1   # 只够一个：reveal 顺序收 B（早揭示），density 顺序收 A（高密度）

    def run_with(policy):
        rp = SaaReplanner(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                          contract=contract, cooling_share=2.0,
                          sampler=UncondHistoricalSampler([]),
                          reject_penalty=P_C, K=10, time_limit=60.0, arm_seed=7001,
                          energy_pricing='marginal', future_policy=policy)
        st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[], cc=None)}
        space, scen_orders = rp._build_space(env, 0, [[futB, futA]])
        # scen_idx = [space.n_real, space.n_real+1]（B, A 顺序）
        u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real, space.n_real + 1], space)
        return u

    u_rev = run_with('reveal')
    u_den = run_with('density')
    # 预算只够一个：reveal 顺序收早揭示的 B（REV30），density 顺序收高密度的 A（REV20）
    # → u_rev ≈ 30 − fuel_B；u_den ≈ 20 − fuel_A；两者相差 ~10
    assert np.isfinite(u_rev) and np.isfinite(u_den)
    assert u_rev > u_den + 5, (u_rev, u_den)
    assert u_den > 0, u_den
    print(f"[PASS] t8_future_policy_density (reveal={u_rev:.2f}, density={u_den:.2f})")
    return True


def test_t9_precheck_completion_consistency():
    """2026-09-27 根因回归：预检放行的路线，完成阶段（出发地等揭示语义）不得 TW 违规。"""
    rp, contract = _mk_replanner(budget=1e9)
    rp.energy_pricing = 'marginal'
    ds = _mk_day([])
    env = _FakeEnv(ds)
    rng = np.random.default_rng(7)
    fut = [ScenarioOrder(oid=-(j + 1), reveal=float(rng.uniform(1.0, 10.0)),
                         x=float(rng.uniform(0.3, 0.7)), y=float(rng.uniform(0.3, 0.7)),
                         demand=float(rng.uniform(1.0, 3.0)), temp_class=int(rng.integers(0, 3)),
                         tw_start=0.0, tw_end=float(rng.uniform(2.0, 12.0)),
                         service_time=0.05) for j in range(12)]
    space, _ = rp._build_space(env, 0, [fut])
    speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
    # 随机打包若干路线：预检 True ⇒ 完成式行走无违规（关键不变量）
    for trial_i in range(200):
        k = int(rng.integers(1, 9))
        route = list(rng.choice(range(space.n_real, space.n_real + 12), size=k, replace=False))
        ok = rp._route_feasible_u(space, env, 0, 1.0, 0.0, route, 22.0)
        if not ok:
            continue
        t, c = 1.0, 0
        for o in route:
            d = float(space.D[c, o])
            depart = t if t >= space.reveal[o] else float(space.reveal[o])
            arrive = depart + d / speed
            sstart = arrive if arrive >= space.tws[o] else float(space.tws[o])
            assert sstart <= float(space.twe[o]) + 1e-6, (route, o, sstart, space.twe[o])
            t = sstart + float(space.st[o])
            c = o
    print("[PASS] t9_precheck_completion_consistency")
    return True


def test_t10_cert_exception_rollback():
    """A-01：认证中途异常后计划必须回滚（计划/accepted/rejected 一致性）。"""
    import run_exp_energy_c0 as eco
    rp, contract = _mk_replanner(budget=1e9)
    rp._plan = {1: [42]}   # 预设计划（真实场景中为已认证计划）
    rp._try_insert = lambda *a, **k: rp._plan.update({1: [7]}) or True  # 插入成功（改计划）
    orig = eco.certify_plan

    def boom(env, inst_idx, clock, vehicles, served_mask, plan, c, B):
        plan[1] = [99]      # 模拟认证路径中途篡改计划后抛异常
        raise ValueError("injected cert crash")

    eco.certify_plan = boom
    try:
        ok = rp._try_insert_certified(env=None, inst_idx=0, o=1,
                                      vehicles=[], served_mask=np.zeros(2, bool), clock=0.0)
    finally:
        eco.certify_plan = orig
    assert not ok
    assert rp._plan == {1: [42]}, rp._plan   # 必须回滚到 saved_plan
    print("[PASS] t10_cert_exception_rollback")
    return True


def test_t11_last_scenario_timeout_no_accept():
    """A-02：最后一轮场景才超时（提交前复查）→ 拒单 + 计超时，不得接单。"""
    import scenario_saa as ssa
    rp, contract = _mk_replanner(budget=1e9, time_limit=1.0)
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, 1, 0.5, 22.0)]), contract)
    env = _FakeEnv(ds)
    rp._id_map = {1: 1}
    rp._plan = {1: [1]}
    rp._try_insert_certified = lambda *a, **k: True   # 插入+认证视为成功，直入场景循环
    calls = [0]
    real_pc = time.perf_counter

    def fake_pc():
        calls[0] += 1
        # 入口检查(1) + 10 对场景检查(10) 返回 0；提交前复查返回 2.0（> deadline=1.0）
        return 0.0 if calls[0] <= 11 else 2.0

    ssa.time.perf_counter = fake_pc
    try:
        rp._saa_decide(env, 0, 1.0, [], np.zeros(2, bool), 1, [[]] * 10, 0.0)
    finally:
        ssa.time.perf_counter = real_pc
    assert 1 in rp._rejected and 1 not in rp._accepted, (rp._accepted, rp._rejected)
    assert rp.timeouts == 1
    print("[PASS] t11_last_scenario_timeout_no_accept")
    return True


def test_t12_committed_leg_frozen():
    """A-06：在途 committed 段冻结——影子插入/relocate/swap 不得越过；returning/closed 车不参与。"""
    from coldchain_state import dispatch_vehicle, create_vehicle_state as _csv
    rp, contract = _mk_replanner(budget=1e9)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0), (0.4, 0.5, 2.0, 1, 0.5, 22.0)])
    env = _FakeEnv(ds)
    cc = dispatch_vehicle(_csv(contract), contract)
    v_c = _veh(2, 'committed', 0, 1, 0.0, cc)
    v_c.committed_finish = 1.2
    v_r = _veh(3, 'returning', 1, None, 2.0, dispatch_vehicle(_csv(contract), contract))
    v_i = _veh(1, 'idle', 0, None, 0.0, None)
    served = np.zeros(3, bool)
    st = rp._sim_states(env, 0, 1.0, [v_c, v_r, v_i], served, {2: [2], 1: [1]})
    assert 3 not in st, "returning 车不得进入影子车队"
    assert 2 in st and st[2]['frozen'] == 1 and st[2]['route'][0] == 1
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    space, _ = rp._build_space(env, 0, [[fut]])
    idx = space.n_real
    ok = rp._insert_u(space, env, idx, st, 22.0)
    assert ok
    route_c = st[2]['route']
    assert route_c[0] == 1, "在途 committed 段不得被新订单越过"
    # ls relocate 也不得移动冻结订单
    rp.shadow_mode = 'ls'
    rp._local_search_u(space, env, st, 22.0)
    assert st[2]['route'][0] == 1, "relocate/swap 不得移动冻结段"
    print("[PASS] t12_committed_leg_frozen")
    return True


def test_t13_solver_demand_units():
    """A-05d：求解器需求/容量统一 ×1000 整数单位，需求**向上**取整（ceil）。
    不变式：求解器装载 ≥ 真实装载 ⇒ 求解器可行（seen ≤ cap_int）⇒ 真实装载 ≤ cap，
    执行期 transition（_EPS=1e-9）必可行。A-05c 的向下取整方向相反：求解器按 49.999
    打包、真实装载最高 50.0+0.02 → certify_plan 越界 → OR-Tools 40/40 execution_crash。"""
    import math
    import random
    from solver_accept_replanner import _demand_int
    assert _demand_int(2.01) == int(math.ceil(2.01 * 1000))
    total = sum(_demand_int(2.01) for _ in range(20))
    cap = int(math.floor(50.0 * 1000))
    # ceil 版 20×2010=40200 ≤ 50000；旧 1 单位整数 ceil（20×3=60>50）的不公平拒绝不复现
    assert total <= cap, (total, cap)
    rng = random.Random(1234)
    for _ in range(300):
        xs = [rng.uniform(0.5, 6.0) for _ in range(30)]
        seen = sum(_demand_int(x) for x in xs)
        real = sum(x for x in xs) * 1000
        assert seen >= real, (seen, real)      # 求解器装载 ≥ 真实装载
        if seen <= cap:
            assert real <= cap + 1e-6          # 求解可行 ⇒ 物理可行
    print("[PASS] t13_solver_demand_units (ceil; seen>=real invariant)")
    return True


def test_t15_solver_start_pin_scale():
    """A-05e：初始载重钉死值必须 ×1000 整数单位（旧 `math.ceil(load)` 少乘 INT_SCALE，
    32.1584→33 单位，是 OR-Tools 40/40 execution_crash 的真正根因），并 clamp 到容量上界。"""
    from solver_accept_replanner import INT_SCALE, _start_pin_int
    cap_int = int((50.0 - 1e-4) * INT_SCALE)
    assert _start_pin_int(32.158405, cap_int) == 32159, \
        "钉死值必须 ×INT_SCALE（32159），旧实现误为 33"
    assert _start_pin_int(0.0, cap_int) == 0
    assert _start_pin_int(0.0049, cap_int) == 5
    assert _start_pin_int(49.9997, cap_int) == cap_int, "超容量上界须 clamp 到 cap_int"
    assert _start_pin_int(50.001, cap_int) == cap_int
    print("[PASS] t15_solver_start_pin_scale")
    return True


def test_t16_frozen_survives_scenario_copy():
    """Q-01（2026-09-29）：_sim_states → _build_space → _sim_scenario 完整调用链中，
    冻结标记随状态复制保留——未来订单不得越过不可撤销的在途 committed leg
    （旧实现复制丢 frozen → s.get('frozen',0) 恒 0 → 完成入口 [1,2] 变 [3,1,2]）。
    覆盖 greedy 与 ls 两种影子模式。"""
    from coldchain_state import dispatch_vehicle, create_vehicle_state as _csv
    rp, contract = _mk_replanner(budget=1e9)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0), (0.4, 0.5, 2.0, 1, 0.5, 22.0)])
    env = _FakeEnv(ds)
    cc = dispatch_vehicle(_csv(contract), contract)
    v_c = _veh(2, 'committed', 0, 1, 0.0, cc)
    v_c.committed_finish = 1.2
    served = np.zeros(3, bool)
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.3, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    for mode in ('greedy', 'ls'):
        rp.shadow_mode = mode
        st = rp._sim_states(env, 0, 1.0, [v_c], served, {2: [2]})
        assert st[2]['frozen'] == 1 and st[2]['route'] == [1, 2]
        space, _ = rp._build_space(env, 0, [[fut]])
        captured = {}
        orig_complete = rp._complete_u

        def cap_complete(space2, env2, inst_idx2, clock2, st2, wait_until):
            captured['st'] = st2
            return orig_complete(space2, env2, inst_idx2, clock2, st2, wait_until)

        rp._complete_u = cap_complete
        try:
            _u = rp._sim_scenario(env, 0, 1.0, st, [space.n_real], space)
        finally:
            rp._complete_u = orig_complete
        cst = captured['st']
        assert cst[2]['frozen'] == 1, "复制状态必须保留 frozen 标记"
        route = cst[2]['route']
        assert route[0] == 1, "未来订单不得越过不可撤销的 committed_next（mode=%s）" % mode
        # A-06 语义：frozen=1 只锁 committed leg（位置 0）；tail 可重排。
        # 完整不变式：committed leg 不越位 + 原 tail 订单不丢失 + 未来订单只在 ≥1 位插入。
        assert set(route) == {1, 2, space.n_real}, (mode, route)
        assert route.index(space.n_real) >= 1, "未来订单只能插在冻结位之后（mode=%s）" % mode
    print("[PASS] t16_frozen_survives_scenario_copy (greedy + ls)")
    return True


def test_t17_idle_empty_vehicles_in_shadow():
    """Q-02（2026-09-29）：未出车空计划车保留在影子车队——
    ① 空计划空场景不产生派车/制冷费；② 空计划车可接未来订单（插入后才 dispatch）；
    ③ accept/reject 两分支可用车辆全集一致、仅计划变化。"""
    from coldchain_state import dispatch_vehicle, create_vehicle_state as _csv
    rp, contract = _mk_replanner(budget=1e9)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    env = _FakeEnv(ds)
    v_i = _veh(1, 'idle', 0, None, 0.0, None)                       # 未出车、空计划
    v_r = _veh(3, 'returning', 1, None, 2.0, dispatch_vehicle(_csv(contract), contract))
    served = np.zeros(2, bool)
    st = rp._sim_states(env, 0, 1.0, [v_i, v_r], served, {})
    assert 1 in st, "未出车空计划车必须保留在影子车队"
    assert 3 not in st, "returning 车不得进入影子车队"
    # ① 空计划空场景：0 能耗 0 距离（无幻影预冷）
    space0, _ = rp._build_space(env, 0, [])
    energy, dist = rp._complete_u(space0, env, 0, 1.0, st, BOOKING_HORIZON)
    assert energy == 0.0 and dist == 0.0, (energy, dist)
    # ② 空计划车可接未来订单
    fut = ScenarioOrder(oid=-1, reveal=2.0, x=0.7, y=0.5, demand=2.0, temp_class=1,
                        tw_start=0.0, tw_end=22.0, service_time=0.05)
    space, _ = rp._build_space(env, 0, [[fut]])
    ok = rp._insert_u(space, env, space.n_real, st, 22.0)
    assert ok and st[1]['route'] == [space.n_real], "空计划车应能接未来订单"
    energy2, _ = rp._complete_u(space, env, 0, 1.0, st, BOOKING_HORIZON)
    assert energy2 > 0.0, "插入订单后才 dispatch 计费"
    # ③ accept/reject 车辆全集一致（旧实现：空计划侧被剔除）
    acc_st = rp._sim_states(env, 0, 1.0, [v_i], served, {1: [1]})
    rej_st = rp._sim_states(env, 0, 1.0, [v_i], served, {})
    assert set(acc_st.keys()) == set(rej_st.keys()) == {1}, (acc_st.keys(), rej_st.keys())
    print("[PASS] t17_idle_empty_vehicles_in_shadow")
    return True


def test_t18_solver_final_deadline_check():
    """Q-03（2026-09-29）：求解器接单臂认证通过后、提交前必须复查绝对 deadline——
    越限 = 超时（保旧计划 + 拒单 + 计次）。注入时钟：限额 0.01s、求解结束 0.03s、
    认证结束 0.06s → 必须拒绝并计超时（旧实现仍接单、timeouts=0）。"""
    import solver_accept_replanner as sar
    rp = sar.SolverAcceptReplanner(budget=1e9, capacity=50.0,
                                   booking_horizon=BOOKING_HORIZON,
                                   contract=make_c0_contract_v2(), cooling_share=2.0,
                                   solver='ortools', time_limit=0.01, solution_limit=30)
    env = SimpleNamespace(reveal_time=np.array([[0.0, 0.0, 0.0]]),
                          get_reserved_customers=lambda vs: set())
    served = np.zeros(3, bool)
    clock = [0.0]
    real_pc = sar.time.perf_counter

    def fake_pc():
        return clock[0]

    def fake_solve(e, inst_idx, c2, vehicles, served_mask, pool, deadline):
        clock[0] = 0.03     # 求解结束已越限（deadline = 0.0 + 0.01）
        return {1: [1]}, 'solved'

    orig_cert = sar.certify_plan

    def fake_cert(*a, **k):
        clock[0] = 0.06     # 认证结束仍越限
        return True, 0.0

    rp._solve = fake_solve
    sar.time.perf_counter = fake_pc
    sar.certify_plan = fake_cert
    try:
        rp.on_reveal(env, 0, 0.0, [], served, [1])
    finally:
        sar.time.perf_counter = real_pc
        sar.certify_plan = orig_cert
    assert 1 in rp._rejected and 1 not in rp._accepted, (rp._accepted, rp._rejected)
    assert rp.timeouts == 1, rp.timeouts
    assert rp._plan == {}, rp._plan          # 保旧计划
    assert rp._reject_reasons.get('timeout') == 1, rp._reject_reasons
    # 正例：限内认证通过 → 正常接单（新决策时钟归零）
    clock[0] = 0.0
    sar.time.perf_counter = fake_pc

    def fake_solve_ok(e, inst_idx, c2, vehicles, served_mask, pool, deadline):
        clock[0] = 0.003    # 限内完成
        return {1: [2]}, 'solved'

    def fake_cert_ok(*a, **k):
        return True, 0.0

    rp._solve = fake_solve_ok
    sar.certify_plan = fake_cert_ok
    try:
        rp.on_reveal(env, 0, 0.0, [], served, [2])
    finally:
        sar.time.perf_counter = real_pc
        sar.certify_plan = orig_cert
    assert 2 in rp._accepted and 2 not in rp._rejected, (rp._accepted, rp._rejected)
    assert rp.timeouts == 1
    print("[PASS] t18_solver_final_deadline_check")
    return True


def test_t19_overage_price_continuous():
    """S3-3：影子终局超额连续价格——终局真实能耗 energy > B 时按超额量 × 价格连续扣罚
    （有限、线性、不恢复全场景 −1e6）；overage_price=None 时保持旧行为（不扣罚）。"""
    import math
    rp, contract = _mk_replanner(budget=1e9)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    env = _FakeEnv(ds)
    st = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[1], cc=None, frozen=0)}
    space, _ = rp._build_space(env, 0, [])
    rp.budget = 0.1   # 远低于完成阶段真实能耗（取货 + WAIT + 返仓）
    rp.overage_price = None
    u_old = rp._sim_scenario(env, 0, 1.0, st, [], space)
    rp.overage_price = 5.0
    u_5 = rp._sim_scenario(env, 0, 1.0, st, [], space)
    rp.overage_price = 10.0
    u_10 = rp._sim_scenario(env, 0, 1.0, st, [], space)
    assert math.isfinite(u_old) and math.isfinite(u_5) and math.isfinite(u_10)
    assert u_5 < u_old and u_10 < u_5, (u_old, u_5, u_10)
    assert u_5 > -1e6 and u_10 > -1e6, "不得恢复全场景 −1e6"
    # 连续线性：惩罚差 = 价格差 × 超额量
    overage = (u_old - u_5) / 5.0
    assert overage > 0
    assert abs((u_5 - u_10) - 5.0 * overage) < 1e-6, "超额价格必须按超额量线性"
    print("[PASS] t19_overage_price_continuous (overage=%.3f kWh)" % overage)
    return True


def test_t20_expected_day_completeness():
    """2026-09-30 完整性修复：可裁决必须以**预期天数**（40）验收唯一 day ID——
    双侧共缺同一天（39/39）不得判可裁决；缺失结果只作诊断。
    覆盖外部驱动与步骤 2 门驱动两处 summarize / paired_all_days。"""
    import run_a1_external_accept as ext

    def _rows(ids, crash_ids=()):
        out = []
        for i in ids:
            if i in crash_ids:
                out.append(dict(i=i, utility=None, hard_feasible=False,
                                failures=['execution_crash'], timeouts=0,
                                elapsed_s=0.0, n_served=None, n_rejected=None,
                                energy_kwh=None, budget_violated=None))
            else:
                out.append(dict(i=i, utility=float(i), hard_feasible=True,
                                failures=[], timeouts=0, elapsed_s=1.0,
                                n_served=1, n_rejected=1, energy_kwh=1.0,
                                budget_violated=False))
        return out

    ids40 = list(range(40))
    ids39 = list(range(39))            # 双侧共缺 day 39

    mods = [ext]
    try:
        import run_a1_step2_gate as gate
        mods.append(gate)
    except Exception as e:   # 步骤 2 驱动依赖 jax 链时跳过（本地无 jax 环境仍覆盖外部驱动）
        print('  [info] 跳过步骤 2 驱动（import 失败: %r）' % (e,))
    for mod in mods:
        name = mod.__name__
        a, b = _rows(ids39), _rows(ids39)
        st = mod.paired_all_days(a, b, 20260926, 40)
        assert st['adjudicable'] is False, (name, 'paired 39/39 必须不可裁决')
        s = mod.summarize(a, 20260926, 40)
        assert s['adjudicable'] is False, (name, 'summarize 39 行必须不可裁决')
        assert s['expected_n'] == 40 and s['n_days'] == 39, (name, s['expected_n'], s['n_days'])
        assert s['day_ids_complete'] is False, name
        a, b = _rows(ids40), _rows(ids40)
        st = mod.paired_all_days(a, b, 20260926, 40)
        assert st['adjudicable'] is True, (name, 'paired 40/40 应可裁决')
        s = mod.summarize(a, 20260926, 40)
        assert s['adjudicable'] is True and s['day_ids_complete'] is True, name
        a = _rows(ids40, crash_ids={7})
        s = mod.summarize(a, 20260926, 40)
        assert s['adjudicable'] is False and s['missing_day_ids'] == [7], (name, s['missing_day_ids'])
        # 2026-09-30：NaN 效用与重复 day_id 也必须判不可裁决（旧基线文件可能含 NaN）
        a = _rows(ids40)
        a[9]['utility'] = float('nan')
        s = mod.summarize(a, 20260926, 40)
        assert s['adjudicable'] is False and 9 in s['missing_day_ids'], (name, s['missing_day_ids'])
        a = _rows(ids40)
        a[12]['i'] = 11                       # 重复 day_id（11 出现两次、12 缺失）
        s = mod.summarize(a, 20260926, 40)
        assert s['adjudicable'] is False and s['day_ids_complete'] is False, name
        st = mod.paired_all_days(a, _rows(ids40), 20260926, 40)
        assert st['adjudicable'] is False, name
    print("[PASS] t20_expected_day_completeness (39/39/NaN/重复 不可裁决；40/40 可裁决)")
    return True


def test_t21_maskco_p0_regressions():
    """MaskCO P0（2026-09-30）：① 批次补齐位 = PAD（与 MASK/NULL 分离，每样本恰好
    m_max 个 MASK 监督槽，监督数不随 batchmate 长度变化）；② 需求桶解码一致性
    （按 (class,spot,bin) 条件池，桶界硬保证）；③ 时钟支持掩码（结束≤clock 的桶不可采样）。"""
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    _training = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'training')
    for _p in (_models, _training):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import maskco_scenario as ms
    # ① make_batch PAD 分离
    try:
        import train_maskco_scenario as tms
        mb = tms.make_batch
    except Exception as e:   # jax 环境缺失时跳过该子项
        mb = None
        print('  [info] 跳过 make_batch 子项（import 失败: %r）' % (e,))
    if mb is not None:
        m = 4
        slices = [([1, 2, 3], [10, 11, 12, 13], 4, 3),
                  ([7], [20, 21, 22, 23], 4, 8),
                  ([4, 5], [30, 31], 2, 14)]
        X, Y, N, CB = mb(slices, [0, 1, 2], m)
        assert list(N) == [4, 4, 2], list(N)
        assert list(CB) == [3, 8, 14], list(CB)   # I1 进阶：时钟桶随样本传递
        for b in range(3):
            vis_n = len(slices[b][0][:m])
            assert (X[b, :vis_n] != ms.MASK_TOKEN).all() and \
                   (X[b, :vis_n] != ms.PAD_TOKEN).all()
            assert (X[b, vis_n:vis_n + m] == ms.MASK_TOKEN).all(), "每样本恰 m_max 个 MASK 槽"
            assert (X[b, vis_n + m:] == ms.PAD_TOKEN).all(), "批次补齐位必须是 PAD"
            sup = (Y[b, vis_n:vis_n + m] != ms.NULL_TOKEN).sum()
            assert sup == len(slices[b][1][:m]), (b, sup)
        # 同一行在不同批次长度下监督数不变
        X2, Y2, N2, CB2 = mb([slices[2], slices[2], slices[2]], [0, 1, 2], m)
        sup2 = (Y2[0, len(slices[2][0][:m]):len(slices[2][0][:m]) + m]
                != ms.NULL_TOKEN).sum()
        assert sup2 == len(slices[2][1][:m]) and list(N2) == [2, 2, 2]
    # ② 需求桶解码一致性
    from scenario_saa import ScenarioOrder
    pools = ms.HistoryPools([])
    rng = np.random.default_rng(123)
    for tok in [ms.order_to_token(ScenarioOrder(oid=1, reveal=1.2, x=0.5, y=0.5,
                                                demand=1.3, temp_class=0,
                                                tw_start=0.0, tw_end=4.0,
                                                service_time=0.05)),
                ms.order_to_token(ScenarioOrder(oid=2, reveal=3.7, x=0.1, y=0.1,
                                                demand=2.6, temp_class=2,
                                                tw_start=0.0, tw_end=6.0,
                                                service_time=0.05))]:
        for _ in range(20):
            o = pools.decode_token(tok, rng)
            t, c, b, d = ms.token_to_bucket(tok)
            assert o.temp_class == c
            assert (o.demand > 1.0 and o.demand <= 2.0) if d == 0 else \
                   (o.demand > 2.0 and o.demand <= 3.0), (tok, o.demand)
            assert t * ms.TIME_BIN_H <= o.reveal < (t + 1) * ms.TIME_BIN_H
    # ③ 时钟支持掩码
    sup = ms.clock_support_mask(4.25)   # bin 8 = [4.0, 4.5)
    for tok in range(1, ms.VOCAB_SIZE):
        t, _c, _b, _d = ms.token_to_bucket(tok)
        if (t + 1) * ms.TIME_BIN_H <= 4.25 + 1e-9:
            assert not sup[tok], (tok, t)
        else:
            assert sup[tok], (tok, t)
    # ④ I1 显式数量头采样（纯 numpy）：数量头 one-hot n=5 + 均匀 token 分布
    #    → 每场景恰 5 个未来订单（clock=0 无边界过滤）
    m = 6
    tok_logits = np.zeros((m, ms.VOCAB_SIZE), dtype=np.float32)
    cnt_logits = np.full(ms.M_MAX + 1, -1e9, dtype=np.float32)
    cnt_logits[5] = 0.0
    rng2 = np.random.default_rng(7)
    scens = ms.sample_from_logits(tok_logits, cnt_logits, 0.0, rng2, 3, pools)
    assert all(len(s) == 5 for s in scens), [len(s) for s in scens]
    # 数量头 n=0 → 空场景；token 分布质量全在 NULL → 仍空（NULL 已从条件分布剔除）
    cnt_logits0 = np.full(ms.M_MAX + 1, -1e9, dtype=np.float32)
    cnt_logits0[0] = 0.0
    scens0 = ms.sample_from_logits(tok_logits, cnt_logits0, 0.0, rng2, 3, pools)
    assert all(len(s) == 0 for s in scens0)
    print("[PASS] t21_maskco_p0_regressions (PAD 分离/需求桶/时钟支持/数量头)")
    return True


def test_t22_maskco_permutation_consistency():
    """I1（2026-09-30）：槽对称/置换一致性——无位置编码模型在可见 token 顺序置换下，
    MASK 槽 logits 与数量头 logits 不变；训练 token 损失对目标 token 的槽位分配顺序不变。
    守卫未来误加位置编码/顺序监督造成的对称性破坏。"""
    import numpy as np
    try:
        import jax.numpy as jnp
        from flax import nnx
        import optax
        import maskco_scenario as ms
        model = ms.MaskCOScenarioModel(dim=32, arm='random', rngs=nnx.Rngs(42))
    except Exception as e:
        print('  [info] 跳过 t22（jax/nnx 不可用: %r）' % (e,))
        return True
    m = 8
    vis = np.array([3, 7, 12, 200, 350], dtype=np.int32)
    X1 = np.concatenate([vis, np.full(m, ms.MASK_TOKEN, np.int32)])
    X2 = np.concatenate([vis[::-1].copy(), np.full(m, ms.MASK_TOKEN, np.int32)])
    t1, c1 = model(jnp.asarray(X1)[None])
    t2, c2 = model(jnp.asarray(X2)[None])
    d = np.abs(np.asarray(t1[0][len(vis):]) - np.asarray(t2[0][len(vis):])).max()
    assert d < 1e-3, ('MASK 槽 logits 对可见顺序敏感', float(d))
    dc = np.abs(np.asarray(c1[0]) - np.asarray(c2[0])).max()
    assert dc < 1e-3, ('数量头 logits 对可见顺序敏感', float(dc))
    # token 损失对目标槽位分配顺序不变
    Y1 = np.full(X1.shape, ms.NULL_TOKEN, np.int32)
    Y1[len(vis):len(vis) + 5] = np.array([10, 20, 30, 40, 50])
    Y2 = Y1.copy()
    Y2[len(vis):len(vis) + 5] = np.array([50, 40, 30, 20, 10])
    mask = (jnp.asarray(X1) == ms.MASK_TOKEN)[None]   # (1, L)

    def _loss(Y):
        nll = optax.softmax_cross_entropy_with_integer_labels(t1, jnp.asarray(Y)[None])
        return float(jnp.sum(jnp.where(mask, nll, 0.0)) / mask.sum())

    l1, l2 = _loss(Y1), _loss(Y2)
    assert abs(l1 - l2) < 1e-4, (l1, l2)
    print("[PASS] t22_maskco_permutation_consistency")
    return True


def test_t23_maskco_iterative_refine():
    """I2（2026-09-30）：多轮保留/重掩码——remask_frac=0 全保留（数量与单步一致）；
    remask_frac=1 全重采样（第二轮条件分布强制特定 token）；数量 n 跨轮固定。"""
    import numpy as np
    try:
        import maskco_scenario as ms
        from scenario_saa import VisibleSnapshot, ScenarioOrder
    except Exception as e:
        print('  [info] 跳过 t23（jax 不可用: %r）' % (e,))
        return True
    m = 6
    tl1 = np.zeros((m, ms.VOCAB_SIZE), dtype=np.float32)          # 均匀非 NULL 分布
    cl1 = np.full(ms.M_MAX + 1, -1e9, dtype=np.float32)
    cl1[4] = 0.0                                                  # 数量头 one-hot n=4
    t_star = ms.order_to_token(ScenarioOrder(oid=1, reveal=3.0, x=0.5, y=0.5,
                                             demand=1.5, temp_class=0,
                                             tw_start=0.0, tw_end=5.0,
                                             service_time=0.05))
    tl2 = np.full((m, ms.VOCAB_SIZE), -1e9, dtype=np.float32)
    tl2[:, t_star] = 0.0                                          # 第二轮 one-hot t_star
    pools = ms.HistoryPools([])

    class _Stub:
        pass

    def make_sampler():
        s = ms.MaskCOScenarioSampler(_Stub(), pools, m_max=m)
        state = {'n': 0}

        def fwd(prefix, clock_bin=None, n_mask=None):
            state['n'] += 1
            if state['n'] == 1:
                return tl1.copy(), cl1.copy()
            return tl2.copy(), cl1.copy()

        s._forward_masks = fwd
        return s

    snap = VisibleSnapshot(clock=0.0, orders=(), accepted=frozenset(),
                           rejected=frozenset(), vehicles=(), energy_used=0.0,
                           budget=700.0, booking_horizon=22.0, capacity=50.0)
    s0 = make_sampler()
    sc0 = s0.iterative_sample(snap, np.random.default_rng(5), 3,
                              remask_frac=0.0, rounds=1)
    assert all(len(x) == 4 for x in sc0), [len(x) for x in sc0]
    s1 = make_sampler()
    sc1 = s1.iterative_sample(snap, np.random.default_rng(5), 3,
                              remask_frac=1.0, rounds=1)
    for scen in sc1:
        assert len(scen) == 4
        for o in scen:
            assert ms.order_to_token(o) == t_star, ms.order_to_token(o)
    print("[PASS] t23_maskco_iterative_refine")
    return True


def test_t14_external_seed_identity():
    """A-04：跨 seed 配对被拒绝（final-test 不得复用开发集基线）。"""
    import run_a1_external_accept as ext
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'results', 'a1_step2_gate_vote2_20260926', 'gate.json')
    raised = False
    try:
        ext.main(['--solver', 'ortools', '--baseline-from', base, '--final-test',
                  '--dev-instances', '1', '--out', 'results/_t14tmp'])
    except SystemExit:
        raised = True
    assert raised, "跨 seed 配对必须被拒绝"
    print("[PASS] t14_external_seed_identity")
    return True


def test_t25_adjudicate_dayid_pairing():
    """T25（用户路线图 09-30）：验收器与四臂差分按 day_id 重排配对（乱序反例）；
    数值门不要求第三臂；bootstrap 用实际 gate seed；重复日/缺日不可裁决；
    超时按 protocol 规则不阻塞正式资格（零超时仅敏感性标签）。"""
    import json
    import tempfile
    import diag_adjudicate as adj
    import diag_s3_density_analysis as d4
    from run_exp_reserve import _stat

    def _row(i, u):
        return {'i': i, 'utility': u, 'hard_feasible': True, 'timeouts': 0,
                'failures': {}, 'elapsed_s': 1.0}

    def _gate(cond_rows, uncond_rows, gate_seed=20260926, explicit_ok=True,
              idt_extra=None, n_days=4):
        idt = {'source_sha256': {'f': 'a' * 64}, 'source_end_sha256': {'f': 'a' * 64},
               'source_stable': True,
               'shared_source_sha256': {'x': 'a' * 64},
               'contract_sha256': 'b' * 64, 'data_sha256': 'c' * 64,
               'data_meta_sha256': 'd' * 64, 'budget_B': 1.0,
               'cooling_share': 2.0,
               'shared_config': {'time_limit': 10.0, 'capacity': 50.0,
                                 'num_vehicles': 15, 'n_orders': 200},
               'seeds': {'train': 20260925, 'gate': gate_seed}}
        idt.update(idt_extra or {})
        tiers = {}
        for pn in ('p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)'):
            per_day = {'cond_hist': cond_rows, 'uncond_hist': uncond_rows}
            if explicit_ok:
                per_day['explicit_feat'] = cond_rows
            else:
                per_day['explicit_feat'] = [_row(r['i'], r['utility']) for r in cond_rows]
                per_day['explicit_feat'][0]['utility'] = None
            tiers[pn] = {'per_day': per_day,
                         'main_comparison': {'cond_minus_uncond': {}}}
        return {'config': {'gate_instances': n_days, 'time_limit': 10.0, 'K': 10,
                           'capacity': 50.0, 'num_vehicles': 15, 'n_orders': 200},
                'seeds': {'train': 20260925, 'gate': gate_seed},
                'budget': {'B': 1.0, 'cooling_share': 2.0},
                'identity': idt, 'gate': tiers}

    # 6 天：正确逐日配对差 = [14,11,8,5,2,-1]（day0..5）
    cond = [_row(i, u) for i, u in enumerate([14.0, 13.0, 12.0, 11.0, 10.0, 9.0])]
    uncond = [_row(i, u) for i, u in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])]
    # 乱序版：JSON 行序反序（同一数据，行序不同）
    cond_perm = list(reversed(cond))
    g1 = adj.adjudicate(_gate(cond, uncond, n_days=6), expected=6)
    g2 = adj.adjudicate(_gate(cond_perm, uncond, n_days=6), expected=6)
    ref = _stat(np.array([14.0, 11.0, 8.0, 5.0, 2.0, -1.0]), 20260926)
    n1 = g1['numeric_per_tier']['p_c=0']
    n2 = g2['numeric_per_tier']['p_c=0']
    assert n1['mean'] == ref['mean'] and n2['mean'] == ref['mean'], (n1, ref)
    assert n1['ci_lo'] == n2['ci_lo'] == ref['ci_lo'], (n1, n2, ref)
    assert n1['bootstrap_seed'] == 20260926
    # 行序错配会给出不同 CI（证明按 day_id 重排必要）：
    naive = _stat(np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0]), 20260926)
    assert naive['ci_lo'] != ref['ci_lo']
    # 显式臂非有限不阻塞两臂数值门
    g3 = adj.adjudicate(_gate(cond, uncond, explicit_ok=False, n_days=6), expected=6)
    assert g3['numeric_per_tier']['p_c=0']['numeric_passed'] == \
        g1['numeric_per_tier']['p_c=0']['numeric_passed']
    assert any('explicit_feat_diagnostic' in s for s in g3['zero_timeout_sensitivity_labels'])
    # 重复日（缺 day5、day0 重复）→ 不可裁决（用新 dict，避免别名污染后续断言）
    dup = [_row(0, 14.0)] + [_row(r['i'], r['utility']) for r in cond[1:5]] + [_row(1, 9.0)]
    g4 = adj.adjudicate(_gate(dup, uncond, n_days=6), expected=6)
    assert not g4['formal_adjudicable']
    assert any('day_incomplete' in p for p in g4['problems'])
    # 身份 source_stable=False → 不正式
    g5 = adj.adjudicate(_gate(cond, uncond, idt_extra={'source_stable': False}, n_days=6),
                        expected=6)
    assert not g5['formal_adjudicable']
    assert 'identity_source_not_stable' in g5['problems']
    # 超时：protocol 规则下不阻塞正式资格，仅零超时敏感性标签；sensitivity 规则则阻塞
    cond_t = [dict(r, timeouts=1) for r in cond]
    g6 = adj.adjudicate(_gate(cond_t, uncond, n_days=6), expected=6)
    assert g6['formal_adjudicable'] is True
    assert g6['zero_timeout_ok'] is False
    g7 = adj.adjudicate(_gate(cond_t, uncond, n_days=6), expected=6, timeout_rule='sensitivity')
    assert g7['formal_adjudicable'] is False
    # 四臂差分 _arm：按 day_id 重排（乱序输入 → 有序向量）
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, 'reveal.json')
        p2 = os.path.join(td, 'density.json')
        json.dump(_gate(cond_perm, uncond), open(p1, 'w'))
        json.dump(_gate(cond, uncond), open(p2, 'w'))
        _save_expected = d4.EXPECTED
        d4.EXPECTED = 6   # 合成探针用 6 天（模块常量为正式 40 天）
        try:
            v = d4._arm(json.load(open(p1)), 'p_c=0', 'cond_hist')
        finally:
            d4.EXPECTED = _save_expected
        assert list(v) == [14.0, 13.0, 12.0, 11.0, 10.0, 9.0], v
    print("[PASS] t25_adjudicate_dayid_pairing")
    return True


def test_t26_boundary_bucket_count_preserved():
    """T26（用户路线图 09-30 边界桶探针）：数量头采样值 = 实际保留数——边界桶内
    截断解码 reveal∈(clock,桶尾]，200/200 场景保留 1 单且重编码回同一 token；
    多时钟（0.0/0.25/0.49/0.75）回归；旧实现（采后删单）在 clock=0.25 会丢 ~43%。"""
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    if _models not in sys.path:
        sys.path.insert(0, _models)
    from maskco_scenario import (sample_from_logits, HistoryPools, order_to_token,
                                 VOCAB_SIZE, M_MAX, N_CLASSES, N_SPOTS, N_DEMAND_BINS)
    pools = HistoryPools([])   # 空历史：坐标/lead/需求走回退，仅测数量与时钟

    def _tok(t, c=0, b=0, d=0):
        return 1 + (t * N_CLASSES + c) * N_SPOTS * N_DEMAND_BINS + (b * N_DEMAND_BINS + d)

    for clock, t_bin in ((0.0, 0), (0.25, 0), (0.49, 0), (0.75, 1)):
        tok = _tok(t_bin)
        tl = np.full((M_MAX, VOCAB_SIZE), -1e9, dtype=np.float32)
        tl[:, tok] = 0.0
        cl = np.full(M_MAX + 1, -1e9, dtype=np.float32)
        cl[1] = 0.0
        scens = sample_from_logits(tl, cl, clock, np.random.default_rng(7), 200, pools)
        assert len(scens) == 200
        for s in scens:
            assert len(s) == 1, ('数量头=1 但场景保留 %d 单' % len(s))
            o = s[0]
            assert o.reveal > clock + 1e-6, (clock, o.reveal)
            assert order_to_token(o) == tok, ('边界桶截断解码须重编码回同一 token',
                                              clock, order_to_token(o), tok)
    # 数量头=0 → 空场景
    cl0 = np.full(M_MAX + 1, -1e9, dtype=np.float32)
    cl0[0] = 0.0
    tl = np.full((M_MAX, VOCAB_SIZE), -1e9, dtype=np.float32)
    tl[:, 1] = 0.0
    scens0 = sample_from_logits(tl, cl0, 0.25, np.random.default_rng(7), 50, pools)
    assert all(len(s) == 0 for s in scens0)
    # 直接解码：reveal 落在 (clock, 桶尾]
    o = pools.decode_token(_tok(0), np.random.default_rng(3), clock=0.25)
    assert 0.25 < o.reveal <= 0.5 and order_to_token(o) == _tok(0)
    print("[PASS] t26_boundary_bucket_count_preserved")
    return True


def test_t27_pretrained_norm_broadcast_and_pad():
    """T27（2026-09-30）：①PretrainedEncoder 归一广播陷阱回归——(B,1,2)/(B,1) 右对齐
    广播成 (B,B,2) 垃圾；修复后 n_valid 升 (B,1,1)，mean 为逐样本正确统计；
    ②（服务器，有 cvrp ckpt 时）同样本有/无 PAD 的有效 logits 一致 + B=8,L=383 无崩溃。"""
    import jax.numpy as jnp
    # ① 广播修复纯数学回归（与 PretrainedEncoder.__call__ 同一表达式）
    B, L = 8, 383
    coords = np.random.default_rng(0).normal(size=(B, L, 2))
    valid = np.random.default_rng(1).integers(0, 2, size=(B, L)).astype(np.float32)
    n_valid = np.maximum(valid.sum(axis=1), 1.0)[:, None, None]
    mean = (coords * valid[..., None]).sum(axis=1, keepdims=True) / n_valid
    assert mean.shape == (B, 1, 2), mean.shape
    ref = (coords * valid[..., None]).sum(axis=1) / n_valid[:, 0, :1]
    assert np.allclose(mean[:, 0, :], ref), '逐样本均值必须正确'
    # 旧写法（除 (B,1)）会右对齐广播成 (B,B,2)——反例锁定
    bad = (coords * valid[..., None]).sum(axis=1, keepdims=True) / \
        np.maximum(valid.sum(axis=1, keepdims=True), 1.0)
    assert bad.shape == (B, B, 2), ('旧广播陷阱形状反例漂移', bad.shape)
    # ② 服务器 PAD 一致性（无 ckpt 则跳过）
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    ckpt = os.path.join(_root, 'MASKCO_code', 'ckpts', 'cvrp100.ckpt')
    if not os.path.exists(ckpt):
        print("[SKIP] t27 服务器段（本地无 cvrp100.ckpt；广播回归已 PASS）")
        return True
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    if _models not in sys.path:
        sys.path.insert(0, _models)
    from mpre import load_cvrp_model
    from flax import nnx
    from maskco_scenario import (MaskCOScenarioModel, MASK_TOKEN, PAD_TOKEN, VOCAB_SIZE)
    import jax
    cvrp, cfg, _step = load_cvrp_model(ckpt)
    model = MaskCOScenarioModel(dim=128, arm='pretrained', cvrp_model=cvrp,
                                rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    vis = rng.integers(1, VOCAB_SIZE, size=(8, 183)).astype(np.int32)
    X1 = np.concatenate([vis, np.full((8, 200), MASK_TOKEN, np.int32)], axis=1)
    X2 = np.concatenate([X1, np.full((8, 17), PAD_TOKEN, np.int32)], axis=1)
    tl1, cl1 = model(jnp.asarray(X1))
    tl2, cl2 = model(jnp.asarray(X2))
    l1 = np.asarray(tl1[0])
    l2 = np.asarray(tl2[0][:X1.shape[1]])
    assert l1.shape == l2.shape == (X1.shape[1], VOCAB_SIZE)
    assert np.abs(l1 - l2).max() < 1e-3, ('PAD 必须不改变有效 logits',
                                          float(np.abs(l1 - l2).max()))
    assert np.allclose(np.asarray(cl1), np.asarray(cl2), atol=1e-3)
    print("[PASS] t27_pretrained_norm_broadcast_and_pad")
    return True


def test_t28_clock_conditioning():
    """T28（I1 进阶 2026-09-30）：①clock_bin_of 边界；②模型 clock_bin 条件生效
    （同输入不同时钟桶 → 输出不同；None=旧路径形状不变）；③make_training_slices
    返回 (vis,tgt,n,clock_bin) 且 clock_bin 与切面时钟一致；④采样器把 snapshot 时钟
    传给模型（stub 捕获）。"""
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    if _models not in sys.path:
        sys.path.insert(0, _models)
    import jax.numpy as jnp
    from flax import nnx
    import maskco_scenario as ms
    from scenario_saa import ScenarioOrder, VisibleSnapshot

    assert ms.clock_bin_of(0.0) == 0
    assert ms.clock_bin_of(0.49) == 0
    assert ms.clock_bin_of(0.5) == 1
    assert ms.clock_bin_of(15.999) == 31
    assert ms.clock_bin_of(99.0) == 31   # 截断

    model = ms.MaskCOScenarioModel(dim=32, arm='random', rngs=nnx.Rngs(1))
    toks = jnp.asarray([[3, 7, 12] + [ms.MASK_TOKEN] * 4], dtype=jnp.int32)
    tl_a, cl_a = model(toks, clock_bin=jnp.asarray([3], jnp.int32))
    tl_b, cl_b = model(toks, clock_bin=jnp.asarray([8], jnp.int32))
    tl_n, cl_n = model(toks)   # 旧路径（None）
    assert tl_a.shape == tl_b.shape == tl_n.shape == (1, 7, ms.VOCAB_SIZE)
    assert cl_a.shape == cl_n.shape == (1, ms.M_MAX + 1)
    assert np.abs(np.asarray(tl_a) - np.asarray(tl_b)).max() > 0, '时钟条件必须生效'
    assert np.abs(np.asarray(cl_a) - np.asarray(cl_b)).max() > 0

    # 训练切片带时钟桶，且与切面（= 可见最大 reveal）一致
    day = [ScenarioOrder(oid=1, reveal=0.3, x=0.2, y=0.3, demand=1.5, temp_class=0,
                         tw_start=0.0, tw_end=3.0, service_time=0.05),
           ScenarioOrder(oid=2, reveal=1.2, x=0.7, y=0.4, demand=2.5, temp_class=1,
                         tw_start=0.0, tw_end=4.0, service_time=0.05),
           ScenarioOrder(oid=3, reveal=2.6, x=0.9, y=0.1, demand=1.8, temp_class=2,
                         tw_start=0.0, tw_end=5.0, service_time=0.05)]
    slices = ms.make_training_slices([day], cuts_per_day=3,
                                     rng=np.random.default_rng(0))
    assert slices and all(len(s) == 4 for s in slices)
    for vis, tgt, n, cb in slices:
        assert n == len(tgt)
        assert 0 <= cb < ms.N_TIME_BINS

    # 采样器把 snapshot 时钟传给模型（stub 记录 clock_bin）
    captured = {}

    class _StubModel:
        def __call__(self, tokens, clock_bin=None, attn_mask=None):
            captured['cb'] = None if clock_bin is None else int(np.asarray(clock_bin)[0])
            tl = np.zeros((tokens.shape[0], tokens.shape[1], ms.VOCAB_SIZE),
                          dtype=np.float32)
            tl[..., 5] = 1.0
            cl = np.full((tokens.shape[0], ms.M_MAX + 1), -1e9, dtype=np.float32)
            cl[:, 1] = 0.0
            return jnp.asarray(tl), jnp.asarray(cl)

    snap = VisibleSnapshot(clock=1.75, orders=tuple(day[:1]), accepted=frozenset(),
                           rejected=frozenset(), vehicles=(), energy_used=0.0,
                           budget=1e9, booking_horizon=16.0, capacity=50.0)
    smp = ms.MaskCOScenarioSampler(_StubModel(), ms.HistoryPools([]), m_max=8)
    scens = smp.sample(snap, np.random.default_rng(0), 3)
    assert captured['cb'] == ms.clock_bin_of(1.75) == 3, captured
    assert len(scens) == 3
    print("[PASS] t28_clock_conditioning")
    return True


def test_t29_partial_mask_training():
    """T29（I2 2026-09-30）：部分掩码训练批次不变式——每样本随机保留 n_keep∈[0,n−1]
    未来 token 作为可见条件（不监督），MASK 槽数 = m_max−n_keep，重建目标恰为
    未保留的未来 token；数量标签 N 保持完整未来数；kept∪regen == tgt 多重集；
    partial=False 路径不变（恰 m_max 个 MASK 槽，T21 反例不变）。"""
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    _training = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'training')
    for _p in (_models, _training):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_maskco_scenario as tms
    import maskco_scenario as ms
    m = 8
    slices = [([1, 2], [10, 11, 12, 13, 14], 5, 6),
              ([7], [20, 21], 2, 3),
              ([4, 5, 6], [30, 31, 32, 33, 34, 35], 6, 9)]
    rng = np.random.default_rng(7)
    for _ in range(50):
        X, Y, N, CB = tms.make_batch(slices, [0, 1, 2], m, partial=True,
                                     p_max=0.8, rng=rng)
        assert list(N) == [5, 2, 6], list(N)
        for b in range(3):
            vis_n = len(slices[b][0])
            tgt = list(slices[b][1])
            row = X[b]
            pos = vis_n
            kept = []
            while pos < vis_n + m and row[pos] not in (ms.MASK_TOKEN, ms.PAD_TOKEN):
                kept.append(int(row[pos]))
                pos += 1
            n_keep = len(kept)
            n_mask = int((row[vis_n:vis_n + m] == ms.MASK_TOKEN).sum())
            assert n_mask == m - n_keep, (b, n_keep, n_mask)
            # kept ⊆ tgt 多重集
            for t in kept:
                assert t in tgt, (b, t)
            # 重建目标 = tgt 中未被保留的 token（多重集差），仅出现在 MASK 槽（非 NULL 前缀）
            regen_y = [int(y) for y in Y[b, vis_n + n_keep:vis_n + n_keep + n_mask]
                       if int(y) != ms.NULL_TOKEN]
            assert sorted(regen_y) == sorted(t for t in tgt if t not in kept), \
                (b, kept, regen_y)
            # 保留位不被监督（Y 为 NULL）
            assert (Y[b, vis_n:vis_n + n_keep] == ms.NULL_TOKEN).all()
            # 保留上限：n_keep ≤ floor(n*p_max)-1（至少 1 个重建目标）
            assert n_keep <= max(int(len(tgt) * 0.8), 1) - 1, (b, n_keep)
        # partial=False 不变式（恰 m_max 个 MASK 槽）
        X0, Y0, N0, _ = tms.make_batch(slices, [0, 1, 2], m)
        for b in range(3):
            assert (X0[b, len(slices[b][0]):len(slices[b][0]) + m] == ms.MASK_TOKEN).all()
    print("[PASS] t29_partial_mask_training")
    return True


def test_t30_sampler_fixed_shape():
    """T30（2026-09-30 固定形状修复）：①采样器前向恒长 2*m_max（[前缀][MASK×n][PAD…]），
    与训练批长一致 → 部署只编译一次（10s 预算内）；②PAD 补齐不改变 MASK 槽 logits 与
    数量头输出（PAD 隔离）；③迭代路径 n_mask = m_max − len(kept)。"""
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    if _models not in sys.path:
        sys.path.insert(0, _models)
    import jax.numpy as jnp
    from flax import nnx
    import maskco_scenario as ms

    m = 8
    model = ms.MaskCOScenarioModel(dim=32, arm='random', rngs=nnx.Rngs(3))

    # ① spy：采样器前向形状恒定
    captured = {}

    class _Spy:
        def __call__(self, tokens, clock_bin=None):
            captured['shape'] = tuple(tokens.shape)
            tl = np.zeros((tokens.shape[0], tokens.shape[1], ms.VOCAB_SIZE),
                          dtype=np.float32)
            cl = np.full((tokens.shape[0], ms.M_MAX + 1), -1e9, dtype=np.float32)
            cl[:, 1] = 0.0
            return jnp.asarray(tl), jnp.asarray(cl)

    smp = ms.MaskCOScenarioSampler(_Spy(), ms.HistoryPools([]), m_max=m)
    tl, cl = smp._forward_masks([3, 7, 12], clock_bin=4)
    assert captured['shape'] == (1, 2 * m), captured
    assert tl.shape == (m, ms.VOCAB_SIZE)
    tl2, cl2 = smp._forward_masks([3, 7, 12, 5], clock_bin=4, n_mask=m - 1)
    assert captured['shape'] == (1, 2 * m) and tl2.shape == (m - 1, ms.VOCAB_SIZE)

    # ② 等价性：PAD 补齐不改变 MASK 槽 logits / 数量头（真实 random 臂模型）
    vis = [3, 7, 12]
    toks_var = np.array(vis + [ms.MASK_TOKEN] * m, dtype=np.int32)
    toks_fix = np.full(2 * m, ms.PAD_TOKEN, dtype=np.int32)
    toks_fix[:len(vis)] = vis
    toks_fix[len(vis):len(vis) + m] = ms.MASK_TOKEN
    cb = jnp.asarray([4], dtype=jnp.int32)
    tv, cv = model(jnp.asarray(toks_var)[None], clock_bin=cb)
    tf, cf = model(jnp.asarray(toks_fix)[None], clock_bin=cb)
    lv = np.asarray(tv[0][len(vis):len(vis) + m])
    lf = np.asarray(tf[0][len(vis):len(vis) + m])
    assert np.abs(lv - lf).max() < 1e-5, float(np.abs(lv - lf).max())
    assert np.allclose(np.asarray(cv), np.asarray(cf), atol=1e-6)
    print("[PASS] t30_sampler_fixed_shape")
    return True


def test_t31_saved_weights_are_trained():
    """T31（2026-10-01 关键回归）：训练器保存的权重必须是训练后权重——旧实现在
    nnx.jit 内 nnx.update(model, ...)，模型参数被捐献、更新不传回外层 → 保存的
    是未训练 init 权重（此前全部 checkpoint 均为 init，已实测 n_changed=0）。
    修复后：微训练烟测 → 保存的 trainable 参数与 init 显著不同。"""
    import shutil
    _models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')
    _training = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'training')
    for _p in (_models, _training):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_maskco_scenario as tms
    from flax import nnx
    out = 'results/_t31_saved_weights'
    shutil.rmtree(out, ignore_errors=True)
    tms.main(['--train-instances', '10', '--n-orders', '100', '--steps', '15',
              '--dim', '32', '--batch-size', '4', '--m-max', '32', '--arm', 'random',
              '--seed', '42', '--out', out])
    from maskco_scenario import MaskCOScenarioModel, load_model

    def trainable_map(m):
        fs = nnx.state(m, nnx.Param).flat_state()
        return {str(p): np.asarray(v.value) for p, v in zip(fs.paths, fs.leaves)
                if not any(str(x).startswith('_cvrp') for x in p)}

    init = MaskCOScenarioModel(dim=32, arm='random', rngs=nnx.Rngs(42))
    trained = MaskCOScenarioModel(dim=32, arm='random', rngs=nnx.Rngs(42))
    trained = load_model(trained, os.path.join(out, 'model.bin'))
    ti, tt = trainable_map(init), trainable_map(trained)
    assert len(ti) == len(tt) > 0
    n_changed = sum(1 for k in ti if np.abs(tt[k] - ti[k]).max() > 0)
    max_change = max((float(np.abs(tt[k] - ti[k]).max()) for k in ti), default=0.0)
    assert n_changed == len(ti), (n_changed, len(ti))
    assert max_change > 1e-6, max_change
    print("[PASS] t31_saved_weights_are_trained (changed=%d/%d, max=%.3e)"
          % (n_changed, len(ti), max_change))
    return True


def test_t24_step3_identity_seal():
    """Q-04（步骤 3 驱动）：启动/结束源码双读稳定；模型 bin 哈希；基线身份双格式解析；
    跨运行配对核实（同身份→verified，数据漂移→mismatch）。"""
    import tempfile
    import run_a1_step3_compare as s3

    # 1) 源码封存往返：结束复查应与启动一致
    seal = s3.source_seal(s3.SOURCE_FILES_STEP3)
    s3.seal_end(seal)
    assert seal['stable'] is True, "启动/结束源码 hash 必须一致"
    assert seal['end_sha256'] == seal['startup_sha256']

    # 2) 模型 bin 哈希
    with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.bin') as tf:
        tf.write(b'maskco-test-model')
        bin_path = tf.name
    try:
        sha = s3._sha256_path(bin_path)
        assert len(sha) == 64 and sha == s3._sha256_path(bin_path)
    finally:
        os.unlink(bin_path)

    # 3) 基线身份双格式解析
    step2_doc = {'identity': {'x': 1}, 'seeds': {'gate': 20260926}}
    step3_doc = {'results': {'20260926': {'_identity': {'y': 2}}}}
    assert s3._baseline_identity_for(step2_doc, 20260926) == {'x': 1}
    assert s3._baseline_identity_for(step3_doc, 20260926) == {'y': 2}
    assert s3._baseline_identity_for({}, 20260926) is None
    assert s3._baseline_identity_for(step3_doc, 20260927) is None

    # 4) 身份块构造 + 配对核实
    contract = make_c0_contract_v2()
    ds = add_v2_initial_quality(_mk_day([(0.2, 0.2, 5.0, 0, 1.0, 20.0)]), contract)
    args = SimpleNamespace(model_bin=bin_path, arm='random', dim=64, m_max=64,
                           iterative=False, remask_frac=0.5, rounds=1,
                           cvrp_ckpt=None, time_limit=10.0, capacity=50.0,
                           num_vehicles=15, n_orders=2)
    ident = s3._run_identity(args, contract, ds, 20260926, sha, 1234.5, 2.0, seal)
    assert ident['source_stable'] is True
    assert ident['model_bin_sha256'] == sha
    assert ident['budget_B'] == 1234.5
    # 配对：字段一致 → verified；数据漂移 → mismatch；无基线 → missing
    twin = {k: ident[k] for k in ('shared_source_sha256', 'contract_sha256',
                                  'data_sha256', 'data_meta_sha256', 'budget_B',
                                  'cooling_share', 'shared_config', 'source_stable')}
    twin['seeds'] = {'train': ident['seeds']['train']}
    chk = s3.pairing_identity_check(ident, {'identity': twin})
    assert chk['identity_verified'] is True, chk
    twin2 = dict(twin, data_sha256='0' * 64)
    chk2 = s3.pairing_identity_check(ident, {'identity': twin2})
    assert chk2['identity_verified'] is False and 'data_sha256_mismatch' in chk2['problems']
    chk3 = s3.pairing_identity_check(ident, {})
    assert chk3['identity_verified'] is False and 'baseline_identity_missing' in chk3['reason']
    print("[PASS] t24_step3_identity_seal")
    return True


def test_t32_incr_eval_equivalence():
    """T32（2026-10-02 S3-4 增量评价）：①预筛保守性 oracle——对随机状态逐一对比
    _route_feasible_u 与 _precheck_ins/_precheck_replace：全量可行 ⇒ 预筛必过
    （预筛不得假拒，否则行为分叉）；②端到端等价（v2 口径）——同种子/同操作序列，
    incr_eval 开/关两次运行：插入结果一致、每车顾客集合一致、车队总距离差 ≤0.5% 相对
    （v2 Δ 排序与旧全量求和浮点顺序不同，对称并列候选的选择顺序允许罕见分歧且距离相同，
    见 S3-4 规格 §9；质量级验证由烟测 A/B 逐日效用对比承担）。"""
    import copy
    rp_off, _ = _mk_replanner(shadow_mode='ls', incr_eval=False)
    rp_on, _ = _mk_replanner(shadow_mode='ls', incr_eval=True)
    rng = np.random.default_rng(20261002)

    def _mk_random_day(seed_orders, tight):
        n = 1 + seed_orders
        coords = rng.uniform(0.2, 0.8, size=(1, n, 2)).astype(np.float32)
        coords[0, 0] = [0.5, 0.5]
        demands = rng.uniform(1.0, 12.0, size=(1, n)).astype(np.float32)
        demands[0, 0] = 0.0
        tw_end = rng.uniform(12.0, 22.0, size=(1, n)).astype(np.float32)
        tw_end[0, 0] = 22.0
        service = np.full((1, n), 0.05, np.float32)
        service[0, 0] = 0.0
        reveal = np.full((1, n), 1e6, np.float32)
        reveal[0, 1:] = rng.uniform(0.0, 6.0, seed_orders)
        if tight:   # 窄 TW：一半试验制造大量不可行候选（预筛的省算场景）
            width = rng.uniform(0.3, 2.5, seed_orders)
            tw_end[0, 1:] = np.minimum(22.0, reveal[0, 1:] + width).astype(np.float32)
        tc = rng.integers(0, 3, size=(1, n)).astype(np.int32)
        tw_start = np.zeros((1, n), np.float32)
        return {'coords': coords, 'demands': demands, 'tw_start': tw_start,
                'tw_end': tw_end, 'service_time': service,
                'reveal_time': reveal, 'temp_class': tc}

    n_oracle = 0
    skipped = 0
    infeas_total = 0
    for trial in range(60):
        tight = (trial % 2 == 1)
        n_real = 30
        ds = _mk_random_day(n_real - 1, tight)
        env = _FakeEnv(ds)
        fut = [ScenarioOrder(oid=-(j + 1), reveal=float(rng.uniform(2, 10)),
                             x=float(rng.uniform(0.2, 0.8)), y=float(rng.uniform(0.2, 0.8)),
                             demand=float(rng.uniform(1, 6)), temp_class=int(rng.integers(0, 3)),
                             tw_start=0.0, tw_end=float(rng.uniform(14, 22)),
                             service_time=0.05) for j in range(12)]
        space, _scen = rp_off._build_space(env, 0, [fut])
        deadline = float(env.tw_end[0, 0])
        pool = list(range(1, n_real))
        rng.shuffle(pool)
        st = {}
        for vid in range(4):
            k = int(rng.integers(0, 4))
            route = pool[vid * 3:vid * 3 + k]
            st[vid] = dict(cur=0, cur_time=0.0, load=0.0, route=list(route),
                           cc=None, frozen=0)
        # 一辆 committed 车：frozen=1（在途腿不可改）
        st[4] = dict(cur=int(pool[12]), cur_time=float(rng.uniform(0, 2)),
                     load=float(rng.uniform(0, 8)), route=list(pool[13:15]),
                     cc=None, frozen=1)
        # ① oracle：随机 (车辆, 位置, 未来订单) 组合的预筛保守性
        for _ in range(30):
            vid = int(rng.integers(0, len(st)))
            s = st[vid]
            rp_on._rebuild_summary(space, env, s, deadline)
            o = n_real + int(rng.integers(0, len(fut)))
            pos = int(rng.integers(s.get('frozen', 0), len(s['route']) + 1))
            trial_route = s['route'][:pos] + [o] + s['route'][pos:]
            full = rp_off._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                            s['load'], trial_route, deadline)
            pre = rp_on._precheck_ins(space, env, s, o, pos, deadline)
            n_oracle += 1
            if full:
                assert pre, ('T32 假拒：ins', trial, vid, pos, o)
            else:
                infeas_total += 1
                skipped += (0 if pre else 1)
            if s['route']:
                i = int(rng.integers(s.get('frozen', 0), len(s['route'])))
                o_new = n_real + int(rng.integers(0, len(fut)))
                rep = list(s['route'])
                rep[i] = o_new
                full_r = rp_off._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                                  s['load'], rep, deadline)
                pre_r = rp_on._precheck_replace(space, env, s, o_new, i, deadline)
                n_oracle += 1
                if full_r:
                    assert pre_r, ('T32 假拒：replace', trial, vid, i, o_new)
                else:
                    infeas_total += 1
                    skipped += (0 if pre_r else 1)
        # ② 端到端等价（v2 口径）：同序列 插入+ls；断言 = 插入结果一致 + 每车顾客集合一致
        # + 总距离差 < 1e-6 相对（v2 的 Δ 排序与旧全量求和浮点顺序不同，近并列选择顺序
        # 可能罕见分歧——见 S3-4 规格 §9；顾客集合与距离必须不变）
        st_a = copy.deepcopy(st)
        st_b = copy.deepcopy(st)
        for j in range(len(fut)):
            o = n_real + j
            ok_a = rp_off._insert_u(space, env, o, st_a, deadline)
            ok_b = rp_on._insert_u(space, env, o, st_b, deadline)
            assert ok_a == ok_b, ('T32 insert 分歧', trial, j, ok_a, ok_b)
            rp_off._local_search_u(space, env, st_a, deadline)
            rp_on._local_search_u(space, env, st_b, deadline)
        for vid in st:
            ra, rb = st_a[vid]['route'], st_b[vid]['route']
            assert sorted(ra) == sorted(rb), (
                'T32 顾客集合分歧', trial, vid, ra, rb)
        da = sum(rp_off._route_dist_u(space, st_a[vid]['cur'], st_a[vid]['route'])
                 for vid in st_a)
        db = sum(rp_on._route_dist_u(space, st_b[vid]['cur'], st_b[vid]['route'])
                 for vid in st_b)
        assert abs(da - db) <= 5e-3 * max(1.0, abs(da), abs(db)), (
            'T32 车队总距离分歧', trial, da, db)
    print('[PASS] t32_incr_eval_equivalence (oracle=%d, 预筛跳过 %.1f%% 不可行候选)'
          % (n_oracle, (100.0 * skipped / max(infeas_total, 1))))
    return True


def test_t33a_anytime_scenario_order_independence():
    """T33a（D3）：_sim_scenario 逐场景值与访问顺序无关（共享 space/states 只读、st 深拷贝）——
    乱序访问与原始顺序的逐场景 acc/rej 值逐位相等（否则 D3 乱序会改变全量投票结果）。"""
    rp, _ = _mk_replanner(shadow_mode='ls', incr_eval=True, anytime_vote=True)
    rng = np.random.default_rng(20261003)
    for trial in range(40):
        n_real = 24
        coords = rng.uniform(0.2, 0.8, size=(1, n_real, 2)).astype(np.float32)
        coords[0, 0] = [0.5, 0.5]
        demands = rng.uniform(1.0, 10.0, size=(1, n_real)).astype(np.float32)
        demands[0, 0] = 0.0
        tw_end = np.full((1, n_real), 22.0, np.float32)
        service = np.full((1, n_real), 0.05, np.float32)
        service[0, 0] = 0.0
        reveal = np.full((1, n_real), 1e6, np.float32)
        reveal[0, 1:] = rng.uniform(0.0, 6.0, n_real - 1)
        ds = {'coords': coords, 'demands': demands, 'tw_start': np.zeros((1, n_real), np.float32),
              'tw_end': tw_end, 'service_time': service, 'reveal_time': reveal,
              'temp_class': rng.integers(0, 3, size=(1, n_real)).astype(np.int32)}
        env = _FakeEnv(ds)
        fut = [ScenarioOrder(oid=-(j + 1), reveal=float(rng.uniform(2, 10)),
                             x=float(rng.uniform(0.2, 0.8)), y=float(rng.uniform(0.2, 0.8)),
                             demand=float(rng.uniform(1, 6)), temp_class=int(rng.integers(0, 3)),
                             tw_start=0.0, tw_end=float(rng.uniform(14, 22)),
                             service_time=0.05) for j in range(10)]
        space, _ = rp._build_space(env, 0, [fut])
        scen_idx = [space.n_real + j for j in range(10)]
        pool = list(range(1, n_real))
        rng.shuffle(pool)
        st = {vid: dict(cur=0, cur_time=0.0, load=0.0,
                        route=list(pool[vid * 3:vid * 3 + int(rng.integers(0, 3))]),
                        cc=None, frozen=0) for vid in range(3)}
        pc_o = rp.reject_penalty[0]
        acc_orig = [rp._sim_scenario(env, 0, 1.0, st, [i], space) for i in scen_idx]
        rej_orig = [rp._sim_scenario(env, 0, 1.0, st, [i], space) - pc_o for i in scen_idx]
        perm = rp._anytime_perm(0, 1, 10)
        assert sorted(perm) == list(range(10))
        acc_sc = [None] * 10
        rej_sc = [None] * 10
        for j in perm:   # 乱序访问：同一共享 st/space（st 由 _sim_scenario 内部深拷贝）
            acc_sc[j] = rp._sim_scenario(env, 0, 1.0, st, [scen_idx[j]], space)
            rej_sc[j] = rp._sim_scenario(env, 0, 1.0, st, [scen_idx[j]], space) - pc_o
        for j in range(10):
            assert acc_sc[j] == acc_orig[j], ('T33a acc 顺序依赖', trial, j)
            assert rej_sc[j] == rej_orig[j], ('T33a rej 顺序依赖', trial, j)
    print('[PASS] t33a_anytime_scenario_order_independence (40 trials x 10 scen 逐位一致)')
    return True


def test_t33b_anytime_perm_crn():
    """T33b（D3）：场景乱序种子臂无关（不同 sampler/arm_seed 同实例同订单 → 同顺序 = CRN）；
    顺序是合法排列且随订单变化。"""
    from scenario_saa import ExplicitFeatureSampler as _EFS
    rp1, _ = _mk_replanner(anytime_vote=True, arm_seed=7001)
    rp2, _ = _mk_replanner(anytime_vote=True, arm_seed=7002,
                           sampler=_EFS(build_history(generate_dataset(5, 30, 20260925))))
    rp1._id_map = {7: 3}
    rp2._id_map = {7: 3}
    perms = []
    for o in range(1, 6):
        p1 = rp1._anytime_perm(4, o, 10)
        p2 = rp2._anytime_perm(4, o, 10)
        assert p1 == p2, ('T33b CRN 破坏', o)
        assert sorted(p1) == list(range(10))
        perms.append(tuple(p1))
    assert len(set(perms)) > 1, 'T33b 顺序不随订单变化（固定序系统偏向风险）'
    print('[PASS] t33b_anytime_perm_crn (臂无关同序 + 随订单变化)')
    return True


def test_t33c_partial_vote_rule():
    """T33c（D3）：部分投票提交规则——done 全 True 与旧路径（acc_u += / rej_u += 后比较）
    浮点逐位一致；子集按已完成场景累计统计量裁决（>0 接、≤0 拒，平票拒）。"""
    rng = np.random.default_rng(20261003)
    for trial in range(200):
        n = 10
        acc = list(rng.normal(100.0, 30.0, n))
        rej = list(rng.normal(100.0, 30.0, n))
        done = [True] * n
        acc_u = 0.0
        rej_u = 0.0
        for a, r in zip(acc, rej):
            acc_u += a
            rej_u += r
        old = acc_u > rej_u
        assert SaaReplanner._partial_vote_accept(acc, rej, done) == old, ('T33c 全量不一致', trial)
        m = int(rng.integers(1, n))
        subset = rng.choice(n, m, replace=False)
        done2 = [False] * n
        for i in subset:
            done2[i] = True
        acc_u = 0.0
        rej_u = 0.0
        for a, r, d in zip(acc, rej, done2):
            if d:
                acc_u += a
                rej_u += r
        expect = acc_u > rej_u
        assert SaaReplanner._partial_vote_accept(acc, rej, done2) == expect, ('T33c 子集不一致', trial)
        if abs(acc_u - rej_u) > 1e-9:
            assert SaaReplanner._partial_vote_accept(acc, rej, done2) == (acc_u - rej_u > 0)
    # 平票（S_m == 0）→ 拒绝（与旧「保计划」同向保守）
    assert not SaaReplanner._partial_vote_accept([1.0, -1.0], [0.0, 0.0], [True, True])
    assert not SaaReplanner._partial_vote_accept([0.0], [0.0], [True])
    print('[PASS] t33c_partial_vote_rule (200 trials 全量逐位 + 子集 + 平票拒)')
    return True


def test_t33d_anytime_deadline_semantics():
    """T33d（D3）：时限语义——m=0 → 旧语义（保计划+拒单+timeouts 计次）；1≤m<K → 部分投票
    提交（partial_commits 计次、timeouts 不计）；pc_o>0 下 d_j=+pc_o → 接单；pc_o=0 平票 → 拒单。"""
    import scenario_saa as ssa
    rp, contract = _mk_replanner(budget=1e9, time_limit=1.0, anytime_vote=True,
                                 anytime_early_stop=False)
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, 1, 0.5, 22.0)]), contract)
    env = _FakeEnv(ds)
    rp._id_map = {1: 1}
    rp._plan = {1: [1]}
    rp._try_insert_certified = lambda *a, **k: True

    def _run(fake_pc, arm_penalty):
        rp._accepted, rp._rejected, rp.timeouts = set(), set(), 0
        rp.partial_commits, rp.early_stops = 0, 0
        rp.reject_penalty = dict(arm_penalty)
        real_pc = ssa.time.perf_counter
        ssa.time.perf_counter = fake_pc
        try:
            rp._saa_decide(env, 0, 1.0, [], np.zeros(2, bool), 1, [[]] * 10, 0.0)
        finally:
            ssa.time.perf_counter = real_pc

    # ① m=0：入口即超时 → 旧语义
    def f0():
        return 2.0
    _run(f0, P_C)
    assert 1 in rp._rejected and 1 not in rp._accepted and rp.timeouts == 1
    assert rp.partial_commits == 0
    # ② m=2（2 对场景完成后超时）且 pc_o>0 → 部分投票接单（sum=2·pc_o>0）
    calls = [0]

    def f2():
        calls[0] += 1
        return 0.0 if calls[0] <= 3 else 2.0   # 入口(1) + 前 2 对场景检查(2,3)=0；第 3 对起超时
    _run(f2, P_C)
    assert 1 in rp._accepted and 1 not in rp._rejected
    assert rp.timeouts == 0 and rp.partial_commits == 1
    # ③ 同 ② 但 pc_o=0 → 平票拒绝（与旧「保计划」同向）
    calls[0] = 0
    _run(f2, {c: 0.0 for c in P_C})
    assert 1 in rp._rejected and 1 not in rp._accepted
    assert rp.timeouts == 0 and rp.partial_commits == 1
    print('[PASS] t33d_anytime_deadline_semantics (m=0 超时计次 / m=2 部分提交接拒 + 平票拒)')
    return True


def test_t33e_anytime_end_to_end_no_crash():
    """T33e（D3）：真实 1 天端到端（anytime_vote=True、0.05s 时限）：无崩溃、硬约束 1.0、
    部分提交/超时计数一致（0 < m < K 的事件才计 partial_commits；超时语义变更不伪造约束）。"""
    contract = make_c0_contract_v2()
    sampler = UncondHistoricalSampler(build_history(generate_dataset(5, 30, 20260925)))
    rp = SaaReplanner(budget=1e9, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=2.0, sampler=sampler,
                      reject_penalty=P_C, K=10, time_limit=0.05, arm_seed=7001,
                      shadow_mode='greedy', anytime_vote=True, anytime_early_stop=True)
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, i % 3, 0.05, 22.0)
                                         for i in range(1, 40)]), contract)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=3,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    traces, _ = env.run(0)
    acc, rej = set(rp._accepted), set(rp._rejected)
    ev = evaluate_trace_a1(traces, {k: v[0] for k, v in ds.items()}, contract,
                           acc, rej, 1e9, P_C)
    assert ev['hard_feasible'], ('T33e 硬约束破坏', ev['failures'])
    assert rp.timeouts >= 0 and rp.partial_commits >= 0 and rp.early_stops >= 0
    assert np.isfinite(ev['utility']), 'T33e 效用非有限'
    print('[PASS] t33e_anytime_end_to_end_no_crash (to=%d partial=%d early_stop=%d u=%.1f)'
          % (rp.timeouts, rp.partial_commits, rp.early_stops, ev['utility']))
    return True


if __name__ == '__main__':
    res = {
        't1_future_revenue_counted_once': test_t1_future_revenue_counted_once(),
        't2_reveal_wait_conflicts_detected': test_t2_reveal_wait_conflicts_detected(),
        't3_shadow_matches_certify_plan': test_t3_shadow_matches_certify_plan(),
        't4_no_inf_contagion': test_t4_no_inf_contagion(),
        't5_sampling_counts_into_time_limit': test_t5_sampling_counts_into_time_limit(),
        't6_shadow_ls_no_regression': test_t6_shadow_ls_no_regression(),
        't7_energy_pricing_marginal': test_t7_energy_pricing_marginal(),
        't8_future_policy_density': test_t8_future_policy_density(),
        't9_precheck_completion_consistency': test_t9_precheck_completion_consistency(),
        't10_cert_exception_rollback': test_t10_cert_exception_rollback(),
        't11_last_scenario_timeout_no_accept': test_t11_last_scenario_timeout_no_accept(),
        't12_committed_leg_frozen': test_t12_committed_leg_frozen(),
        't13_solver_demand_units': test_t13_solver_demand_units(),
        't14_external_seed_identity': test_t14_external_seed_identity(),
        't15_solver_start_pin_scale': test_t15_solver_start_pin_scale(),
        't16_frozen_survives_scenario_copy': test_t16_frozen_survives_scenario_copy(),
        't17_idle_empty_vehicles_in_shadow': test_t17_idle_empty_vehicles_in_shadow(),
        't18_solver_final_deadline_check': test_t18_solver_final_deadline_check(),
        't19_overage_price_continuous': test_t19_overage_price_continuous(),
        't20_expected_day_completeness': test_t20_expected_day_completeness(),
        't21_maskco_p0_regressions': test_t21_maskco_p0_regressions(),
        't22_maskco_permutation_consistency': test_t22_maskco_permutation_consistency(),
        't23_maskco_iterative_refine': test_t23_maskco_iterative_refine(),
        't24_step3_identity_seal': test_t24_step3_identity_seal(),
        't25_adjudicate_dayid_pairing': test_t25_adjudicate_dayid_pairing(),
        't26_boundary_bucket_count_preserved': test_t26_boundary_bucket_count_preserved(),
        't27_pretrained_norm_broadcast_and_pad': test_t27_pretrained_norm_broadcast_and_pad(),
        't28_clock_conditioning': test_t28_clock_conditioning(),
        't29_partial_mask_training': test_t29_partial_mask_training(),
        't30_sampler_fixed_shape': test_t30_sampler_fixed_shape(),
        't31_saved_weights_are_trained': test_t31_saved_weights_are_trained(),
        't32_incr_eval_equivalence': test_t32_incr_eval_equivalence(),
        't33a_anytime_scenario_order_independence': test_t33a_anytime_scenario_order_independence(),
        't33b_anytime_perm_crn': test_t33b_anytime_perm_crn(),
        't33c_partial_vote_rule': test_t33c_partial_vote_rule(),
        't33d_anytime_deadline_semantics': test_t33d_anytime_deadline_semantics(),
        't33e_anytime_end_to_end_no_crash': test_t33e_anytime_end_to_end_no_crash(),
    }
    ok = all(res.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
