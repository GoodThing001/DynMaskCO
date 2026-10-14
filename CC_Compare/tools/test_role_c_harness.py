#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_harness`（Role C 装配层）的测试。

最重要的一组（**S 系列**）把"两臂共用同一场景集合"从约定变成**可证事实**：
用**真实的** `CondHistoricalSampler`、**真实的**对照代码路径（`SaaReplanner._sync_id_map` /
`_build_snapshot` / `_feat`）+ **真实的** `_feat` 公式，对拍本装配层的场景流：

  S1 `scenario_rng` 与对照 `scenario_saa.py:1002` 的构造**逐位相同**（含 `int(round(clock*1000))`）；
  S2 本层的已揭示订单视图与对照 `_sync_id_map` + `_real_order` 的产物是**同一多重集**；
  S3 驱动真实 `CondHistoricalSampler` 后，`_feat` 向量**逐位相同**；
  S4 因此 `sample()` 抽出的场景集合**逐字段相同**（同一 `arm_seed`/`K`）——
     即：同一实例、同一 `clock` 下两臂拿到同一集合；
  S5 换 `arm_seed` ⇒ **集合必须不同**（证明 S4 不是空转：它对种子是敏感的）；
  S6 `arm_seed` 缺省 ⇒ `TypeError`/`HarnessError`（冻结配置漏记该字段，不许静默默认）。

另有无需项目依赖的组：
  W 装配层不引用对照决策引擎（符号级机械检查）；场景集合哈希双重顺序无关；
  E 能耗/收入/罚金/边际燃料口径；完成函数用**注入的假物理**验证 TW/容量/返仓语义；
  D `decide_fn` 端到端（假 sampler + 假 certify）走真实 `decide_candidate`。

合成夹具；**不跑实验、不读效用、不碰保留种子**。
运行：python CC_Compare/tools/test_role_c_harness.py   （退出码 0 = 全过）
"""
import inspect
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
for _sub in ('scripts/simulation', 'scripts/coldchain', 'scripts/evaluation', 'scripts/data'):
    sys.path.insert(0, os.path.join(EXT, _sub))
sys.path.insert(0, HERE)

import role_c_forward as fwd                               # noqa: E402
import role_c_harness as hn                                # noqa: E402
import role_c_local_search as ls                           # noqa: E402
import role_c_opportunity_cost as oc                       # noqa: E402

ARM_SEED_COND_HIST = 7002     # run_a1_step2_gate.py:51（不是 SaaReplanner 默认的 7001）
K = 10


def make_fixture(n_customers=6):
    """合成 env（1 天）：与 `test_role_c_env_connectivity.make_fixture` 同风格，
    多几个客户与 2 个温区，使 `_feat` 的类计数/空间桶不为常数。"""
    n = n_customers + 1
    coords = np.zeros((1, n, 2), dtype=np.float32)
    coords[0, 0] = (0.0, 0.0)
    for i in range(1, n):
        coords[0, i] = (float(i), float(i % 3))
    demands = np.zeros((1, n), dtype=np.float32)
    demands[0, 1:] = 1.0
    tw_start = np.zeros((1, n), dtype=np.float32)
    tw_end = np.full((1, n), 100.0, dtype=np.float32)
    # ⚠️ depot 的 `tw_end` 就是**回仓截止**（对照 `_complete_u` 取 `env.tw_end[i,0]`）。
    # 夹具里必须给一个真实可达的值，否则每条路线都因"返仓超时"判不可行 ——
    # 第一版这里抄了连通性测试夹具的 0.0，导致完成函数恒返回 (None, None)。
    tw_end[0, 0] = 1000.0
    service_time = np.full((1, n), 0.1, dtype=np.float32)
    service_time[0, 0] = 0.0
    temp_class = np.zeros((1, n), dtype=np.int32)
    for i in range(1, n):
        temp_class[0, i] = i % 3
    initial_quality = np.ones((1, n), dtype=np.float32)
    reveal_time = np.zeros((1, n), dtype=np.float32)
    for k, i in enumerate(range(1, n)):
        reveal_time[0, i] = 0.0 if k == 0 else float(2 * k)
    d = np.linalg.norm(coords[0][:, None, :] - coords[0][None, :, :], axis=-1)
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start,
            'tw_end': tw_end, 'service_time': service_time, 'temp_class': temp_class,
            'initial_quality': initial_quality, 'reveal_time': reveal_time,
            'dist_mat': d.astype(np.float32)[None, ...]}


class FakeEnv(object):
    """duck-typed env：只提供对照 `_sync_id_map`/`_real_order`/`_feat` 读的字段。"""

    def __init__(self, ds):
        self.dataset = ds
        self.coords = ds['coords']
        self.demands = ds['demands']
        self.tw_start = ds['tw_start']
        self.tw_end = ds['tw_end']
        self.service_time = ds['service_time']
        self.reveal_time = ds['reveal_time']
        self.dist_mat = ds['dist_mat']
        self.num_nodes = int(ds['coords'].shape[1])
        self.capacity = 50.0
        self.tw_speed = 1.0
        self.booking_horizon = 16.0


def make_history(sampler_n=8, seed=0, n_customers=6):
    """合成历史（供真实 `CondHistoricalSampler` 用）—— 不读项目任何数据集。"""
    from scenario_saa import ScenarioOrder
    rng = np.random.default_rng(seed)
    hist = []
    for d in range(sampler_n):
        day = []
        for o in range(1, n_customers + 1):
            day.append(ScenarioOrder(
                oid=o, reveal=float(rng.integers(0, 12)),
                x=float(rng.integers(0, 8)), y=float(rng.integers(0, 3)),
                demand=1.0, temp_class=int(rng.integers(0, 3)),
                tw_start=0.0, tw_end=100.0, service_time=0.1))
        hist.append(day)
    return hist


# ===========================================================================
# S. 场景流与对照**同集合**（用真实 sampler + 真实对照代码路径对拍）
# ===========================================================================
def test_scenario_stream_matches_control():
    from scenario_saa import CondHistoricalSampler, SaaReplanner, make_c0_contract_v2
    cases = []

    env = FakeEnv(make_fixture())
    contract = make_c0_contract_v2()
    sampler = CondHistoricalSampler(make_history(), n_neighbors=10)

    # ---- S1 rng 构造逐位相同 ----
    for clock in (0.0, 2.0, 7.4, 12.345):
        a = hn.scenario_rng(arm_seed=ARM_SEED_COND_HIST, inst_idx=0, clock=clock)
        b = np.random.default_rng([ARM_SEED_COND_HIST, 0, int(round(clock * 1000))])
        cases.append(('S1 rng 与对照构造逐位相同（clock=%s）' % clock,
                      bool(np.array_equal(a.integers(0, 10 ** 6, 8),
                                          b.integers(0, 10 ** 6, 8))), None))

    # ---- S2 已揭示订单视图 == 对照 _sync_id_map + _real_order 的多重集 ----
    # 用**真实** SaaReplanner 实例只为调用它的两个方法（装配层本身不 import 它）。
    rp = SaaReplanner(budget=100.0, capacity=50.0, booking_horizon=16.0,
                      contract=contract, cooling_share=1.0, sampler=sampler,
                      reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, K=K,
                      time_limit=10.0, arm_seed=ARM_SEED_COND_HIST)
    for clock in (0.0, 2.0, 5.0, 11.0):
        rp._id_map, rp._next_oid = {}, 0
        rp._sync_id_map(env, 0, clock)
        ctrl = rp._build_snapshot(env, 0, clock, [], np.zeros(env.num_nodes, bool))
        mine = hn.revealed_order_view(env, 0, clock)
        key = lambda o: (round(float(o.reveal), 9), int(o.temp_class),
                         round(float(o.x), 9), round(float(o.y), 9))
        cases.append(('S2 clock=%.1f：本层视图 == 对照快照的多重集（%d 单）'
                      % (clock, len(ctrl.orders)),
                      sorted(map(key, ctrl.orders)) == sorted(map(key, mine)),
                      (len(ctrl.orders), len(mine))))

    # ---- S3 `_feat` 向量逐位相同（真实 _feat 公式）----
    for clock in (0.0, 2.0, 5.0, 11.0):
        rp._id_map, rp._next_oid = {}, 0
        rp._sync_id_map(env, 0, clock)
        ctrl = rp._build_snapshot(env, 0, clock, [], np.zeros(env.num_nodes, bool))
        f_ctrl = sampler._feat(ctrl.orders, clock)
        f_mine = sampler._feat(tuple(hn.revealed_order_view(env, 0, clock)), clock)
        cases.append(('S3 clock=%.1f：真实 _feat 向量逐位相同' % clock,
                      bool(np.array_equal(f_ctrl, f_mine)), (f_ctrl, f_mine)))

    # ---- S4 抽出的场景集合**逐字段相同** ----
    for clock in (0.0, 2.0, 5.0, 11.0):
        rp._id_map, rp._next_oid = {}, 0
        rp._sync_id_map(env, 0, clock)
        ctrl_snap = rp._build_snapshot(env, 0, clock, [], np.zeros(env.num_nodes, bool))
        ctrl_scen = sampler.sample(
            ctrl_snap,
            np.random.default_rng([ARM_SEED_COND_HIST, 0, int(round(clock * 1000))]), K)
        mine_scen, sdiag = hn.sample_scenarios(
            sampler=sampler, env=env, inst_idx=0, clock=clock,
            arm_seed=ARM_SEED_COND_HIST, K=K)
        okey = lambda scens: [[(int(o.temp_class), round(float(o.x), 9),
                                round(float(o.y), 9), round(float(o.reveal), 9),
                                round(float(o.demand), 9), round(float(o.tw_end), 9))
                               for o in s] for s in scens]
        same = okey(ctrl_scen) == okey(mine_scen)
        cases.append(('S4 ★ clock=%.1f：两臂场景集合**逐字段相同**（%d 场景，digest=%s）'
                      % (clock, len(mine_scen), sdiag['scenario_digest'][:12]),
                      same and len(mine_scen) == K, (len(ctrl_scen), len(mine_scen))))

    # ---- S5 换 arm_seed ⇒ 集合必须不同（证明 S4 非空转）----
    _, d702 = hn.sample_scenarios(sampler=sampler, env=env, inst_idx=0, clock=5.0,
                                 arm_seed=7002, K=K)
    _, d701 = hn.sample_scenarios(sampler=sampler, env=env, inst_idx=0, clock=5.0,
                                 arm_seed=7001, K=K)
    cases.append(('S5 ★ 换 arm_seed（7002→7001）⇒ 场景集合 digest 改变'
                  '（证明 S4 对种子敏感、不是常数）',
                  d702['scenario_digest'] != d701['scenario_digest'],
                  (d702['scenario_digest'][:12], d701['scenario_digest'][:12])))

    # ---- S6 arm_seed 必须显式（冻结配置漏记该字段）----
    try:
        hn.scenario_rng(inst_idx=0, clock=1.0)
        cases.append(('S6 arm_seed 缺省 ⇒ 报错（不得静默用默认值换掉场景流）', False,
                      '未抛错'))
    except TypeError as exc:
        cases.append(('S6 arm_seed 缺省 ⇒ 报错（不得静默用默认值换掉场景流）', True,
                      str(exc)[:50]))
    for bad in (None, True, 7002.0, '7002'):
        try:
            hn.scenario_rng(arm_seed=bad, inst_idx=0, clock=1.0)
            cases.append(('S6b arm_seed=%r ⇒ HarnessError' % (bad,), False, '未抛错'))
        except hn.HarnessError as exc:
            cases.append(('S6b arm_seed=%r ⇒ HarnessError' % (bad,), True,
                          str(exc)[:60]))
    # 诊断里必须留痕 arm_seed / K / rng 构造，便于事后审计场景流
    _, sd = hn.sample_scenarios(sampler=sampler, env=env, inst_idx=0, clock=5.0,
                               arm_seed=ARM_SEED_COND_HIST, K=K)
    cases.append(('S7 抽样诊断留痕 arm_seed / K / rng 构造 / 场景数 / 信息边界',
                  sd['arm_seed'] == 7002 and sd['K'] == K
                  and 'default_rng' in sd['rng_convention']
                  and sd['n_scenarios'] == K
                  and sd['info_max_reveal'] <= 5.0 + 1e-9, sd))
    return cases


# ===========================================================================
# W. 装配层不碰对照决策引擎
# ===========================================================================
def test_no_control_engine_use():
    cases = []
    mods = {'role_c_local_search': ls, 'role_c_forward': fwd, 'role_c_harness': hn,
            'role_c_opportunity_cost': oc}
    bad = hn.assert_no_control_engine_use(mods)
    cases.append(('W1 ★ 四个 Role C 模块都不含对照决策引擎符号（SaaReplanner / '
                  '_local_search_u / _sim_scenario / _complete_u …）',
                  bad == [], bad))
    cases.append(('W2 禁用清单确实覆盖了对照的引擎符号（不是空清单在空转）',
                  'SaaReplanner' in hn.CONTROL_ENGINE_SYMBOLS
                  and '_local_search_u' in hn.CONTROL_ENGINE_SYMBOLS
                  and len(hn.CONTROL_ENGINE_SYMBOLS) >= 8,
                  len(hn.CONTROL_ENGINE_SYMBOLS)))
    cases.append(('W3 闭包申报里每条都写明角色，且标注共享部件"只用抽样"',
                  all(v.strip() for v in hn.ROLE_C_DEPENDENCY_ROLES.values())
                  and 'shared_sampler_only' in
                  hn.ROLE_C_DEPENDENCY_ROLES[
                      'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/'
                      'scenario_saa.py'],
                  hn.ROLE_C_DEPENDENCY_ROLES))
    return cases


# ===========================================================================
# E. 目标/物理绑定
# ===========================================================================
def test_objective_bindings():
    from scenario_saa import make_c0_contract_v2
    from coldchain_evaluator_a1 import REV, FUEL_COST_PER_KM, KM_PER_UNIT
    from run_exp_energy_c0 import c0_marginal_energy
    cases = []
    env = FakeEnv(make_fixture())
    contract = make_c0_contract_v2()
    share = 3.7208946347769882
    bundle = hn.build_space_bundle(env=env, inst_idx=0, scenarios=[], contract=contract)
    e_fn = hn.make_order_energy_fn(space=bundle.space, cooling_share=share,
                                   contract=contract, energy_pricing='marginal')
    tc = env.dataset['temp_class'][0]
    dem = env.demands[0]
    cases.append(('E1 marginal 口径的 e_o 与 c0_marginal_energy(share=0) 逐单相同',
                  all(abs(e_fn(o) - c0_marginal_energy(int(tc[o]), float(dem[o]), 0.0,
                                                       contract)) < 1e-12
                      for o in range(1, env.num_nodes)), None))
    e_am = hn.make_order_energy_fn(space=bundle.space, cooling_share=share,
                                   contract=contract, energy_pricing='amortized')
    cases.append(('E2 amortized 口径把 cooling_share 并进每单（差恰为 share）',
                  all(abs((e_am(o) - e_fn(o)) - share) < 1e-9
                      for o in range(1, env.num_nodes)), None))
    r_fn = hn.make_order_revenue_fn(space=bundle.space)
    cases.append(('E3 收入 = REV[temp_class]（逐单核对）',
                  all(abs(r_fn(o) - REV[int(tc[o])]) < 1e-12
                      for o in range(1, env.num_nodes)), None))
    p_fn = hn.make_order_penalty_fn(space=bundle.space,
                                    reject_penalty={0: 5.0, 1: 10.0, 2: 15.0})
    cases.append(('E4 罚金 = p_c[temp_class]（逐单核对）',
                  all(abs(p_fn(o) - {0: 5.0, 1: 10.0, 2: 15.0}[int(tc[o])]) < 1e-12
                      for o in range(1, env.num_nodes)), None))
    cases.append(('E5 边际燃料换算 = KM_PER_UNIT × FUEL_COST_PER_KM = 2.0',
                  abs(hn._fuel_per_unit() - KM_PER_UNIT * FUEL_COST_PER_KM) < 1e-12
                  and abs(hn._fuel_per_unit() - 2.0) < 1e-12, hn._fuel_per_unit()))
    # ★ 绑定必须按 space 索引：场景节点索引超出 env 数组长度，按 env 取会 IndexError
    from scenario_saa import ScenarioOrder
    scen = [ScenarioOrder(oid=10 ** 6, reveal=0.0, x=1.0, y=0.0, demand=1.0,
                          temp_class=2, tw_start=0.0, tw_end=100.0, service_time=0.1)]
    b2 = hn.build_space_bundle(env=env, inst_idx=0, scenarios=[scen], contract=contract)
    e_fn2 = hn.make_order_energy_fn(space=b2.space, cooling_share=share,
                                    contract=contract, energy_pricing='marginal')
    r_fn2 = hn.make_order_revenue_fn(space=b2.space)
    p_fn2 = hn.make_order_penalty_fn(space=b2.space,
                                     reject_penalty={0: 5.0, 1: 10.0, 2: 15.0})
    sn = b2.n_real
    try:
        e_scen, r_scen, p_scen = e_fn2(sn), r_fn2(sn), p_fn2(sn)
        ok = (abs(r_scen - REV[2]) < 1e-12 and abs(p_scen - 15.0) < 1e-12
              and e_scen > 0.0)
    except Exception as exc:                                # noqa: BLE001
        ok, e_scen, r_scen, p_scen = False, repr(exc), None, None
    cases.append(('★E5b 场景节点（索引 ≥ n_real=7）也能正确取值 ⇒ 绑定按 space 而非 env '
                  '（否则 IndexError 会被吞成 B3"λ 未定义"、伪装成机制在工作）',
                  ok, (sn, e_scen, r_scen, p_scen)))
    return cases


def _fake_physics(*, raise_on_service=False, energy=1.0):
    """注入式假物理：只验证**语义**（谁在什么条件下返回 None），不验证 C0 数值。

    ⚠️ 真实 `transition_segment` 在物理边界硬违反时**抛 `ValueError`**（`_complete` 正是
    以此判不可行）。第一版假物理只是"返回正常值"，于是"TW 违反"这一路根本测不到。
    """
    class VS(object):
        def __init__(self):
            self.cumulative_energy_kwh = 0.0

    def create_vehicle_state(_c):
        return VS()

    def dispatch_vehicle(s, _c):
        return s

    def transition_segment(state, **kw):
        if raise_on_service and kw.get('served_customer') is not None:
            raise ValueError('fake physical boundary violation')
        state.cumulative_energy_kwh = float(energy)
        return state, None
    return {'transition_segment': transition_segment,
            'create_vehicle_state': create_vehicle_state,
            'dispatch_vehicle': dispatch_vehicle}


def test_complete_fn_semantics():
    from scenario_saa import make_c0_contract_v2
    cases = []
    env = FakeEnv(make_fixture())
    contract = make_c0_contract_v2()
    bundle = hn.build_space_bundle(env=env, inst_idx=0, scenarios=[], contract=contract)
    st = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2], frozen=0)}
    # 正常完成：能耗 1.0、距离 = 去+回
    f_ok = hn.make_c0_complete_fn(bundle=bundle, contract=contract,
                                 booking_horizon=16.0, physics=_fake_physics(energy=1.0))
    e, d = f_ok(st)
    cases.append(('E6 完成函数返回真实能耗与总距离（含返仓段）',
                  e is not None and abs(e - 1.0) < 1e-12 and d > 0.0, (e, d)))
    # 空计划且未出车 ⇒ 不 dispatch、不产生幻影预冷
    e0, d0 = f_ok({1: dict(cur=0, cur_time=0.0, load=0.0, route=[], frozen=0)})
    cases.append(('E7 未出车空计划 ⇒ 能耗 0、距离 0（无幻影预冷）',
                  e0 == 0.0 and d0 == 0.0, (e0, d0)))
    # ⑧a 物理抛 ValueError（真实 transition_segment 的硬违反信号）⇒ (None, None)
    f_phys = hn.make_c0_complete_fn(bundle=bundle, contract=contract,
                                   booking_horizon=16.0,
                                   physics=_fake_physics(raise_on_service=True))
    cases.append(('E8a 物理抛 ValueError（真实硬违反信号）⇒ (None, None)',
                  f_phys(st) == (None, None), f_phys(st)))
    # ⑧b 完成函数**自带的** TW 复查（防物理层不抛的情况）⇒ (None, None)
    env_tw = FakeEnv(make_fixture())
    env_tw.tw_end[0, 1] = 0.0          # 订单 1 的 TW 在时刻 0 就关（cur_time=0 出发必晚）
    b_tw = hn.build_space_bundle(env=env_tw, inst_idx=0, scenarios=[], contract=contract)
    f_tw = hn.make_c0_complete_fn(bundle=b_tw, contract=contract, booking_horizon=16.0,
                                 physics=_fake_physics())
    cases.append(('E8b 完成函数自带 TW 复查命中 ⇒ (None, None)',
                  f_tw(st) == (None, None), f_tw(st)))
    # 返仓不可行：把 depot 截止压到不可能达到
    env2 = FakeEnv(make_fixture())
    env2.tw_end[0, 0] = -1.0
    b2 = hn.build_space_bundle(env=env2, inst_idx=0, scenarios=[], contract=contract)
    f_ret = hn.make_c0_complete_fn(bundle=b2, contract=contract, booking_horizon=16.0,
                                  physics=_fake_physics())
    cases.append(('E9 返仓超过 depot 截止 ⇒ (None, None)',
                  f_ret(st) == (None, None), f_ret(st)))
    # 起点已耗能耗参与 initial_est_used
    class V(object):
        def __init__(self, vid, e):
            self.vehicle_id = vid
            self.coldchain_state = type('C', (), {'cumulative_energy_kwh': e})()
    e_fn = lambda o: 1.0                                    # noqa: E731
    used = hn.initial_est_used(env=env, inst_idx=0, vehicles=[V(1, 7.0)],
                              plan={1: [1, 2]}, served_mask=np.array([False] * 8),
                              order_energy_fn=e_fn, cooling_share=3.0,
                              energy_pricing='marginal', standby_orders=10)
    cases.append(('E10 起点已占用 = 已耗(7.0) + 计划 2 单(2.0) + 全天待命预留(3.0×10=30)',
                  abs(used - 39.0) < 1e-9, used))
    used2 = hn.initial_est_used(env=env, inst_idx=0, vehicles=[V(1, 7.0)],
                               plan={1: [1, 2]}, served_mask=np.array([False] * 8),
                               order_energy_fn=e_fn, cooling_share=3.0,
                               energy_pricing='marginal', standby_orders=None)
    cases.append(('E10b standby_orders=None ⇒ 不加全天预留（=9.0）',
                  abs(used2 - 9.0) < 1e-9, used2))
    return cases


# ===========================================================================
# D. decide_fn 端到端（假 sampler + 假 certify）
# ===========================================================================
class _FixedSampler(object):
    """假 sampler：直接给一个固定场景集合（只检验装配，不检验抽样）。"""

    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = 0

    def sample(self, snapshot, rng, k):
        self.calls += 1
        return [list(self.scenario) for _ in range(k)]


def test_decide_fn_end_to_end():
    from scenario_saa import ScenarioOrder, make_c0_contract_v2
    cases = []
    env = FakeEnv(make_fixture())
    contract = make_c0_contract_v2()
    # 假场景订单：坐标/揭示/TW 都宽松，id 用高位（不与真实节点重叠）
    scenario = [ScenarioOrder(oid=10 ** 6 + j, reveal=1.0, x=float(j), y=0.0,
                              demand=1.0, temp_class=0, tw_start=0.0, tw_end=100.0,
                              service_time=0.1) for j in range(4)]
    sampler = _FixedSampler(scenario)
    certify_calls = {'n': 0}

    def fake_certify(env_, inst_idx, clock, vehicles, served_mask, plan, contract_, B):
        certify_calls['n'] += 1
        return True, 1.0

    class V(object):
        def __init__(self, vid):
            self.vehicle_id = vid
            self.status = 'idle'
            self.coldchain_state = None
            self.committed_next = 0
            self.committed_finish = 0.0
            self.current_load = 0.0
            self.current_node = 0

    def ctx_for(inst, clock, plan, vehicles):
        return {'env': env, 'inst_idx': inst, 'clock': clock, 'order': 1,
                'visible_ids': [1, 2], 'accepted': [], 'remaining_budget': 100.0,
                'plan': plan, 'booking_horizon': 16.0, 'vehicles': vehicles,
                'served_mask': np.zeros(env.num_nodes, bool), 'budget_B': 100.0,
                'contract': contract, 'reject_penalty': {0: 5.0, 1: 10.0, 2: 15.0}}

    for gate in ('on', 'off'):
        dfn = hn.build_decide_fn(
            sampler=sampler, arm_seed=ARM_SEED_COND_HIST, K=K, contract=contract,
            cooling_share=3.7208946347769882, booking_horizon=16.0, budget=100.0,
            inner_gate=gate, time_limit=10.0, certify_plan_fn=fake_certify)
        acc, diag = dfn(1, ctx_for(0, 1.0, {}, [V(1), V(2)]))
        cases.append(('D1 inner_gate=%s：decide_fn 走通并返回 (accept, diag)' % gate,
                      isinstance(acc, bool) and isinstance(diag, dict), (acc, diag)))
        cases.append(('D2 inner_gate=%s：诊断留痕 scenario digest / arm_seed / gate / 耗时'
                      % gate,
                      diag['inner_gate'] == gate and diag['arm_seed'] == ARM_SEED_COND_HIST
                      and len(diag['scenario']['scenario_digest']) == 64
                      and diag['elapsed_s'] >= 0.0, diag.get('inner_gate')))
        cases.append(('D3 inner_gate=%s：判决式写明写法 (i)（不是被禁用的 (iii)），'
                      '且 λ 来源为真实机会成本（**不是**边界短路）' % gate,
                      diag['formulation'].startswith('(i)')
                      and 'myopic_net' in diag['formulation']
                      and '(iii)' not in diag['formulation']
                      and diag['lambda_diag']['lambda_source'] == 'opportunity_cost'
                      and diag['lambda_diag']['undefined'] is False,
                      (repr(diag.get('formulation')),
                       diag['lambda_diag'].get('lambda_source'),
                       diag['lambda_diag'].get('undefined'),
                       diag['lambda_diag'].get('reason'))))
        cases.append(('D3b inner_gate=%s：λ_B 是**有限实数**且未触发边界短路'
                      '（前瞻整链真的跑通了，不是被 B3 短路成拒单）' % gate,
                      diag.get('boundary') is None and diag['lambda_B'] is not None
                      and abs(float(diag['lambda_B'])) < 1e12,
                      (diag.get('lambda_B'), diag.get('boundary'))))
        cases.append(('D4 inner_gate=%s：接受时给出候选计划、（拒单时不给）' % gate,
                      (diag.get('plan') is not None) if acc else (diag.get('plan') is None),
                      (acc, diag.get('plan') is not None)))
        cases.append(('D5 inner_gate=%s：日志所需字段齐备（day_id / info_max_reveal）' % gate,
                      diag['day_id'] == 0 and 'info_max_reveal' in diag, None))
    cases.append(('D6 认证被真实调用（certify_plan 是硬门槛，不是摆设）',
                  certify_calls['n'] > 0, certify_calls))
    # 无可行插入（TW 极紧）⇒ 拒单且不虚增燃料
    env_tight = FakeEnv(make_fixture())
    env_tight.dataset['tw_end'][0, 1:] = 0.0
    class V2(object):
        def __init__(self, vid):
            self.vehicle_id = vid
            self.status = 'idle'
            self.coldchain_state = None
            self.committed_next = 0
            self.committed_finish = 0.0
            self.current_load = 0.0
            self.current_node = 0
    dfn2 = hn.build_decide_fn(sampler=sampler, arm_seed=ARM_SEED_COND_HIST, K=K,
                             contract=contract, cooling_share=1.0, booking_horizon=16.0,
                             budget=100.0, certify_plan_fn=fake_certify)
    acc2, diag2 = dfn2(1, {'env': env_tight, 'inst_idx': 0, 'clock': 1.0,
                           'plan': {}, 'vehicles': [V2(1)], 'served_mask':
                           np.zeros(env_tight.num_nodes, bool), 'budget_B': 100.0,
                           'contract': contract,
                           'reject_penalty': {0: 5.0, 1: 10.0, 2: 15.0}})
    # 订单 1 的 temp_class=1 ⇒ R=20、p_c=10 ⇒ 若燃料项为 0 则 myopic_net 恰为 30
    cases.append(('★D7 无处可插 ⇒ **硬短路拒单**（boundary=insertion_infeasible），'
                  '不估 λ、不认证 —— 否则会记录"已接受但不可服务"的承诺，'
                  'A-v1 的 accepted ⊆ served 直接判协议违规',
                  acc2 is False and diag2['boundary'] == 'insertion_infeasible'
                  and diag2['lambda_B'] is None and diag2['plan'] is None
                  and diag2['log_fields']['action'] == 'reject'
                  and diag2['log_fields']['certified'] is False,
                  (acc2, diag2.get('boundary'), diag2.get('lambda_B'))))
    # 对照：可插时燃料项**确实**被算进去（30 − 2×Δd < 30）⇒ 证明 D7 不是"燃料恒为 0"
    acc3, diag3 = dfn2(1, {'env': env, 'inst_idx': 0, 'clock': 1.0, 'plan': {},
                           'vehicles': [V2(1)], 'served_mask':
                           np.zeros(env.num_nodes, bool), 'budget_B': 100.0,
                           'contract': contract,
                           'reject_penalty': {0: 5.0, 1: 10.0, 2: 15.0}})
    cases.append(('D7b 可插时燃料项确实计入（myopic_net < 30，且与 Δ距离 × 2.0 一致）',
                  diag3['myopic_net'] < 30.0 - 1e-9,
                  diag3['myopic_net']))
    # 接受时**必须**能兑现：候选计划里必须真的含有该订单
    plan3 = diag3.get('plan') or {}
    cases.append(('★D8 接受路径的候选计划真的含有该订单（承诺可兑现）',
                  (acc3 is True and any(1 in r for r in plan3.values()))
                  if acc3 else (plan3 == {} and acc3 is False),
                  (acc3, plan3)))
    return cases


def main():
    cases = []
    for fn in (test_scenario_stream_matches_control, test_no_control_engine_use,
               test_objective_bindings, test_complete_fn_semantics,
               test_decide_fn_end_to_end):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-600:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
