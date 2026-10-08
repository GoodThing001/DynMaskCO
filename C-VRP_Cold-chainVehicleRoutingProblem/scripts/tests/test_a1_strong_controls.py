# -*- coding: utf-8 -*-
"""A1 强动态对照两臂回归测试（2026-10-03，T34/T35）。

T34 合议臂：投票语义（多数接/平票拒）、超时语义沿用、计数口径、真实小天端到端。
T35 rollout 臂：h=2 截断（全窗口内 ⇒ 与全影子逐位一致）、V 特征确定性/维度/语义、
V 预测（零权重=0、b 平移、标准化）、V 权重版本守卫。
"""
import os
import sys
import time

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from a1_strong_controls import (ConsensusReplanner, RolloutVReplanner,  # noqa: E402
                                V_FEATURE_DIM, V_FEATURE_VERSION, a1_v_features)
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1  # noqa: E402
from run_exp_reserve import (BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,  # noqa: E402
                             generate_dataset)
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, ScenarioOrder,  # noqa: E402
                          build_history, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv  # noqa: E402
from test_a1_saa_correctness import _FakeEnv, _mk_day, _mk_replanner, P_C  # noqa: E402


def _mk_a1(budget=1e9, K=10, time_limit=60.0, arm='consensus', v_ckpt=None, h=2.0):
    contract = make_c0_contract_v2()
    sampler = UncondHistoricalSampler(build_history(generate_dataset(5, 30, 20260925)))
    kwargs = dict(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                  contract=contract, cooling_share=2.0, sampler=sampler,
                  reject_penalty=P_C, K=K, time_limit=time_limit, arm_seed=7001)
    if arm == 'consensus':
        rp = ConsensusReplanner(**kwargs)
    else:
        rp = RolloutVReplanner(**kwargs, v_ckpt=v_ckpt, h=h)
    return rp, contract


def test_t34a_consensus_vote_semantics():
    """合议臂：多数接/平票拒；超时路径沿用旧语义；完成率计数。"""
    import scenario_saa as ssa
    rp, contract = _mk_a1(time_limit=1.0, arm='consensus')
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, 1, 0.5, 22.0)]), contract)
    env = _FakeEnv(ds)
    rp._id_map = {1: 1}
    rp._plan = {1: [1]}
    rp._try_insert_certified = lambda *a, **k: True

    def _run(script, fake_pc):
        rp._accepted, rp._rejected, rp.timeouts = set(), set(), 0
        rp.scen_completed_sum, rp.n_decisions = 0, 0
        rp._sim_scenario = script
        real_pc = ssa.time.perf_counter
        ssa.time.perf_counter = fake_pc
        try:
            rp._saa_decide(env, 0, 1.0, [], np.zeros(2, bool), 1, [[]] * 10, 0.0)
        finally:
            ssa.time.perf_counter = real_pc

    # ① 6/10 票接单：投票统计 d_i = acc − rej + pc_o（rej 侧已含 −pc_o；pc_o=10）——
    # acc=+1 ⇒ d=+11 接；acc=−20 ⇒ d=−10 拒（6:4）
    calls = {'i': 0}

    def script1(env, i, c, st, scen, space):
        calls['i'] += 1
        return 1.0 if calls['i'] % 2 == 1 and calls['i'] <= 11 else -20.0 if calls['i'] % 2 == 1 else 0.0

    def never():  # 永不超时
        return 0.0
    _run(script1, never)
    assert 1 in rp._accepted and 1 not in rp._rejected, (rp._accepted, rp._rejected)
    assert rp.timeouts == 0 and rp.n_decisions == 1 and rp.scen_completed_sum == 10
    assert abs(rp.saa_completion_rate - 1.0) < 1e-12
    # ② 5/5 平票拒
    calls['i'] = 0

    def script2(env, i, c, st, scen, space):
        calls['i'] += 1
        return 1.0 if calls['i'] % 2 == 1 and calls['i'] <= 9 else -20.0 if calls['i'] % 2 == 1 else 0.0
    _run(script2, never)
    assert 1 in rp._rejected and 1 not in rp._accepted
    assert rp.timeouts == 0 and rp.n_decisions == 1
    # ③ 超时路径：入口即超时 → 保计划+拒单+计次（沿用协议语义）
    _run(script2, lambda: 2.0)
    assert 1 in rp._rejected and rp.timeouts == 1 and rp.n_decisions == 0
    print('[PASS] t34a_consensus_vote_semantics (多数接/平票拒/超时旧语义/完成率)')
    return True


def test_t34b_consensus_end_to_end():
    """合议臂真实 1 天（0.05s 时限）：无崩溃、硬约束 1.0、效用有限、计数一致。"""
    rp, contract = _mk_a1(time_limit=0.05, arm='consensus')
    ds = add_v2_initial_quality(
        _mk_day([(0.6, 0.5, 2.0, i % 3, 0.05, 22.0) for i in range(1, 40)]), contract)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=3,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    traces, _ = env.run(0)
    acc, rej = set(rp._accepted), set(rp._rejected)
    ev = evaluate_trace_a1(traces, {k: v[0] for k, v in ds.items()}, contract,
                           acc, rej, 1e9, P_C)
    assert ev['hard_feasible'], ('T34b 硬约束破坏', ev['failures'])
    assert np.isfinite(ev['utility'])
    if rp.n_decisions > 0:
        assert 0.0 <= rp.saa_completion_rate <= 1.0 + 1e-12
    print('[PASS] t34b_consensus_end_to_end (to=%d dec=%d rate=%s u=%.1f)'
          % (rp.timeouts, rp.n_decisions,
             ('%.2f' % rp.saa_completion_rate) if rp.saa_completion_rate is not None
             else 'None', ev['utility']))
    return True


def test_t35a_h2_truncation_equals_full_when_all_in_window():
    """rollout 臂：全部未来单 reveal ≤ clock+h 时，_sim_scenario_h2(V=0) 与全影子逐位一致。"""
    rp, _ = _mk_a1(arm='rollout', h=2.0)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    env = _FakeEnv(ds)
    fut = [ScenarioOrder(oid=-(j + 1), reveal=3.0 + 0.05 * j,   # clock=3.0 → 全部 ≤ 3+2
                         x=float(0.2 + 0.05 * j), y=0.5, demand=2.0, temp_class=j % 3,
                         tw_start=0.0, tw_end=22.0, service_time=0.05) for j in range(6)]
    space, _ = rp._build_space(env, 0, [fut])
    scen = [space.n_real + j for j in range(6)]
    st = {1: dict(cur=0, cur_time=3.0, load=0.0, route=[1], cc=None, frozen=0)}
    full = rp._sim_scenario(env, 0, 3.0, st, scen, space)
    h2 = rp._sim_scenario_h2(env, 0, 3.0, st, scen, space)
    assert np.isfinite(full) and full > -1e5
    assert abs(full - h2) < 1e-9, (full, h2)
    # 窗口外单被截断：加一个 reveal=20 的单 → h2 不插它（V=0 ⇒ h2 ≠ full）
    fut2 = fut + [ScenarioOrder(oid=-99, reveal=20.0, x=0.9, y=0.9, demand=2.0,
                                temp_class=0, tw_start=0.0, tw_end=22.0, service_time=0.05)]
    space2, _ = rp._build_space(env, 0, [fut2])
    scen2 = [space2.n_real + j for j in range(7)]
    full2 = rp._sim_scenario(env, 0, 3.0, st, scen2, space2)
    h2_2 = rp._sim_scenario_h2(env, 0, 3.0, st, scen2, space2)
    assert np.isfinite(full2) and np.isfinite(h2_2)
    assert full2 != h2_2
    # 截断后的 h2 值 = 只有窗口内 6 单的全影子值（reveal=20 的单不进入截断影子）
    assert abs(h2_2 - full) < 1e-9, (h2_2, full)
    print('[PASS] t35a_h2_truncation_equals_full_when_all_in_window (全窗口逐位一致 + 截断正确)')
    return True


def test_t35b_v_features():
    """V 特征：维度/有限/确定性；窗口外单计数语义。"""
    rp, _ = _mk_a1(arm='rollout', h=2.0)
    ds = _mk_day([(0.6, 0.5, 2.0, 0, 0.5, 22.0)])
    env = _FakeEnv(ds)
    fut = [ScenarioOrder(oid=-(j + 1), reveal=5.0, x=0.4, y=0.4, demand=2.0,
                         temp_class=0, tw_start=0.0, tw_end=22.0, service_time=0.05)
           for j in range(4)]
    space, _ = rp._build_space(env, 0, [fut])
    st = {1: dict(cur=0, cur_time=1.0, load=5.0, route=[1], cc=None, frozen=0),
          2: dict(cur=0, cur_time=2.0, load=0.0, route=[], cc=None, frozen=0)}
    f1 = a1_v_features(space, env, 0, 1.0, st, 10.0, 100.0, 50.0, 2.0)
    f2 = a1_v_features(space, env, 0, 1.0, st, 10.0, 100.0, 50.0, 2.0)
    assert f1.shape == (V_FEATURE_DIM,)
    assert np.all(np.isfinite(f1))
    assert np.array_equal(f1, f2)
    # 语义：reveal=5 > 1+2 ⇒ n_remaining=4, tot_remaining=8；n_active=1（车1有路线）
    assert f1[11] == 4 and abs(f1[12] - 8.0) < 1e-9 and f1[10] == 1.0
    # budget−est_used 槽位
    assert abs(f1[9] - 90.0) < 1e-9
    # clock 槽位
    assert abs(f1[13] - 1.0) < 1e-9
    print('[PASS] t35b_v_features (dim=14 有限确定性 + 语义槽位)')
    return True


def test_t35c_v_predict():
    """V 预测：零权重=b 平移；标准化生效；版本守卫拒绝不匹配。"""
    import tempfile
    rp0, _ = _mk_a1(arm='rollout', h=2.0)   # v_ckpt=None → V≡0
    assert rp0._v_predict(np.ones(V_FEATURE_DIM)) == 0.0
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'v.npz')
    np.savez(p, w=np.zeros(V_FEATURE_DIM), b=7.5, f_mean=np.ones(V_FEATURE_DIM),
             f_std=np.ones(V_FEATURE_DIM), feature_version=V_FEATURE_VERSION)
    rp1, _ = _mk_a1(arm='rollout', h=2.0, v_ckpt=p)
    assert abs(rp1._v_predict(np.ones(V_FEATURE_DIM)) - 7.5) < 1e-12
    # 标准化：f=2 → (2−1)/1=1 → b + w·1 = 7.5（w=0 时与输入无关）
    assert abs(rp1._v_predict(np.full(V_FEATURE_DIM, 2.0)) - 7.5) < 1e-12
    p2 = os.path.join(tmp, 'vbad.npz')
    np.savez(p2, w=np.zeros(V_FEATURE_DIM), b=0.0, f_mean=np.zeros(V_FEATURE_DIM),
             f_std=np.ones(V_FEATURE_DIM), feature_version=V_FEATURE_VERSION + 1)
    try:
        _mk_a1(arm='rollout', h=2.0, v_ckpt=p2)
        raise AssertionError('版本不匹配必须拒绝')
    except ValueError:
        pass
    # 线性响应：w=e1, b=0, 无标准化（std 很大/mean=0）→ f[0]
    p3 = os.path.join(tmp, 'v3.npz')
    w3 = np.zeros(V_FEATURE_DIM)
    w3[0] = 3.0
    np.savez(p3, w=w3, b=0.0, f_mean=np.zeros(V_FEATURE_DIM),
             f_std=np.full(V_FEATURE_DIM, 1e9), feature_version=V_FEATURE_VERSION)
    rp3, _ = _mk_a1(arm='rollout', h=2.0, v_ckpt=p3)
    f = np.zeros(V_FEATURE_DIM)
    f[0] = 2.0
    assert abs(rp3._v_predict(f) - 3.0 * 2.0 / 1e9) < 1e-12
    print('[PASS] t35c_v_predict (V≡0 / b 平移 / 标准化 / 版本守卫)')
    return True


def test_t35d_rollout_end_to_end():
    """rollout 臂真实 1 天（0.05s 时限）：无崩溃、硬约束 1.0、效用有限。"""
    rp, contract = _mk_a1(time_limit=0.05, arm='rollout', h=2.0)
    ds = add_v2_initial_quality(
        _mk_day([(0.6, 0.5, 2.0, i % 3, 0.05, 22.0) for i in range(1, 40)]), contract)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=3,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    traces, _ = env.run(0)
    acc, rej = set(rp._accepted), set(rp._rejected)
    ev = evaluate_trace_a1(traces, {k: v[0] for k, v in ds.items()}, contract,
                           acc, rej, 1e9, P_C)
    assert ev['hard_feasible'], ('T35d 硬约束破坏', ev['failures'])
    assert np.isfinite(ev['utility'])
    print('[PASS] t35d_rollout_end_to_end (to=%d u=%.1f)' % (rp.timeouts, ev['utility']))
    return True


def test_t36_driver_a1_arms():
    """驱动集成：--arms 加 a1 两臂 + V 零权重模型，小批跑通；gate.json 结构
    （臂/主比较/a1 配对/identity.a1）完整。"""
    import json
    import tempfile
    from run_a1_step2_gate import main as step2_main
    tmp = tempfile.mkdtemp()
    vckpt = os.path.join(tmp, 'vzero.npz')
    np.savez(vckpt, w=np.zeros(V_FEATURE_DIM), b=0.0, f_mean=np.zeros(V_FEATURE_DIM),
             f_std=np.ones(V_FEATURE_DIM), feature_version=V_FEATURE_VERSION)
    out = os.path.join(tmp, 'gate')
    step2_main(['--train-instances', '8', '--gate-instances', '2', '--n-orders', '40',
                '--penalty', 'p_c=0', '--time-limit', '0.1', '--workers', '1',
                '--arms',
                'uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout,soft_knn',
                '--a1-v-ckpt', vckpt, '--a1-h', '2.0', '--softknn-mult', '1.0',
                '--out', out])
    doc = json.load(open(os.path.join(out, 'gate.json'), encoding='utf-8'))
    blk = doc['gate']['p_c=0']
    arms = blk['arms']
    assert set(arms) >= {'uncond_hist', 'cond_hist', 'a1_consensus', 'a1_rollout',
                         'soft_knn'}, arms.keys()
    assert 'main_comparison' in blk, 'cond−uncond 主比较缺失'
    for key in ('a1_consensus_minus_cond_hist', 'a1_rollout_minus_cond_hist',
                'soft_knn_minus_cond_hist', 'explicit_feat_minus_cond_hist'):
        assert key in blk, key + ' 缺失'
    assert doc['identity']['a1'] and doc['identity']['a1']['v_ckpt_sha256']
    assert doc['identity']['a1']['arms'] == ['a1_consensus', 'a1_rollout']
    assert doc['identity']['source_stable']
    # 合议臂的 SAA 完成率已入逐日行
    row0 = blk['per_day']['a1_consensus'][0]
    assert 'saa_completion_rate' in row0 and 'n_decisions' in row0
    print('[PASS] t36_driver_a1_arms (两臂入批 + 身份含 V 权重 hash + 三层对比键齐)')
    return True


if __name__ == '__main__':
    res = {
        't34a_consensus_vote_semantics': test_t34a_consensus_vote_semantics(),
        't34b_consensus_end_to_end': test_t34b_consensus_end_to_end(),
        't35a_h2_truncation_equals_full_when_all_in_window':
            test_t35a_h2_truncation_equals_full_when_all_in_window(),
        't35b_v_features': test_t35b_v_features(),
        't35c_v_predict': test_t35c_v_predict(),
        't35d_rollout_end_to_end': test_t35d_rollout_end_to_end(),
        't36_driver_a1_arms': test_t36_driver_a1_arms(),
    }
    ok = all(res.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
