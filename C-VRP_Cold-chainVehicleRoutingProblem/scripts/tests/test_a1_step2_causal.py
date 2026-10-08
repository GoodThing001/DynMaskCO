"""A-v1 步骤 2：因果隔离（隐藏未来不变性）+ 采样器边界 + 共用决策器硬约束测试。"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history, make_c0_contract_v2,
                          VisibleSnapshot)
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}


def _single(ds):
    return {k: v[0] for k, v in ds.items()}


def _mk_day(revealed, hidden):
    """revealed/hidden: list[(x, y, demand, class, reveal)]。revealed 揭示早、hidden 揭示晚。"""
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
    for j, (x, y, dem, c, rev) in enumerate(orders, start=1):
        coords[0, j] = [x, y]
        demands[0, j] = dem
        tc[0, j] = c
        reveal[0, j] = rev
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
            'service_time': service, 'reveal_time': reveal, 'temp_class': tc}


class _Spy:
    """包裹真实采样器，记录每次 sample 的 (clock, snapshot订单摘要, 场景摘要)。"""

    def __init__(self, inner):
        self.inner = inner
        self.log = []

    def sample(self, snapshot, rng, k):
        scens = self.inner.sample(snapshot, rng, k)
        self.log.append((
            round(snapshot.clock, 6),
            tuple((o.oid, round(o.reveal, 6), round(o.x, 6), round(o.y, 6),
                   round(o.demand, 6), o.temp_class) for o in snapshot.orders),
            tuple(tuple((o.oid, round(o.reveal, 6), o.temp_class) for o in s) for s in scens)))
        return scens


def _run(ds, contract, budget, sampler, arm_seed=7001, K=5, time_limit=60.0, num_vehicles=2):
    rp = SaaReplanner(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=2.0, sampler=sampler,
                      reject_penalty=P_C, K=K, time_limit=time_limit, arm_seed=arm_seed)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT,
                          replanner=rp, coldchain_contract=contract,
                          booking_horizon=BOOKING_HORIZON)
    traces, _ = env.run(0)
    return traces, set(rp._accepted), set(rp._rejected)


def test_hidden_future_invariance():
    """固定可见前缀与种子，只换测试日隐藏未来 → 快照/采样/已揭示部分决策必须不变。"""
    contract = make_c0_contract_v2()
    revealed = [(0.6, 0.5, 2.0, 1, 0.5), (0.4, 0.5, 2.0, 0, 0.5)]
    hidden_a = [(0.7, 0.5, 2.0, 2, 2.0), (0.3, 0.5, 2.0, 2, 2.0)]
    hidden_b = [(0.7, 0.7, 5.0, 0, 2.0), (0.3, 0.3, 5.0, 1, 2.0)]
    ds_a = add_v2_initial_quality(_mk_day(revealed, hidden_a), contract)
    ds_b = add_v2_initial_quality(_mk_day(revealed, hidden_b), contract)
    hist = build_history(generate_dataset(20, 60, 20260925))
    spy_a, spy_b = _Spy(UncondHistoricalSampler(hist)), _Spy(UncondHistoricalSampler(hist))
    tr_a, acc_a, rej_a = _run(ds_a, contract, 1e9, spy_a)
    tr_b, acc_b, rej_b = _run(ds_b, contract, 1e9, spy_b)
    calls_a = [c for c in spy_a.log if abs(c[0] - 0.5) < 1e-6]
    calls_b = [c for c in spy_b.log if abs(c[0] - 0.5) < 1e-6]
    assert calls_a, "expected a sample call at t=0.5"
    assert calls_a == calls_b, "snapshot/samples at t=0.5 must be identical under hidden-future swap"
    assert (acc_a & {1, 2}) == (acc_b & {1, 2}), "decisions on revealed orders must be identical"
    assert (rej_a & {1, 2}) == (rej_b & {1, 2}), "decisions on revealed orders must be identical"
    for tr, acc, rej, ds in ((tr_a, acc_a, rej_a, ds_a), (tr_b, acc_b, rej_b, ds_b)):
        r = evaluate_trace_a1(tr, _single(ds), contract, acc, rej, 1e9, P_C)
        assert r['hard_feasible'], r['failures']
    print("[PASS] hidden_future_invariance")
    return True


def test_snapshot_temp_ids():
    """快照只含临时编号与已揭示字段，无 env/真实索引。"""
    contract = make_c0_contract_v2()
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, 1, 0.5), (0.4, 0.5, 2.0, 0, 0.5)], []),
                                contract)
    seen = []

    class SpyS(UncondHistoricalSampler):
        def sample(self, snapshot, rng, k):
            seen.append(snapshot)
            return [[] for _ in range(k)]

    _run(ds, contract, 1e9, SpyS(build_history(generate_dataset(5, 30, 20260925))))
    assert seen, "snapshot should be built"
    snap = seen[0]
    assert isinstance(snap, VisibleSnapshot)
    for o in snap.orders:
        assert o.oid > 0 and o.reveal <= snap.clock + 1e-6 and o.tw_end >= o.tw_start
    for v in snap.vehicles:
        assert v.node == 0 or v.node > 0
        assert all(x > 0 for x in v.plan_tail)
    assert all(x > 0 for x in snap.accepted | snap.rejected)
    print("[PASS] snapshot_temp_ids")
    return True


def test_sampler_edges():
    """零未来 / 数量可变 / 揭示时间晚于时钟 / TW 合法 / 负临时编号。"""
    contract = make_c0_contract_v2()
    hist = build_history(generate_dataset(30, 80, 20260925))
    early = sorted((o for day in hist for o in day), key=lambda o: o.reveal)[:20]
    snap = VisibleSnapshot(clock=10.0, orders=tuple(early), accepted=frozenset(),
                           rejected=frozenset(), vehicles=(), energy_used=0.0,
                           budget=700.0, booking_horizon=16.0, capacity=50.0)
    for sampler in (UncondHistoricalSampler(hist), CondHistoricalSampler(hist),
                    ExplicitFeatureSampler(hist)):
        scens = sampler.sample(snap, np.random.default_rng(12345), 20)
        counts = set()
        for s in scens:
            counts.add(len(s))
            for o in s:
                assert o.reveal > snap.clock + 1e-6
                assert o.tw_end >= o.tw_start
                assert o.oid < 0
        assert len(counts) > 1, "scenario counts should vary"
    snap_late = VisibleSnapshot(clock=20.0, orders=tuple(early), accepted=frozenset(),
                                rejected=frozenset(), vehicles=(), energy_used=0.0,
                                budget=700.0, booking_horizon=16.0, capacity=50.0)
    for sampler in (UncondHistoricalSampler(hist), CondHistoricalSampler(hist),
                    ExplicitFeatureSampler(hist)):
        scens = sampler.sample(snap_late, np.random.default_rng(1), 10)
        assert all(len(s) == 0 for s in scens), "no future orders past all reveals"
    print("[PASS] sampler_edges")
    return True


def test_shared_decider_accept_reject():
    """共用决策器：预算宽松 SAA 接受 / 超预算硬认证拒绝；终局 v2 评价硬可行、鱼可售。"""
    contract = make_c0_contract_v2()
    ds = add_v2_initial_quality(_mk_day([(0.6, 0.5, 2.0, 1, 0.5)], []), contract)
    hist = build_history(generate_dataset(5, 30, 20260925))

    class Empty(UncondHistoricalSampler):
        def sample(self, snapshot, rng, k):
            return [[] for _ in range(k)]

    tr, acc, rej = _run(ds, contract, 1e9, Empty(hist))
    assert acc == {1} and rej == set(), (acc, rej)
    r = evaluate_trace_a1(tr, _single(ds), contract, acc, rej, 1e9, P_C)
    assert r['hard_feasible'], r['failures']
    assert r['n_fish_unsalable'] == 0

    tr2, acc2, rej2 = _run(ds, contract, 10.0, Empty(hist))
    assert acc2 == set() and rej2 == {1}, (acc2, rej2)
    r2 = evaluate_trace_a1(tr2, _single(ds), contract, acc2, rej2, 10.0, P_C)
    assert r2['hard_feasible'], r2['failures']
    print("[PASS] shared_decider_accept_reject")
    return True


def test_capacity_boundary_no_crash():
    """满容量边界（float32 噪声使累计载重恰在 50+1e-9 与 50+1e-6 之间）不崩溃、正确拒绝。"""
    contract = make_c0_contract_v2()
    ds = _mk_day([(0.6, 0.5, 16.666667, 0, 0.5),
                  (0.4, 0.5, 16.666667, 0, 0.5),
                  (0.5, 0.6, 16.666667, 0, 0.5)], [])
    ds = add_v2_initial_quality(ds, contract)
    total = float(np.sum(ds['demands'][0, 1:]))
    print(f"  [info] float32 demand sum = {total!r}")
    hist = build_history(generate_dataset(5, 30, 20260925))

    class Empty(UncondHistoricalSampler):
        def sample(self, snapshot, rng, k):
            return [[] for _ in range(k)]

    tr, acc, rej = _run(ds, contract, 1e9, Empty(hist), num_vehicles=1)
    r = evaluate_trace_a1(tr, _single(ds), contract, acc, rej, 1e9, P_C)
    assert r['hard_feasible'], r['failures']
    # 前两单必接；第三单累计载重越过合同 50+1e-9 硬界 → 必须拒绝而非崩溃
    assert acc == {1, 2}, (acc, rej)
    assert rej == {3}, (acc, rej)
    print("[PASS] capacity_boundary_no_crash")
    return True


def test_hidden_future_invariance_maskco():
    """MaskCO 采样器（随机权重模型）同样满足隐藏未来不变性：快照/采样/决策不变。"""
    from flax import nnx
    from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools)
    contract = make_c0_contract_v2()
    revealed = [(0.6, 0.5, 2.0, 1, 0.5), (0.4, 0.5, 2.0, 0, 0.5)]
    hidden_a = [(0.7, 0.5, 2.0, 2, 2.0), (0.3, 0.5, 2.0, 2, 2.0)]
    hidden_b = [(0.7, 0.7, 5.0, 0, 2.0), (0.3, 0.3, 5.0, 1, 2.0)]
    ds_a = add_v2_initial_quality(_mk_day(revealed, hidden_a), contract)
    ds_b = add_v2_initial_quality(_mk_day(revealed, hidden_b), contract)
    hist = build_history(generate_dataset(20, 60, 20260925))
    model = MaskCOScenarioModel(dim=64, arm='random', rngs=nnx.Rngs(42))
    mk = lambda: MaskCOScenarioSampler(model, HistoryPools(hist), m_max=60)
    spy_a, spy_b = _Spy(mk()), _Spy(mk())
    tr_a, acc_a, rej_a = _run(ds_a, contract, 1e9, spy_a)
    tr_b, acc_b, rej_b = _run(ds_b, contract, 1e9, spy_b)
    calls_a = [c for c in spy_a.log if abs(c[0] - 0.5) < 1e-6]
    calls_b = [c for c in spy_b.log if abs(c[0] - 0.5) < 1e-6]
    assert calls_a and calls_a == calls_b, "maskco snapshots/samples at t=0.5 must match"
    assert (acc_a & {1, 2}) == (acc_b & {1, 2}) and (rej_a & {1, 2}) == (rej_b & {1, 2})
    print("[PASS] hidden_future_invariance_maskco")
    return True


if __name__ == '__main__':
    res = {
        'hidden_future_invariance': test_hidden_future_invariance(),
        'hidden_future_invariance_maskco': test_hidden_future_invariance_maskco(),
        'snapshot_temp_ids': test_snapshot_temp_ids(),
        'sampler_edges': test_sampler_edges(),
        'shared_decider_accept_reject': test_shared_decider_accept_reject(),
        'capacity_boundary_no_crash': test_capacity_boundary_no_crash(),
    }
    ok = all(res.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
