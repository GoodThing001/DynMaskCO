"""B2 步骤 8/9/11/12 集成：future 扰动 parity、frozen prefix、deterministic
parity、late reveal 完整服务——PyVRP-RH-D 走完整 common runner。

适配器：固定 max_iterations + seed（确定性协议），无 warm start、无 fallback。
"""
import os
import sys

_DCC_TESTS = os.path.dirname(os.path.abspath(__file__))
_DCC_VRP = os.path.dirname(_DCC_TESTS)
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_TESTS, _DCC_VRP, _COMMON, os.path.join(_COMMON, 'tests')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import helpers  # noqa: F401
from strict_online_runner import run_instance
from pyvrp_adapter import PyVRPRHDAdapter
from coldchain_contract import default_pilot_profile


def _factory(max_iterations=300, seed=0):
    return lambda: PyVRPRHDAdapter(max_iterations=max_iterations, seed=seed)


def _run(ds, seed=0, max_iterations=300):
    return run_instance(ds, capacity=50, num_vehicles=4,
                        adapter_factory=_factory(max_iterations, seed=0),
                        inst_idx=0, objective='coldchain',
                        profile=default_pilot_profile(), seed=seed,
                        data_sha256='t', instance_seed=0)


def _prefix_signature(rec):
    """扰动客户 reveal（5.0）之前的完整决策序列（B1.2 全 prefix parity）。"""
    return helpers.prefix_signature(rec, until_clock=5.0)


def test_future_perturbation_parity():
    """步骤 8：future 内容/身份/数量扰动不改变 t=0 动作。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=7)
    base = _run(ds)
    sig_base = _prefix_signature(base)

    ds2 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=7)
    rng = np.random.RandomState(123)
    ds2['coords'][0, 5] = rng.uniform(0.1, 0.9, 2)
    ds2['coords'][0, 6] = rng.uniform(0.1, 0.9, 2)
    ds2['tw_end'][0, 5] = 7.0
    ds2['tw_end'][0, 6] = 8.5
    ds2['demands'][0, 5] = 12.0
    ds2['demands'][0, 6] = 40.0
    pert = _run(ds2)
    assert _prefix_signature(pert) == sig_base, 'future 内容扰动改变了 reveal 前决策序列'

    ds3 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=7)
    for key in ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
                'temp_class', 'initial_quality'):
        ds3[key][0, 5], ds3[key][0, 6] = ds3[key][0, 6].copy(), ds3[key][0, 5].copy()
    pert3 = _run(ds3)
    assert _prefix_signature(pert3) == sig_base, 'future 身份扰动改变了 reveal 前决策序列'

    ds4 = helpers.make_synthetic_dataset(n_customers=7,
                                         reveal_spec={5: 5.0, 6: 5.0, 7: 6.0},
                                         seed=7)
    pert4 = _run(ds4)
    assert _prefix_signature(pert4) == sig_base, 'future 数量扰动改变了 reveal 前决策序列'
    print('  future 内容/身份/数量扰动全 prefix parity 通过（PyVRP-RH-D）')


def test_frozen_prefix_and_writeback():
    """步骤 9：frozen prefix 由 Bridge 强制（proposal 键集合 == replan_ids），
    记录通过校验即证明；所有 plan_diff writeback_ok=True。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=8)
    rec = _run(ds)
    assert rec['protocol']['error'] is None
    assert rec['audit']['ownership_violations'] == 0
    for a in rec['actions']:
        assert a.get('writeback_ok') is True
    print('  frozen prefix / writeback 通过（audit 0 违规）')


def test_deterministic_parity():
    """步骤 11：固定 seed + 固定迭代数两次运行 decision_hash 完全一致。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=9)
    rec1 = _run(ds, seed=42)
    rec2 = _run(ds, seed=42)
    assert rec1['decision_hash'] == rec2['decision_hash'], \
        '同 seed 两次运行 decision_hash 不一致（PyVRP 非确定）'
    assert rec1['actions'] == rec2['actions']
    print('  deterministic parity 通过（同 seed decision_hash 一致）')


def test_late_reveal_full_service():
    """步骤 12：late reveal 合成动态实例完整服务（每客户恰好一次）。"""
    from collections import Counter
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 6.0}, seed=10)
    rec = _run(ds, max_iterations=500)
    assert rec['outcome']['complete'], \
        f"late reveal 未 complete: {rec['outcome']['n_unserved']} unserved"
    counts = Counter()
    for v in rec['execution_trace']['vehicles']:
        for s in v['services']:
            if s['picked_order_id'] is not None:
                counts[s['picked_order_id']] += 1
    universe = [i for i in range(1, len(ds['demands'][0]))
                if ds['demands'][0, i] > 0]
    for c in universe:
        assert counts[c] == 1, f'客户 {c} 服务 {counts[c]} 次'
    revealed = [c for ev in rec['events'] for c in ev['revealed_customer_ids']]
    assert 5 in revealed and 6 in revealed
    print('  late reveal 完整服务通过（含 reveal 事件记录）')


def main():
    test_future_perturbation_parity()
    test_frozen_prefix_and_writeback()
    test_deterministic_parity()
    test_late_reveal_full_service()
    print('PASS test_adapter_integration')


if __name__ == '__main__':
    main()
