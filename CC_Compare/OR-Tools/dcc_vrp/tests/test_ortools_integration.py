"""OR7-7/8/9：adapter 集成（complete + hard vector + trace replay）、
future 全 prefix parity、确定性（3 次重复）。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common', 'tests')))

import numpy as np

import helpers  # noqa: F401
from strict_online_runner import run_instance, load_objective_profile
from ortools_adapter import ORToolsRHDAdapter

# 正式冻结 profile v2（集成测试用正式口径；pilot 仅留纯单元测试）
_V2_PROFILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', '..', '..',
    'C-VRP_Cold-chainVehicleRoutingProblem', 'results', 'o0cc', 'scale_v2',
    'objective_profile.json')


def _v2_profile():
    import hashlib
    import json
    with open(_V2_PROFILE, 'rb') as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    profile = load_objective_profile(_V2_PROFILE)
    assert data['profile_hash'] == profile.profile_hash, 'profile_hash 不一致'
    for k in ('distance_scale', 'quality_scale', 'energy_scale',
              'lambda_quality', 'lambda_energy'):
        import math
        assert math.isfinite(float(data[k])), f'profile 字段 {k} 非有限'
    return profile, {'profile_file_sha256': sha,
                     'profile_hash': profile.profile_hash}


def _run(ds, seed=0, solution_limit=30):
    profile, ident = _v2_profile()
    return run_instance(ds, 50, 4,
                        adapter_factory=lambda: ORToolsRHDAdapter(
                            solution_limit=solution_limit, check_env=False),
                        inst_idx=0, objective='coldchain',
                        profile=profile, seed=seed,
                        data_sha256='t', instance_seed=0)


def test_complete_service_and_trace():
    """合成动态实例：complete、hard vector 全过、每客户恰好一次。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 6.0}, seed=11)
    rec = _run(ds)
    assert rec['outcome']['complete'], f'unserved={rec["outcome"]["n_unserved"]}'
    assert all(rec['hard_vector'].values()), rec['hard_vector']
    assert rec['audit']['ownership_violations'] == 0
    counts = Counter()
    for v in rec['execution_trace']['vehicles']:
        for s in v['services']:
            if s['picked_order_id'] is not None:
                counts[s['picked_order_id']] += 1
    universe = [i for i in range(1, len(ds['demands'][0]))
                if ds['demands'][0, i] > 0]
    for c in universe:
        assert counts[c] == 1, f'客户 {c} 服务 {counts[c]} 次'
    # trace replay 由 runner 内置校验（不抛即过）
    print('  complete + hard vector + 每客户恰好一次（trace replay 内置校验）')


def test_future_full_prefix_parity():
    """内容/身份/数量扰动：reveal（5.0）前完整决策序列不变。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    base = _run(ds)
    sig_base = helpers.prefix_signature(base, until_clock=5.0)

    ds2 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    rng = np.random.RandomState(123)
    ds2['coords'][0, 5] = rng.uniform(0.1, 0.9, 2)
    ds2['coords'][0, 6] = rng.uniform(0.1, 0.9, 2)
    ds2['tw_end'][0, 5] = 7.0
    ds2['tw_end'][0, 6] = 8.5
    ds2['demands'][0, 5] = 12.0
    ds2['demands'][0, 6] = 40.0
    assert helpers.prefix_signature(_run(ds2), 5.0) == sig_base, '内容扰动改变 prefix'

    ds3 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    for key in ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
                'temp_class', 'initial_quality'):
        ds3[key][0, 5], ds3[key][0, 6] = ds3[key][0, 6].copy(), ds3[key][0, 5].copy()
    assert helpers.prefix_signature(_run(ds3), 5.0) == sig_base, '身份扰动改变 prefix'

    ds4 = helpers.make_synthetic_dataset(n_customers=7,
                                         reveal_spec={5: 5.0, 6: 5.0, 7: 6.0},
                                         seed=12)
    assert helpers.prefix_signature(_run(ds4), 5.0) == sig_base, '数量扰动改变 prefix'
    print('  future 内容/身份/数量全 prefix parity 通过（OR-Tools-RH-D）')


def test_determinism_3_runs():
    """同环境同输入重复 3 次：decision_hash 完全一致。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 5.0}, seed=13)
    recs = [_run(ds, seed=42) for _ in range(3)]
    h = recs[0]['decision_hash']
    for i, r in enumerate(recs[1:], 1):
        assert r['decision_hash'] == h, f'第 {i+1} 次运行 decision_hash 不一致'
    assert recs[0]['actions'] == recs[1]['actions'] == recs[2]['actions']
    print('  3 次重复 decision_hash / actions 完全一致（无 random_seed，'
          '固定 solution_limit + 单线程确定性）')


def main():
    test_complete_service_and_trace()
    test_future_full_prefix_parity()
    test_determinism_3_runs()
    print('PASS test_ortools_integration')


if __name__ == '__main__':
    main()
