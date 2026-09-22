"""JF1-H-F × common contract 集成测试（本地）。

  - 合成动态实例：complete + 0 违规 + repair_applicable=True + 记录校验；
  - 真实 R1/EDoD=0.5 val 冒烟（工程参照，非正式证据）；
  - future 扰动 parity（内容/身份/数量，JF1-H-F 为项目 P0 验证过的
    non-anticipatory baseline，经公共合同层复核）；
  - 确定性（同 seed decision_hash 一致）。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_DCC_VRP = os.path.dirname(_TESTS)
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_TESTS, _DCC_VRP, _COMMON, os.path.join(_COMMON, 'tests')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import helpers  # noqa: F401  (common/tests 合成数据)
from strict_online_runner import run_instance, load_objective_profile
from jf1hf_adapter import Jf1hfAdapter

# 正式冻结 profile v2（集成测试用正式口径；pilot 仅留给纯单元测试）
_V2_PROFILE = os.path.join(
    _COMMON, '..', '..',
    'C-VRP_Cold-chainVehicleRoutingProblem', 'results', 'o0cc', 'scale_v2',
    'objective_profile.json')


def _run(ds, seed=0):
    return run_instance(ds, 50, 4,
                        adapter_factory=Jf1hfAdapter, inst_idx=0,
                        objective='coldchain', profile=load_objective_profile(_V2_PROFILE),
                        seed=seed, data_sha256='t', instance_seed=0)


def _prefix_signature(rec):
    """扰动客户 reveal（5.0）之前的完整决策序列（B1.2 全 prefix parity）。"""
    return helpers.prefix_signature(rec, until_clock=5.0)


def test_synthetic_complete_service():
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 6.0}, seed=11)
    rec = _run(ds)
    assert rec['outcome']['complete'], f'JF1-H-F 合成实例未 complete'
    assert rec['audit']['ownership_violations'] == 0
    assert rec['audit']['terminal_unresolved'] == 0
    assert rec['outcome']['repair_applicable'] is True
    assert rec['outcome']['audit_source'] == 'common_runner + repair layer'
    assert rec['outcome']['repair_ownership_violations'] == 0
    assert rec['outcome']['repair_terminal_unresolved'] == 0
    assert all(rec['hard_vector'].values())
    print('  合成动态实例：complete + 0 违规 + repair 字段正确')


def test_real_r1_smoke():
    data_path = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
                 r'\data\baseline\50_node\val\dcc_50_r1_edod05_val.npz')
    if not os.path.exists(data_path):
        print('  SKIP：本地无 val npz')
        return
    ds = dict(np.load(data_path))
    rec = run_instance(ds, 50, 25, adapter_factory=Jf1hfAdapter, inst_idx=0,
                       objective='coldchain',
                       profile=load_objective_profile(_V2_PROFILE),
                       seed=0, data_sha256='t', instance_seed=0)
    assert rec['outcome']['complete']
    assert rec['audit']['ownership_violations'] == 0
    print(f"  R1 val 冒烟：complete，distance={rec['outcome']['distance_cost']:.3f} "
          f"（JF1-H-F 工程参照，非正式证据）")


def test_future_perturbation_parity():
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    base = _run(ds)
    sig_base = _prefix_signature(base)

    ds2 = helpers.make_synthetic_dataset(capacity_tight=True,
                                         reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    rng = np.random.RandomState(123)
    ds2['coords'][0, 5] = rng.uniform(0.1, 0.9, 2)
    ds2['coords'][0, 6] = rng.uniform(0.1, 0.9, 2)
    ds2['tw_end'][0, 5] = 7.0
    ds2['tw_end'][0, 6] = 8.5
    ds2['demands'][0, 5] = 12.0
    ds2['demands'][0, 6] = 40.0
    assert _prefix_signature(_run(ds2)) == sig_base, 'future 内容扰动改变 reveal 前决策序列'

    ds3 = helpers.make_synthetic_dataset(capacity_tight=True,
                                         reveal_spec={5: 5.0, 6: 5.0}, seed=12)
    for key in ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
                'temp_class', 'initial_quality'):
        ds3[key][0, 5], ds3[key][0, 6] = ds3[key][0, 6].copy(), ds3[key][0, 5].copy()
    assert _prefix_signature(_run(ds3)) == sig_base, 'future 身份扰动改变 reveal 前决策序列'

    ds4 = helpers.make_synthetic_dataset(n_customers=7, capacity_tight=True,
                                         reveal_spec={5: 5.0, 6: 5.0, 7: 6.0},
                                         seed=12)
    assert _prefix_signature(_run(ds4)) == sig_base, 'future 数量扰动改变 reveal 前决策序列'
    print('  future 内容/身份/数量扰动全 prefix parity 通过（JF1-H-F 经公共合同层）')


def test_determinism():
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 5.0}, seed=13)
    r1 = _run(ds, seed=42)
    r2 = _run(ds, seed=42)
    assert r1['decision_hash'] == r2['decision_hash']
    print('  同 seed decision_hash 一致')


def main():
    test_synthetic_complete_service()
    test_real_r1_smoke()
    test_future_perturbation_parity()
    test_determinism()
    print('PASS test_jf1hf_integration')


if __name__ == '__main__':
    main()
