"""B1.2 测试：native bridge 环境不可变保护 + repair 属性边界。

  - 恶意 native adapter 改写 coords / dist_mat / reveal_time / capacity
    → 必须被 NativeBridgeReplanner 以 ContractViolation 拒绝；
  - 无 repair 层的 native adapter：bridge 不暴露 repair_stats
    （_eval 不崩溃），outcome repair_applicable=False；
  - 受信任 native adapter（项目 GreedyReplanner 包装）经完整 runner：
    记录通过校验、终局 complete。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import numpy as np

import helpers
from method_adapter import (NativeReplanner, NativeBridgeReplanner,
                            ContractViolation)
from strict_online_env import VehicleState, GreedyReplanner
from strict_online_runner import run_instance
from coldchain_contract import default_pilot_profile


class GreedyNativeAdapter(NativeReplanner):
    """受信任 native adapter：包装项目 GreedyReplanner（无 repair 层）。"""

    method_name = 'native-greedy-edd'
    method_revision = '1'
    adapter_revision = '1'

    def __init__(self):
        self.inner = GreedyReplanner(incumbent_builder='edd')
        self.checkpoint_hash = 'none'

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                        replan_ids=replan_ids)


def _bridge_env(ds, adapter):
    env = helpers.build_env(ds, 50, 4, None)
    vehicles = [VehicleState(vehicle_id=i) for i in range(4)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    bridge = NativeBridgeReplanner(adapter)
    return env, vehicles, served, bridge


def _expect_violation(adapter, label):
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    env, vehicles, served, bridge = _bridge_env(ds, adapter)
    try:
        bridge.plan(env, 0, 0.0, vehicles, served, [1, 2, 3, 4], replan_ids={0})
    except ContractViolation as e:
        assert '环境数据' in str(e), f'{label}: 拒绝原因错误: {e}'
        print(f'  {label} 被拒绝: {str(e)[:70]}')
        return
    raise AssertionError(f'{label} 未被拒绝（环境被改写）')


def test_env_mutations_rejected():
    class MutateCoords(GreedyNativeAdapter):
        def plan(self, env, inst_idx, *a, **k):
            env.coords[inst_idx, 1, 0] = 0.0
            super().plan(env, inst_idx, *a, **k)

    class MutateDistMat(GreedyNativeAdapter):
        def plan(self, env, inst_idx, *a, **k):
            env.dist_mat[inst_idx, 0, 1] = 0.0
            super().plan(env, inst_idx, *a, **k)

    class MutateRevealTime(GreedyNativeAdapter):
        def plan(self, env, inst_idx, *a, **k):
            env.reveal_time[inst_idx, 5] = 0.0
            super().plan(env, inst_idx, *a, **k)

    class MutateCapacity(GreedyNativeAdapter):
        def plan(self, env, inst_idx, *a, **k):
            env.capacity = 1.0
            super().plan(env, inst_idx, *a, **k)

    class MutateTwEnd(GreedyNativeAdapter):
        def plan(self, env, inst_idx, *a, **k):
            env.tw_end[inst_idx, 1] = 999.0
            super().plan(env, inst_idx, *a, **k)

    _expect_violation(MutateCoords(), '改写 coords')
    _expect_violation(MutateDistMat(), '改写 dist_mat')
    _expect_violation(MutateRevealTime(), '改写 reveal_time')
    _expect_violation(MutateCapacity(), '改写 capacity')
    _expect_violation(MutateTwEnd(), '改写 tw_end')


def test_no_repair_layer_boundary():
    """无 repair 层的 native adapter：bridge 不暴露 repair_stats，
    _eval 不崩溃，repair_applicable=False。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 5.0}, seed=2)
    rec = run_instance(ds, 50, 4, adapter_factory=GreedyNativeAdapter, inst_idx=0,
                       objective='coldchain', profile=default_pilot_profile(),
                       seed=0, data_sha256='t', instance_seed=0)
    assert rec['outcome']['repair_applicable'] is False
    assert rec['outcome']['audit_source'] == 'common_runner'
    assert 'repair_ownership_violations' not in rec['outcome']
    assert rec['outcome']['complete'], '受信任 native adapter 应 complete'
    assert rec['audit']['ownership_violations'] == 0
    print('  无 repair 层 native adapter：_eval 不崩溃，repair_applicable=False，'
          '终局 complete')


def test_trusted_native_full_run():
    """受信任 native adapter 经完整 runner：写保护全程无违规（记录校验通过）。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 6.0}, seed=3)
    rec = run_instance(ds, 50, 4, adapter_factory=GreedyNativeAdapter, inst_idx=0,
                       objective='coldchain', profile=default_pilot_profile(),
                       seed=0, data_sha256='t', instance_seed=0)
    assert rec['protocol']['error'] is None
    assert all(rec['hard_vector'].values())
    print('  受信任 native adapter：完整 runner 记录通过校验，hard vector 全过')


def main():
    test_env_mutations_rejected()
    test_no_repair_layer_boundary()
    test_trusted_native_full_run()
    print('PASS test_native_env_protection')


if __name__ == '__main__':
    main()
