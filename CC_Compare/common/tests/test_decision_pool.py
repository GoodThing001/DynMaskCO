"""B1.1 P0 回归：DecisionView 可变池 ownership 语义。

  - 非 replan 车辆的 committed_next + 完整 mutable_suffix 冻结（protected）；
  - replan 车辆原有 suffix 进入 pool（可在 replan 车辆间重新分配）；
  - visible_unserved == protected ∪ pool（无交叉）；
  - future/served 客户不进 pool（结构隔离，已有测试）。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import numpy as np

import helpers
from method_adapter import build_decision_view
from strict_online_env import VehicleState


def _mk(ds, replan_ids):
    env = helpers.build_env(ds, 50, 4, None)
    vehicles = [VehicleState(vehicle_id=i) for i in range(4)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    visible = [c for c in range(1, ds['coords'].shape[1])
               if ds['demands'][0, c] > 0 and ds['reveal_time'][0, c] <= 0]
    return env, vehicles, served, visible, set(replan_ids)


def test_non_replan_tail_protected():
    """非 replan 车辆的 committed_next 与完整 suffix 全部冻结。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=0)
    env, vehicles, served, visible, replan = _mk(ds, {0})
    # 车 1 非 replan：committed 到客户 2，suffix=[3,4,0]（plan persistence 冻结）
    vehicles[1].status = 'committed'
    vehicles[1].committed_next = 2
    vehicles[1].committed_finish = 1.0
    vehicles[1].mutable_suffix = [3, 4, 0]
    # 车 2 非 replan：ready，suffix=[5,0]
    vehicles[2].status = 'ready'
    vehicles[2].current_node = 6
    vehicles[2].ready_time = 1.0
    vehicles[2].mutable_suffix = [5, 0]

    view = build_decision_view(env, 0, 1.0, vehicles, served, visible, replan)
    pool = set(view.pool_customer_ids)
    prot = set(view.protected_customer_ids)
    assert {2, 3, 4, 5} <= prot, f'非 replan 持有的客户未保护: prot={prot}'
    assert not (pool & {2, 3, 4, 5}), f'冻结客户进入 pool: {pool}'
    print('  非 replan tail 保护通过（committed_next + 完整 suffix 冻结）')


def test_replan_own_tail_replannable():
    """replan 车辆原有 suffix 进入 pool，可重新分配。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=0)
    env, vehicles, served, visible, replan = _mk(ds, {0, 1})
    # 车 0 replan：ready@客户6，suffix=[4,5,0]（旧 tail 可重分配）
    vehicles[0].status = 'ready'
    vehicles[0].current_node = 6
    vehicles[0].ready_time = 1.0
    vehicles[0].mutable_suffix = [4, 5, 0]
    # 车 1 replan：idle
    view = build_decision_view(env, 0, 1.0, vehicles, served, visible, replan)
    pool = set(view.pool_customer_ids)
    assert {4, 5} <= pool, f'replan 车辆旧 tail 应进入 pool: pool={pool}'
    # 分区不变式
    assert set(view.protected_customer_ids) & pool == set()
    assert set(view.protected_customer_ids) | pool == set(visible)
    print('  replan 旧 tail 可重分配 + 分区不变式通过')


def test_partition_invariant_random():
    """随机状态扫描：visible_unserved == protected ∪ pool 恒成立（无交叉）。"""
    rng = np.random.RandomState(42)
    for trial in range(50):
        ds = helpers.make_synthetic_dataset(n_customers=8, seed=trial)
        env, vehicles, served, visible, replan = _mk(
            ds, {int(v) for v in rng.choice(4, size=int(rng.randint(1, 5)),
                                            replace=False)})
        for v in vehicles:
            if v.vehicle_id in replan:
                v.status = rng.choice(['idle', 'ready'])
            else:
                if rng.rand() < 0.5:
                    v.status = 'committed'
                    v.committed_next = int(rng.randint(1, 9))
                    v.committed_finish = 1.0
                else:
                    v.status = 'ready'
                    v.current_node = int(rng.randint(1, 9))
            v.mutable_suffix = [int(x) for x in rng.choice(9, size=3)] + [0]
        view = build_decision_view(env, 0, 1.0, vehicles, served, visible, replan)
        pool = set(view.pool_customer_ids)
        prot = set(view.protected_customer_ids)
        vis_unserved = set(visible)
        assert pool & prot == set(), f'trial {trial}: pool/protected 交叉'
        assert pool | prot == vis_unserved, \
            f'trial {trial}: 分区不完整 missing={vis_unserved-(pool|prot)}'
    print('  随机扫描 50 例：分区不变式恒成立')


def main():
    test_non_replan_tail_protected()
    test_replan_own_tail_replannable()
    test_partition_invariant_random()
    print('PASS test_decision_pool')


if __name__ == '__main__':
    main()
