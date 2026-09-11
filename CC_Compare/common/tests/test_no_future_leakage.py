"""B1 Gate 测试 2：future perturbation 不改变当前动作。

三种扰动（只动未来客户，不动 t=0 可见客户）：
  P1 内容扰动：改未来客户坐标/TW/需求；
  P2 身份扰动：交换两个未来客户的编号内容；
  P3 数量扰动：增加一个额外未来客户。

断言：t=0 决策点的 COMMIT 动作序列、fleet plan hash、终局 hard vector 与
扰动前完全一致（当前动作不变 = non-anticipatory）。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import numpy as np

import helpers
from strict_online_runner import run_instance
from coldchain_contract import default_pilot_profile


def _run(ds):
    return run_instance(ds, capacity=50, num_vehicles=4,
                        adapter_factory=helpers.GreedyEDDAdapter, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=0, data_sha256='t', instance_seed=0)


def _prefix_signature(rec):
    """扰动客户 reveal（5.0）之前的完整决策序列（B1.2 全 prefix parity）。"""
    return helpers.prefix_signature(rec, until_clock=5.0)


def test_structural_isolation():
    """结构保证：DecisionView 不含未来客户任何信息。

    直接构造 view 并断言：node_ids == depot + 车辆 anchor + 可见未服务；
    未来客户（reveal>0）的 id 不在 node_ids、特征不在任何数组；view 不持有
    env/dataset/VehicleState 引用（proposal 模式无对象通道）。
    """
    from strict_online_env import VehicleState
    from method_adapter import build_decision_view
    import numpy as np

    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    env = helpers.build_env(ds, 50, 4, None)
    vehicles = [VehicleState(vehicle_id=i) for i in range(4)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    visible = [1, 2, 3, 4]
    view = build_decision_view(env, 0, 0.0, vehicles, served, visible, {0, 1, 2, 3})

    assert set(view.node_ids) == {0, 1, 2, 3, 4}, \
        f'view 节点泄露未来信息: {view.node_ids}'
    assert sum(view.pool_mask) == 4
    # 未来客户特征结构不可达
    for arr in (view.coords, view.demands, view.tw_start, view.tw_end,
                view.service_time, view.dist_mat, view.travel_mat,
                view.temp_class, view.initial_quality):
        assert len(arr) == len(view.node_ids), \
            f'view 数组长度 {len(arr)} != 公开节点数 {len(view.node_ids)}'
    assert not hasattr(view, 'env') and not hasattr(view, 'dataset')
    # node_index 对非公开节点抛 ContractViolation（结构不可达）
    from method_adapter import ContractViolation
    try:
        view.node_index(5)
        raise AssertionError('view.node_index(未来客户) 未拒绝')
    except ContractViolation:
        pass
    print('  结构隔离：view 节点集合/数组长度/对象通道/索引 全部封闭')


def test_future_content_perturbation():
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    base = _run(ds)
    sig_base = _prefix_signature(base)

    ds2 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    rng = np.random.RandomState(123)
    ds2['coords'][0, 5] = rng.uniform(0.1, 0.9, 2)
    ds2['coords'][0, 6] = rng.uniform(0.1, 0.9, 2)
    ds2['tw_end'][0, 5] = 7.0
    ds2['tw_end'][0, 6] = 8.5
    ds2['demands'][0, 5] = 42.0
    ds2['demands'][0, 6] = 3.0
    pert = _run(ds2)
    sig_pert = _prefix_signature(pert)

    assert sig_pert == sig_base, (
        f'future 内容扰动改变了 t=0 动作：\nbase={sig_base}\npert={sig_pert}')
    assert pert['hard_vector'] == base['hard_vector']
    print('  P1 内容扰动全 prefix parity 通过')


def test_future_identity_perturbation():
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    base = _run(ds)
    sig_base = _prefix_signature(base)

    ds2 = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    for key in ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
                'temp_class', 'initial_quality'):
        ds2[key][0, 5], ds2[key][0, 6] = ds2[key][0, 6].copy(), ds2[key][0, 5].copy()
    pert = _run(ds2)
    sig_pert = _prefix_signature(pert)

    assert sig_pert == sig_base, (
        f'future 身份扰动改变了 t=0 动作：\nbase={sig_base}\npert={sig_pert}')
    assert pert['hard_vector'] == base['hard_vector']
    print('  P2 身份扰动全 prefix parity 通过')


def test_future_count_perturbation():
    ds = helpers.make_synthetic_dataset(n_customers=6, reveal_spec={5: 5.0, 6: 5.0},
                                        seed=0)
    base = _run(ds)
    sig_base = _prefix_signature(base)

    ds2 = helpers.make_synthetic_dataset(n_customers=7, reveal_spec={5: 5.0, 6: 5.0,
                                                                    7: 6.0}, seed=0)
    pert = _run(ds2)
    sig_pert = _prefix_signature(pert)

    assert sig_pert == sig_base, (
        f'future 数量扰动改变了 t=0 动作：\nbase={sig_base}\npert={sig_pert}')
    print('  P3 数量扰动全 prefix parity 通过')


def main():
    test_structural_isolation()
    test_future_content_perturbation()
    test_future_identity_perturbation()
    test_future_count_perturbation()
    print('PASS test_no_future_leakage')


if __name__ == '__main__':
    main()
