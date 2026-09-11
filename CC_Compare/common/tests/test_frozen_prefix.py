"""B1 Gate 测试 3：frozen prefix / 写保护——proposal 层违规全部被拒绝。

  3a 跨车重复客户（duplicate_suffix）      → ContractViolation
  3b 同车重复客户（intra-route duplicate） → ContractViolation
  3c proposal 含非 replan 车辆键           → ContractViolation（直接 Bridge 单测）
  3d proposal 缺少 replan 车辆键           → ContractViolation（直接 Bridge 单测）
  3e proposal 含未来客户                   → ContractViolation
  3f proposal 含非法客户编号               → ContractViolation
  3g 正例：记录通过校验，writeback 全部 ok
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import numpy as np

import helpers
from strict_online_runner import run_instance
from strict_online_env import VehicleState
from method_adapter import BridgeReplanner, ContractViolation
from coldchain_contract import default_pilot_profile


def _run_with(adapter_factory, ds=None):
    ds = ds or helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    return run_instance(ds, capacity=50, num_vehicles=4,
                        adapter_factory=adapter_factory, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=0, data_sha256='t', instance_seed=0)


def _bridge_direct(adapter, replan_ids, vehicles=None):
    """直接 Bridge 单元测试：手工构造决策点（env 事件语义覆盖不到的 replan 组合）。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    env = helpers.build_env(ds, 50, 4, None)
    if vehicles is None:
        vehicles = [VehicleState(vehicle_id=i) for i in range(4)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    bridge = BridgeReplanner(adapter)
    bridge.plan(env, 0, 0.0, vehicles, served, [1, 2, 3, 4],
                replan_ids=replan_ids)
    return bridge


def test_duplicate_customer_rejected():
    try:
        _run_with(helpers.DuplicateCustomerAdapter)
    except ContractViolation as e:
        assert '重复' in str(e)
        print('  3a 跨车重复客户被拒绝:', str(e)[:80])
        return
    raise AssertionError('跨车重复客户 proposal 未被拒绝')


def test_intra_route_duplicate_rejected():
    try:
        _run_with(helpers.IntraRouteDuplicateAdapter)
    except ContractViolation as e:
        assert '重复' in str(e)
        print('  3b 同车重复客户被拒绝:', str(e)[:80])
        return
    raise AssertionError('同车重复客户 proposal 未被拒绝')


def test_non_replan_key_rejected():
    try:
        _bridge_direct(helpers.NonReplanKeyAdapter(), replan_ids={0})
    except ContractViolation as e:
        assert '键' in str(e)
        print('  3c proposal 含非 replan 车辆键被拒绝:', str(e)[:80])
        return
    raise AssertionError('非 replan 车辆键未被拒绝')


def test_missing_key_rejected():
    try:
        _bridge_direct(helpers.MissingKeyAdapter(), replan_ids={0, 1})
    except ContractViolation as e:
        assert '键' in str(e)
        print('  3d proposal 缺少 replan 车辆键被拒绝:', str(e)[:80])
        return
    raise AssertionError('缺少 replan 车辆键未被拒绝')


def test_future_customer_rejected():
    try:
        _run_with(lambda: helpers.FutureCustomerAdapter(5))
    except ContractViolation as e:
        assert '非法客户' in str(e) or '不在可变池' in str(e)
        print('  3e 未来客户入 proposal 被拒绝:', str(e)[:80])
        return
    raise AssertionError('未来客户 proposal 未被拒绝（未来泄漏）')


def test_invalid_customer_rejected():
    try:
        _run_with(helpers.InvalidCustomerAdapter)
    except ContractViolation as e:
        assert '非法客户' in str(e)
        print('  3f 非法客户编号被拒绝:', str(e)[:80])
        return
    raise AssertionError('非法客户编号未被拒绝')


def test_positive_and_committed_untouched():
    # 正例：正常 adapter 全程无违规；committed 车在 plan 前后逐字段不变
    # （Bridge 每次 plan 强制校验，记录通过校验即证明）
    ds = helpers.make_early_reveal_dataset()
    rec = _run_with(helpers.GreedyEDDAdapter, ds=ds)
    for a in rec['actions']:
        assert a.get('writeback_ok') is True
    assert rec['audit']['ownership_violations'] == 0
    print('  3g 正例：early-reveal（含 committed 决策点）通过校验，审计 0 违规')


def main():
    test_duplicate_customer_rejected()
    test_intra_route_duplicate_rejected()
    test_non_replan_key_rejected()
    test_missing_key_rejected()
    test_future_customer_rejected()
    test_invalid_customer_rejected()
    test_positive_and_committed_untouched()
    print('PASS test_frozen_prefix')


if __name__ == '__main__':
    main()
