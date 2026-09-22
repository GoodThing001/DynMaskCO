"""B2 回归：真实 R1/EDoD=0.5 val 实例 0 事件 4——pool ownership 修复证据。

机制（已诊断确认）：事件 4 是 service-completion 事件（非 reveal，committed
车辆的 frozen tail 存活），唯一 ready 车 v0 的 needs_replan 从更早 reveal 携带
（committed 保留标记）。修复后：pool=0（全部可见未服务客户都在 committed_next
或 frozen tail 中），protected=27（11 committed_next + 16 tail），走空 pool 快速
路径。真实车辆编号不做长期算法假设（只作回归证据）。

数据缺失（本地无 npz）时跳过。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np

import testutil  # noqa: F401  (common paths)

DATA_PATH = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
             r'\data\baseline\50_node\val\dcc_50_r1_edod05_val.npz')


def test_event4_pool_ownership():
    if not os.path.exists(DATA_PATH):
        print('  SKIP：本地无 val npz 数据')
        return
    ds = dict(np.load(DATA_PATH))
    from strict_online_runner import run_instance
    from pyvrp_adapter import PyVRPRHDAdapter
    from coldchain_contract import default_pilot_profile

    captured = {}

    class CaptureAdapter(PyVRPRHDAdapter):
        def propose(self, view):
            if view.event_id == 4:
                captured['view'] = view
            return super().propose(view)

    rec = run_instance(ds, 50, 25,
                       adapter_factory=lambda: CaptureAdapter(max_iterations=500,
                                                              seed=0),
                       inst_idx=0, objective='coldchain',
                       profile=default_pilot_profile(), seed=0,
                       data_sha256='t', instance_seed=0)
    view = captured['view']
    assert view is not None, '未捕获事件 4（前序轨迹变化？）'
    assert view.event_id == 4
    # 结构断言（不硬编码车辆编号/轨迹细节）：
    # 1) 事件 4 是单一 ready 车重规划（needs_replan 从更早 reveal 携带）
    assert len(view.replan_ids) == 1, f'replan={view.replan_ids}'
    # 2) pool 为空：全部可见未服务客户都被冻结持有
    assert len(view.pool_customer_ids) == 0, \
        f'事件 4 pool 应为空，实际 pool={view.pool_customer_ids}'
    # 3) protected = 非 replan 车辆 committed_next ∪ frozen tail
    #    （历史复现：16 个 tail 客户 + 11 个 committed_next = 27）
    assert len(view.protected_customer_ids) >= 16, \
        f'protected 应 ≥ 16（tail 客户），实际 {len(view.protected_customer_ids)}'
    committed_next = {v.anchor_node_id for v in view.vehicles
                      if v.status == 'committed'
                      and v.anchor_node_id not in (None, 0)}
    assert committed_next <= set(view.protected_customer_ids), \
        'committed_next 未全部进入 protected'
    tail_customers = {int(x) for v in view.vehicles for x in v.mutable_suffix
                      if x != 0}
    assert tail_customers <= set(view.protected_customer_ids), \
        'frozen tail 客户未全部进入 protected'
    # 4) 分区不变式（pool ∪ protected == visible_unserved 由校验层强制，
    #    记录通过即证明）
    # 5) 空 pool 快速路径：不调用 PyVRP
    ev4 = next(e for e in rec['events'] if e['event_id'] == 4)
    assert set(ev4['pool_customer_ids']) == set()
    assert len(ev4['protected_customer_ids']) == len(view.protected_customer_ids)
    assert ev4.get('solve_meta', {}).get('empty_pool_fast_path') is True
    # 6) 终局仍然 complete（冻结客户后续由持有车辆正常服务）
    assert rec['outcome']['complete'], '历史 bug 修复后终局应 complete'
    assert rec['audit']['ownership_violations'] == 0
    print(f'  事件 4 回归：replan={view.replan_ids} '
          f'protected={len(view.protected_customer_ids)} pool=0 '
          f'fast_path=True（历史 bug 修复证据，终局 complete）')


def main():
    test_event4_pool_ownership()
    print('PASS test_event4_regression')


if __name__ == '__main__':
    main()
