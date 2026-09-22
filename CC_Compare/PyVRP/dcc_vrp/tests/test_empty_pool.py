"""B2 回归：空 pool 快速路径（不调用 PyVRP）+ WAIT/CLOSE 规则 + 分区审计。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view
import helpers
import numpy as np


def _adapter_view(ds, num_vehicles=4, replan=None, statuses=None, ready_times=None,
                  loads=None):
    """构造空 pool 视图：所有 t=0 可见客户已服务（pool 空），
    未来客户（reveal>0）保持未服务（has_future_reveal=True）。"""
    env = helpers.build_env(ds, 50, num_vehicles, None)
    from strict_online_env import VehicleState
    vehicles = [VehicleState(vehicle_id=i) for i in range(num_vehicles)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    for c in range(1, ds['coords'].shape[1]):
        if ds['reveal_time'][0, c] <= 0:
            served[c] = True          # 只服务已 reveal 的 → 可见未服务为空
    visible = []
    if statuses:
        for vid, s in statuses.items():
            vehicles[vid].status = s
            vehicles[vid].current_node = 1      # ready@customer
    if ready_times:
        for vid, t in ready_times.items():
            vehicles[vid].ready_time = float(t)
    if loads:
        for vid, l in loads.items():
            vehicles[vid].current_load = float(l)
    from method_adapter import build_decision_view
    view = build_decision_view(env, 0, 2.0, vehicles, served, visible,
                               set(replan or range(num_vehicles)))
    return view


def test_empty_pool_no_solve_call():
    """空 pool：不调用 PyVRP（build_model 不得被调用）。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    view = _adapter_view(ds, replan={0, 1})
    assert sum(view.pool_mask) == 0
    from pyvrp_adapter import PyVRPRHDAdapter
    adapter = PyVRPRHDAdapter(max_iterations=100, seed=0)

    import pyvrp_adapter as pa
    calls = []
    orig = pa.build_model

    def counting(view_, **kw):
        calls.append(1)
        return orig(view_, **kw)

    pa.build_model = counting
    try:
        proposal = adapter.propose(view)
    finally:
        pa.build_model = orig
    assert not calls, '空 pool 仍调用了 PyVRP'
    assert set(proposal.suffixes.keys()) == {0, 1}
    assert proposal.solve_meta['empty_pool_fast_path'] is True
    print('  空 pool 不调用 PyVRP，proposal 键 == replan_ids')


def test_wait_close_rules():
    """WAIT/CLOSE 规则（统一版）：WAIT iff 有 future 且现在返仓可行；
    其余 CLOSE。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=0)
    from pyvrp_adapter import PyVRPRHDAdapter
    adapter = PyVRPRHDAdapter(max_iterations=100, seed=0)

    # (a) 有 future，客户处 ready 且现在返仓可行 → WAIT
    view = _adapter_view(ds, replan={0}, statuses={0: 'ready'},
                         ready_times={0: 2.0})
    assert view.has_future_reveal
    prop = adapter.propose(view)
    assert prop.suffixes[0] == (), f'应 WAIT，实际 {prop.suffixes[0]}'

    # (b) 无 future reveal → CLOSE
    ds_static = helpers.make_synthetic_dataset(seed=1)   # 无 reveal_spec
    view2 = _adapter_view(ds_static, replan={0}, statuses={0: 'ready'},
                          ready_times={0: 2.0})
    assert not view2.has_future_reveal
    prop2 = adapter.propose(view2)
    assert prop2.suffixes[0] == (0,), f'应 CLOSE，实际 {prop2.suffixes[0]}'

    # (c) 有 future，depot 等待（travel=0 恒可行）→ WAIT
    view3 = _adapter_view(ds, replan={0})      # idle@depot
    assert view3.has_future_reveal
    assert view3.vehicles[0].anchor_node_id == 0
    prop3 = adapter.propose(view3)
    assert prop3.suffixes[0] == (), f'depot 应 WAIT，实际 {prop3.suffixes[0]}'

    # (d) 有 future 但现在返仓已不可行（ready 已超 horizon）→ CLOSE
    view4 = _adapter_view(ds, replan={0}, statuses={0: 'ready'},
                          ready_times={0: 12.5})
    prop4 = adapter.propose(view4)
    assert prop4.suffixes[0] == (0,), f'返仓不可行应 CLOSE，实际 {prop4.suffixes[0]}'
    print('  WAIT/CLOSE 规则：有 future 且返仓可行→WAIT（depot 恒 WAIT）；'
          '无 future 或返仓不可行→CLOSE')


def test_empty_pool_through_runner():
    """空 pool 经完整 runner：手工构造事件 4 同机制场景。

    机制：长腿车 v_c1 在 reveal 后完成 leg（needs_replan 从 reveal 携带），
    此时所有可见未服务客户都在另一辆车的 committed_next/frozen tail 中
    → pool 空 → 快速路径（无 future reveal → CLOSE）。
    """
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=0)
    # 手工几何：depot (0.1,0.1)；c1 (0.9,0.9) 长腿 ~1.13；
    # c2 (0.9,0.1)、c3 (0.1,0.9)，reveal 0.3
    ds['coords'][0, 0] = [0.1, 0.1]
    ds['coords'][0, 1] = [0.9, 0.9]
    ds['coords'][0, 2] = [0.9, 0.1]
    ds['coords'][0, 3] = [0.1, 0.9]
    ds['reveal_time'][0] = [0, 0, 0.3, 0.3]
    ds['demands'][0] = [0, 10, 10, 10]
    ds['tw_end'][0, 0] = 12.0

    from strict_online_runner import run_instance
    from pyvrp_adapter import PyVRPRHDAdapter
    from coldchain_contract import default_pilot_profile
    rec = run_instance(ds, 50, 2,
                       adapter_factory=lambda: PyVRPRHDAdapter(max_iterations=300,
                                                               seed=0),
                       inst_idx=0, objective='coldchain',
                       profile=default_pilot_profile(), seed=0,
                       data_sha256='t', instance_seed=0)
    assert rec['audit']['ownership_violations'] == 0
    fast_events = [e for e in rec['events']
                   if e.get('solve_meta') and e['solve_meta'].get('empty_pool_fast_path')]
    assert fast_events, '空 pool 事件未记录 solve_meta（快速路径未触发）'
    ev = fast_events[0]
    assert set(ev['pool_customer_ids']) == set()
    assert ev['replan_vehicle_ids'], '空 pool 事件应有 replan 车辆'
    # 无未来 reveal → CLOSE：该事件的 execution 层存在对应 CLOSE
    assert any(a['action_layer'] == 'execution' and a['action_type'] == 'CLOSE'
               and a['event_id'] == ev['event_id'] for a in rec['actions']), \
        '空 pool + 无未来 reveal 应 CLOSE'
    assert rec['outcome']['complete'], '终局应 complete'
    print(f'  runner 集成：空 pool 快速路径触发（event {ev["event_id"]}），'
          f'audit 0 违规，终局 complete')


def main():
    test_empty_pool_no_solve_call()
    test_wait_close_rules()
    test_empty_pool_through_runner()
    print('PASS test_empty_pool')


if __name__ == '__main__':
    main()
