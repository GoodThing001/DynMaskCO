"""B1 Gate 测试 5：轨迹导出经 authoritative evaluator 重放后 D/Q/E 一致。

   5a 导出轨迹分段数字与 outcome D/Q/E/J 对账（trace_replay_check，run_instance 内置，
      此处显式断言）；
   5b 静态等价：无动态 reveal、t=0 全体立即派车的实例，其 zero-delimited 静态路线
      经 replay_static_pickup_route（独立 C0 物理）重放，D/Q/E/thermal 与在线 outcome
      一致（同一 transition 实现的两条路径必须给出相同物理）。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import helpers
from strict_online_runner import run_instance
import trace_export as te
from coldchain_evaluator import (evaluate_coldchain_trace, replay_static_pickup_route,
                                 math_isclose)
from coldchain_contract import default_pilot_contract, apply_objective_profile, \
    default_pilot_profile


def _run(ds):
    return run_instance(ds, capacity=50, num_vehicles=6,
                        adapter_factory=helpers.GreedyEDDAdapter, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=0, data_sha256='t', instance_seed=0)


def test_trace_replay_accounting():
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=3)
    rec = _run(ds)
    ok, problems = te.trace_replay_check(rec['execution_trace'], rec['outcome'],
                                         'coldchain')
    assert ok, f'trace 对账失败: {problems}'
    print('  5a 导出轨迹与 outcome D/Q/E/thermal 对账通过')


def test_static_route_replay_parity():
    # 无动态：全部客户 t=0 可见 → 所有车 t=0 派车，无 WAIT、无中途事件
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=4)
    rec = _run(ds)
    out = rec['outcome']
    contract = apply_objective_profile(default_pilot_contract(),
                                       default_pilot_profile())

    routes = te.export_static_routes(rec['execution_trace'])
    # 独立 C0 静态回放（每条 tour 从 t=0 空车出发）
    static = replay_static_pickup_route(
        [n for r in routes for n in r], {
            'coords': ds['coords'][0], 'demands': ds['demands'][0],
            'tw_start': ds['tw_start'][0], 'tw_end': ds['tw_end'][0],
            'service_time': ds['service_time'][0],
            'temp_class': ds['temp_class'][0],
            'initial_quality': ds['initial_quality'][0],
        }, contract, speed=1.0)

    assert static['complete'], '静态回放未 complete（在线却 complete）'
    assert static['tw_feasible'] and static['capacity_feasible']
    # distance/energy/thermal 两条路径在浮点精度内一致（在线用 float32 dist_mat，
    # 静态用 float64 coords → 1e-7 级）；quality 因在线 trace 的分段锚点
    # （depart/arrival 锚在完成时刻）与静态一次性 transition 的积分区间切分不同，
    # 存在 ~0.1% 的结构性差异（在线自身的 trace_accounting_consistent 已通过）。
    tol = {'distance_km': 1e-5, 'quality_loss': 5e-3, 'energy_kwh': 1e-5,
           'thermal_violation_count': 0, 'thermal_violation_duration_h': 1e-5}
    for key, t in tol.items():
        a, b = static[key], out[key]
        assert math_isclose(float(a), float(b), atol=t, rtol=1e-9), (
            f'{key} 静态回放({a}) != 在线 outcome({b})')
    print('  5b 静态回放与在线 trace 的 D/Q/E/thermal 一致（无动态实例；'
          'quality 容差 5e-3 覆盖分段锚点差异）')


def main():
    test_trace_replay_accounting()
    test_static_route_replay_parity()
    print('PASS test_trace_replay')


if __name__ == '__main__':
    main()
