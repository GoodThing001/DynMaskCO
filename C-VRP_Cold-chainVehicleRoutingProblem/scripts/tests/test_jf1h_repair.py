"""JF1-H-F 阶段 A 加固回归测试（阻塞点 1/2/3 的验证）。

验证：
  1. write-back 一致性：_repair_missing 内部 plans 与写回后的车辆状态一致（阻塞点 1）；
  2. 忠实性：_greedy_sequence_report 与 _greedy_sequence 在「无 dropped」时 route 完全一致
     （JF1-H-F 无 missing 时 == JF1-H，阻塞点 1 的 faithfulness 侧）；
  3. 硬 Gate：JF1-H-F(slack=1) 在小样本 val 上 hard 100% + ownership_violations=0
     + terminal_unresolved=0（阻塞点 2/3 的整实例验证）。

纯 NumPy（JF1-H continuation），可本地跑：python scripts/tests/test_jf1h_repair.py
产物：results/o0cc/jf1h_repair_tests.json
"""
import sys, os, json
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv, VehicleState
from jf1h_repair import JF1HRepairReplanner
from joint_fleet import JointAssignmentReplanner
from action_contract import build_vehicle_plans
from coldchain_contract import default_pilot_contract
from coldchain_evaluator import evaluate_coldchain_trace

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


class WritebackCheckReplanner(JF1HRepairReplanner):
    """每次 _repair_missing 后重建 plans，校验写回与内部 plans 一致（阻塞点 1）。"""

    def __init__(self):
        super().__init__(slack_vehicles=1)
        self.writeback_violations = []

    def _repair_missing(self, env, inst_idx, clock, vehicles, served_mask, missing, has_future,
                        mutable_ids):
        nonmutable_before = {v.vehicle_id: list(v.mutable_suffix)
                             for v in vehicles if v.vehicle_id not in mutable_ids}
        status = super()._repair_missing(env, inst_idx, clock, vehicles, served_mask, missing,
                                         has_future, mutable_ids)
        # 1. 非 mutable 车前后必须完全不变（跨范围 phantom insertion 会在这里暴露）
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                if list(v.mutable_suffix) != nonmutable_before.get(v.vehicle_id):
                    self.writeback_violations.append(
                        ('nonmutable', float(clock), int(v.vehicle_id),
                         nonmutable_before.get(v.vehicle_id), list(v.mutable_suffix)))
        # 2. mutable 车写回与内部 plans 一致
        rebuilt = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in mutable_ids:
            expected = self._last_repair_plans.get(vid)
            got = rebuilt.get(vid)
            if expected is None and got is None:
                continue
            if expected is None or got is None or expected.suffix != got.suffix:
                self.writeback_violations.append(
                    (float(clock), int(vid),
                     expected.suffix if expected else None,
                     got.suffix if got else None))
        return status


def test_writeback_consistency(dataset, contract):
    rp = WritebackCheckReplanner()
    env = StrictOnlineEnv(dataset, 50, 1.0, 25, replanner=rp, coldchain_contract=contract)
    n = min(8, dataset['coords'].shape[0])
    for i in range(n):
        env.run(i)
    ok = (len(rp.writeback_violations) == 0)
    record('writeback_consistency', ok,
           f'{len(rp.writeback_violations)} violations over {n} instances')
    return ok


def test_greedy_report_equivalence(dataset, contract):
    """_greedy_sequence_report 的 route 与 _greedy_sequence 完全一致，且 dropped = assigned−route。"""
    rep = JF1HRepairReplanner(slack_vehicles=0)
    base = JointAssignmentReplanner(score_mode='heuristic')
    env = StrictOnlineEnv(dataset, 50, 1.0, 25, replanner=None, coldchain_contract=contract)
    n = min(8, dataset['coords'].shape[0])
    checked = 0
    for i in range(n):
        v = VehicleState(vehicle_id=0, status='idle', current_node=0, ready_time=0.0,
                         current_load=0.0)
        assigned = [c for c in range(1, env.num_nodes)
                    if env.demands[i, c] > 0 and env.reveal_time[i, c] <= 1e-6][:6]
        r_rep, dropped = rep._greedy_sequence_report(env, i, v, assigned)
        r_base = base._greedy_sequence(env, i, v, assigned)
        route_customers = set(c for c in r_rep if c != 0)
        ok_route = (r_rep == r_base)
        ok_dropped = (set(dropped) == (set(assigned) - route_customers))
        checked += 1
        if not (ok_route and ok_dropped):
            record('greedy_report_equivalence', False,
                   f'inst {i}: route_eq={ok_route} dropped_eq={ok_dropped} '
                   f'rep={r_rep} base={r_base} dropped={sorted(dropped)}')
            return False
    record('greedy_report_equivalence', True, f'{checked} events: route==base && dropped==assigned minus route')
    return True


def test_hard_gate(dataset, contract):
    rp = JF1HRepairReplanner(slack_vehicles=1)
    env = StrictOnlineEnv(dataset, 50, 1.0, 25, replanner=rp, coldchain_contract=contract)
    n = min(8, dataset['coords'].shape[0])
    n_unserved = 0
    n_hard = 0
    n_own = 0
    n_term = 0
    for i in range(n):
        traces, served_mask = env.run(i)
        out = evaluate_coldchain_trace(traces, {
            'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
            'service_time': env.service_time[i], 'demands': env.demands[i],
            'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
            'initial_quality': env.initial_quality[i], 'speed': env.tw_speed}, contract)
        n_unserved += sum(1 for c in range(1, env.num_nodes)
                          if env.demands[i, c] > 0 and not served_mask[c])
        n_hard += not (out['complete'] and out['temperature_hard_feasible']
                       and out['trace_accounting_consistent'] and out['tw_feasible']
                       and out['capacity_feasible'] and out['depot_return_feasible']
                       and out['all_orders_picked'] and out['all_cargo_delivered_to_depot']
                       and out['terminal_manifests_empty']
                       and out['distance_accounting_consistent'])
        n_own += rp.repair_stats['ownership_violations']
        n_term += sum(1 for c in rp.deferred_customers if not served_mask[int(c)])
    ok = (n_unserved == 0 and n_hard == 0 and n_own == 0 and n_term == 0)
    record('hard_gate', ok,
           f'unserved={n_unserved} hard={n_hard} ownership={n_own} terminal={n_term}')
    return ok


def main():
    data_path = os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'val',
                             'dcc_50_r1_edod05_val.npz')
    dataset = dict(np.load(data_path))
    contract = default_pilot_contract()

    ok1 = test_writeback_consistency(dataset, contract)
    ok2 = test_greedy_report_equivalence(dataset, contract)
    ok3 = test_hard_gate(dataset, contract)

    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'jf1h_repair_tests.json'), 'w') as f:
        json.dump({'all_pass': ok1 and ok2 and ok3, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == '__main__':
    sys.exit(main())
