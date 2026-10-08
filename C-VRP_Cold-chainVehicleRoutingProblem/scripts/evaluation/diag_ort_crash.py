# -*- coding: utf-8 -*-
"""OR-Tools 接单臂执行期容量崩溃定位：1 天复现并 dump 崩溃时刻状态。

2026-09-28 v2：崩溃发生在 on_reveal → certify_plan（严格在线环境 strict_online_env.py:539），
不在 _advance_fleet。因此额外钩住 solver_accept_replanner.certify_plan：
在真实 C0 认证模拟 raise ValueError 的瞬间，dump 计划、每车清单载荷、
每条路线「真实载荷 vs 求解器整数载荷」逐单对比（验证 floor 需求低估假设）。
"""
import os
import sys
import traceback

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import make_c0_contract_v2
import solver_accept_replanner as _sar
from solver_accept_replanner import SolverAcceptReplanner
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926


def _load_budget():
    """预算 B/cooling_share 从 C1 门结果直接读取（与外部驱动同口径，免 200 天重算）。"""
    import json
    gate_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'results',
        'a1_step2_gate_c1_20260926', 'gate.json')
    if os.path.exists(gate_path):
        with open(gate_path, encoding='utf-8') as f:
            d = json.load(f)
        return float(d['budget']['B']), float(d['budget']['cooling_share'])
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import build_history, build_history_times_classes
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, TRAIN_SEED,
                                                   0.6, hist_tc, k=10)
    return B, cooling_share


def main():
    solver = sys.argv[1] if len(sys.argv) > 1 else 'ortools'
    contract = make_c0_contract_v2()
    B, cooling_share = _load_budget()
    dev_ds = add_v2_initial_quality(generate_dataset(60, 200, DEV_SEED), contract)
    rp = SolverAcceptReplanner(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                               contract=contract, cooling_share=cooling_share,
                               solver=solver, time_limit=10.0, solution_limit=30)
    env = StrictOnlineEnv(dev_ds, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)

    # 钩住 transition 路径上的 advance（若崩溃在 execute 阶段）
    orig_advance = StrictOnlineEnv._advance_fleet

    def hooked(self, inst_idx, clock, vehicles, traces, served_mask):
        try:
            return orig_advance(self, inst_idx, clock, vehicles, traces, served_mask)
        except ValueError as e:
            print("=== CRASH dump (advance_fleet) ===")
            print("inst_idx", inst_idx, "clock", clock, "err", e)
            for v in vehicles:
                cc = v.coldchain_state
                if cc is None:
                    continue
                man = [lot.quantity for lot in cc.cargo_manifest]
                print(f"  v{v.vehicle_id} status={v.status} cur={v.current_node} "
                      f"load={v.current_load} committed_next={v.committed_next} "
                      f"manifest={man} sum={sum(man):.6f}")
            print("  replanner plan:", {k: v2 for k, v2 in rp._plan.items()})
            print("  accepted:", sorted(rp._accepted))
            raise

    StrictOnlineEnv._advance_fleet = hooked

    # 钩住 certify_plan：崩溃点（on_reveal → certify_plan → transition_segment）
    orig_certify = _sar.certify_plan
    cap_int = int((50.0 - 1e-4) * _sar.INT_SCALE)

    # 钩住求解器：对每个返回计划做求解器级容量审计（pin + Σ ceil(demand) ≤ cap_int?）
    orig_solve = rp._solve_ortools if solver == 'ortools' else rp._solve_pyvrp

    def solve_hooked(env2, inst_idx2, clock2, vehicles2, pool2, t02):
        sts2, clients2, _dd = rp._collect(env2, inst_idx2, clock2, vehicles2, pool2)
        plan2, reason2 = orig_solve(env2, inst_idx2, clock2, vehicles2, pool2, t02)
        if plan2 is not None:
            for vid2, seq2 in plan2.items():
                pin2 = min(_sar.ceil_int(sts2[vid2]['load']), cap_int)
                seen2 = pin2 + sum(_sar._demand_int(env2.demands[inst_idx2, o])
                                   for o in seq2)
                if seen2 > cap_int:
                    vv2 = next(v for v in vehicles2 if v.vehicle_id == vid2)
                    print(f"SOLVER-RETURNED-OVER-CAPACITY: vid={vid2} load={sts2[vid2]['load']:.6f} "
                          f"pin={pin2} seq={seq2} seen={seen2} cap={cap_int} "
                          f"manifest={[l.quantity for l in vv2.coldchain_state.cargo_manifest]} "
                          f"committed_next={vv2.committed_next} "
                          f"status={vv2.status}")
                    # 落盘完整求解输入供独立重放
                    import json as _json
                    dump = dict(
                        inst_idx=int(inst_idx2), clock=float(clock2),
                        vids=sorted(sts2), sts={str(k): v3 for k, v3 in sts2.items()},
                        clients=[int(c) for c in clients2],
                        pool=[int(p) for p in pool2],
                        depot_deadline=float(_dd),
                        dist=[list(map(float, row)) for row in env2.dist_mat[inst_idx2]],
                        tw_start=[float(x) for x in env2.tw_start[inst_idx2]],
                        tw_end=[float(x) for x in env2.tw_end[inst_idx2]],
                        service=[float(x) for x in env2.service_time[inst_idx2]],
                        demands=[float(x) for x in env2.demands[inst_idx2]],
                        tw_speed=float(env2.tw_speed),
                        plan={str(k): [int(o) for o in v3] for k, v3 in plan2.items()},
                    )
                    _dump_path = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))), 'results',
                        'diag_solve_inputs.json')
                    with open(_dump_path, 'w', encoding='utf-8') as _f:
                        _json.dump(dump, _f, indent=2)
                    print('solve inputs dumped to', _dump_path)
        return plan2, reason2

    if solver == 'ortools':
        rp._solve_ortools = solve_hooked
    else:
        rp._solve_pyvrp = solve_hooked

    def cert_hooked(env2, inst_idx2, clock2, vehicles2, served_mask2, plan2, contract2, B2):
        try:
            return orig_certify(env2, inst_idx2, clock2, vehicles2, served_mask2,
                                plan2, contract2, B2)
        except ValueError as e:
            print("=== CERTIFY CRASH dump ===")
            print("err:", e)
            print("plan:", plan2)
            print("capacity_int =", cap_int)
            for v in vehicles2:
                cc = v.coldchain_state
                if cc is None:
                    continue
                man = [lot.quantity for lot in cc.cargo_manifest]
                route = [o for o in plan2.get(v.vehicle_id, []) if not served_mask2[o]]
                if v.status == 'committed' and v.committed_next not in (None, 0):
                    route = [int(v.committed_next)] + route   # certify 会重放 committed leg
                real_route = sum(float(env2.demands[inst_idx2, o]) for o in route)
                seen_route = sum(_sar._demand_int(env2.demands[inst_idx2, o])
                                 for o in route) / float(_sar.INT_SCALE)
                print(f"  v{v.vehicle_id} status={v.status} cur={v.current_node} "
                      f"committed_next={v.committed_next} "
                      f"manifest_sum={sum(man):.6f} route={route} "
                      f"route_real={real_route:.6f} route_seen={seen_route:.6f} "
                      f"total_real={sum(man) + real_route:.6f}")
                for o in route:
                    d = float(env2.demands[inst_idx2, o])
                    print(f"    o{o} demand_real={d:.6f} demand_int={_sar._demand_int(d)}")
            raise

    _sar.certify_plan = cert_hooked

    try:
        env.run(0)
        print("day 0 OK")
    except ValueError:
        traceback.print_exc()
        print("crashed as expected")


if __name__ == "__main__":
    main()
