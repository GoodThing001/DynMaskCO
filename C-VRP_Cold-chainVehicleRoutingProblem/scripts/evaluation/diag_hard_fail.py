"""诊断 hard_fail 实例：找出未服务客户 + 判断 TW 不可达 vs repair 贪心。"""
import argparse, os, sys
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from coldchain_contract import default_pilot_contract
from coldchain_evaluator import evaluate_coldchain_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    ap.add_argument('--max_fail', type=int, default=10)
    args = ap.parse_args()

    dataset = dict(np.load(args.data))
    contract = default_pilot_contract()
    n = dataset['coords'].shape[0]

    for i in range(n):
        rp = JF1HRepairReplanner()
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=rp, coldchain_contract=contract)
        traces, served_mask = env.run(i)
        out = evaluate_coldchain_trace(traces, {
            'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
            'service_time': env.service_time[i], 'demands': env.demands[i],
            'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
            'initial_quality': env.initial_quality[i], 'speed': env.tw_speed,
        }, contract)
        cond = {
            'complete': out.get('complete'), 'tw': out.get('tw_feasible'),
            'cap': out.get('capacity_feasible'), 'return': out.get('depot_return_feasible'),
            'temp': out.get('temperature_hard_feasible'),
            'acct': out.get('trace_accounting_consistent'),
        }
        failed = [k for k, v in cond.items() if v is False]
        unserved = [int(c) for c in range(1, env.num_nodes)
                    if env.demands[i, c] > 0 and not served_mask[int(c)]]
        if not failed and not unserved:
            continue
        print(f"\n=== instance {i} === failed={failed} n_unserved={len(unserved)}")
        dmat = env.dist_mat[i]
        for c in unserved:
            tw_s, tw_e = env.tw_start[i, c], env.tw_end[i, c]
            rev = env.reveal_time[i, c]
            d_depot = dmat[0, c]
            dep_arrive = rev + d_depot
            reachable = dep_arrive <= tw_e + 1e-6
            print(f"  cust {c}: demand={env.demands[i,c]:.0f} tw=[{tw_s:.1f},{tw_e:.1f}] "
                  f"reveal={rev:.1f} d_depot={d_depot:.1f} earliest_depart_arrive={dep_arrive:.1f} "
                  f"REACHABLE={reachable}")
        # 终局 deferred 未服务客户
        term = [c for c in rp.deferred_customers if not served_mask[int(c)]]
        print(f"  deferred_at_terminal={sorted(term)} "
              f"repair(fail={rp.repair_stats['failures']})")


if __name__ == '__main__':
    main()
