"""通用 hard-gate 检查：对任意数据跑 JF1-H-F baseline，报告 hard-pass / ownership / terminal。

用于 DEV-PROTO / DEV-GATE 的基线 hard-feasibility 前置校验（不限 128 实例）。

用法：python scripts/evaluation/check_hard_gate.py --data <npz...> --capacity 50 --num_vehicles 25
"""
import argparse, os, sys, json
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
from jf1h_repair import make_continuation
from coldchain_contract import default_pilot_contract
from coldchain_evaluator import evaluate_coldchain_trace
from hard_gate import hard_vector_from_outcome


def check(data_path, capacity, num_vehicles, inst_out_dir):
    dataset = dict(np.load(data_path))
    contract = default_pilot_contract()
    rp = make_continuation()
    env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles, replanner=rp,
                          coldchain_contract=contract)
    n = dataset['coords'].shape[0]
    cell = os.path.splitext(os.path.basename(data_path))[0]
    hard_fail = 0
    term = 0
    own = 0
    per_instance = []
    for i in range(n):
        traces, served = env.run(i)
        out = evaluate_coldchain_trace(traces, {
            'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
            'service_time': env.service_time[i], 'demands': env.demands[i],
            'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
            'initial_quality': env.initial_quality[i], 'speed': env.tw_speed}, contract)
        hv = hard_vector_from_outcome(out)
        inst_term = sum(1 for c in rp.deferred_customers if not served[int(c)])
        inst_own = rp.repair_stats['ownership_violations']
        inst_pass = all(hv.values()) and inst_term == 0 and inst_own == 0
        if not inst_pass:
            hard_fail += 1
        term += inst_term
        own += inst_own
        per_instance.append({
            'cell': cell, 'instance_index': i,
            'hard_pass': inst_pass, 'hard_vector': hv,
            'terminal_unresolved': inst_term, 'ownership_violations': inst_own,
            'distance_km': out['distance_km'], 'quality_loss': out['quality_loss'],
            'energy_kwh': out['energy_kwh'],
        })
    if inst_out_dir is not None:
        os.makedirs(inst_out_dir, exist_ok=True)
        with open(os.path.join(inst_out_dir, f'{cell}.json'), 'w') as f:
            json.dump({'cell': cell, 'instances': per_instance}, f, indent=2)
    return hard_fail, term, own, n, per_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    ap.add_argument('--out', default=None, help='输出目录（保存逐实例结果 + manifest）')
    args = ap.parse_args()
    total = [0, 0, 0, 0]
    all_per = []
    for d in args.data:
        inst_dir = os.path.join(args.out, 'instances') if args.out else None
        hf, term, own, n, per = check(d, args.capacity, args.num_vehicles, inst_dir)
        total[0] += hf; total[1] += term; total[2] += own; total[3] += n
        all_per.extend(per)
        print(f"  {os.path.basename(d)}: hard_fail={hf}/{n} terminal={term} ownership={own}")
    ok = (total[0] == 0 and total[1] == 0 and total[2] == 0)
    print(f"  TOTAL: hard_fail={total[0]} terminal={total[1]} ownership={total[2]} n={total[3]}")
    print(f"  {'ALL HARD-PASS' if ok else 'FAIL'}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'hard_gate.json'), 'w') as f:
            json.dump({'all_hard_pass': ok, 'n_hard_fail': total[0],
                       'n_terminal_unresolved': total[1],
                       'n_ownership_violations': total[2], 'n': total[3]}, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
