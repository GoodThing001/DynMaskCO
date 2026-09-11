"""OR-Tools-RH-D 评估入口：common run_batch 薄封装（协议开发期）。

用法：
    python dcc_vrp/run_dcc.py --data <dcc npz> --capacity 50 --num_vehicles 25 \
        --objective coldchain --objective-profile <profile.json> \
        --instance-ids 0,1 --solution-limit 30 --out <dir>

确定性：固定 --solution-limit + 固定策略 + 单线程（wall-clock 仅安全上限）。
"""
import argparse
import json
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import run_batch, load_objective_profile, dataset_hash
from ortools_adapter import ORToolsRHDAdapter
import identity as ortools_identity


def main():
    parser = argparse.ArgumentParser(description='OR-Tools-RH-D strict-online baseline')
    parser.add_argument('--data', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num-vehicles', type=int, default=25)
    parser.add_argument('--objective', choices=['distance', 'coldchain'],
                        default='coldchain')
    parser.add_argument('--objective-profile', default=None)
    parser.add_argument('--max-instances', type=int, default=8)
    parser.add_argument('--instance-ids', type=str, default=None)
    parser.add_argument('--solution-limit', type=int, default=30,
                        help='固定 solution limit（确定性预算；wall-clock 仅安全上限）')
    parser.add_argument('--time-limit-s', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    env_identity = ortools_identity.check_environment()
    print(f"[env] ortools={env_identity['ortools_version']} "
          f"py={env_identity['python_version'].split()[0]}", flush=True)

    dataset = dict(np.load(args.data))
    actual_n = dataset['coords'].shape[0]
    if args.instance_ids is not None:
        idxs = [int(x) for x in args.instance_ids.split(',') if x.strip()]
    else:
        idxs = list(range(min(args.max_instances, actual_n)))

    profile = None
    if args.objective == 'coldchain':
        profile = load_objective_profile(args.objective_profile)

    adapter_module = os.path.abspath(os.path.join(_DCC_VRP, 'ortools_adapter.py'))

    def factory():
        return ORToolsRHDAdapter(solution_limit=args.solution_limit,
                                 time_limit_s=args.time_limit_s, check_env=True)

    records, summary = run_batch(
        dataset, args.capacity, args.num_vehicles, factory, idxs,
        objective=args.objective, profile=profile, seed=args.seed,
        adapter_module_path=adapter_module,
        data_sha256=dataset_hash(dataset), out_dir=args.out)

    print(f"[summary] n={summary['n']} complete={summary['n_complete']}"
          f"/{summary['n']} service_ok={summary['n_service_ok']} "
          f"cost_mean={summary['cost_mean']}", flush=True)
    for rec in records:
        print(f"  inst {rec['instance_id']}: complete={rec['outcome']['complete']} "
              f"viol={rec['audit']['ownership_violations']} "
              f"fallback_ev={rec['stats']['fallback_triggered_events']}",
              flush=True)

    with open(os.path.join(args.out, 'ortools_environment.json'), 'w',
              encoding='utf-8') as f:
        json.dump(ortools_identity.environment_identity(include_freeze=True), f,
                  indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
