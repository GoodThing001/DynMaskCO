"""PyVRP-RH-D 评估入口：common run_batch 的薄封装。

用法：
    python dcc_vrp/run_dcc.py --data <dcc npz> --capacity 50 --num_vehicles 25 \
        --objective coldchain --objective-profile <profile.json> \
        --instance-ids 0,1 --max-iterations 1000 --pyvrp-seed 0 \
        --out <dir> [--workers 1]

确定性协议：固定 --pyvrp-seed + --max-iterations + 无 warm start；
同 seed 两次运行 decision_hash 一致。固定 MaxRuntime 实验另立协议。
"""
import argparse
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import (run_batch, run_instance, load_objective_profile,
                                  dataset_hash, code_hash)
from pyvrp_adapter import PyVRPRHDAdapter
from route_mapper import MappingError
import identity as pyvrp_identity


def main():
    parser = argparse.ArgumentParser(description='PyVRP-RH-D strict-online baseline')
    parser.add_argument('--data', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--objective', choices=['distance', 'coldchain'],
                        default='coldchain')
    parser.add_argument('--objective-profile', default=None)
    parser.add_argument('--max-instances', type=int, default=8)
    parser.add_argument('--instance-ids', type=str, default=None)
    parser.add_argument('--max-iterations', type=int, default=1000,
                        help='固定 MaxIterations（确定性协议）')
    parser.add_argument('--pyvrp-seed', type=int, default=0,
                        help='固定 PyVRP solver seed')
    parser.add_argument('--seed', type=int, default=0, help='run 随机种子')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    # 环境身份强制检查（版本 0.14.0 + site-packages 导入路径）
    env_identity = pyvrp_identity.check_environment()
    print(f"[env] pyvrp={env_identity['pyvrp_version']} "
          f"native={env_identity['_pyvrp_sha256'][:16]} "
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

    adapter_module = os.path.abspath(os.path.join(_DCC_VRP, 'pyvrp_adapter.py'))

    def factory():
        return PyVRPRHDAdapter(max_iterations=args.max_iterations,
                               seed=args.pyvrp_seed, check_env=True)

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
              f"unresolved={rec['audit']['terminal_unresolved']}", flush=True)

    # 环境身份落盘（manifest 补充）
    import json
    with open(os.path.join(args.out, 'pyvrp_environment.json'), 'w',
              encoding='utf-8') as f:
        json.dump(pyvrp_identity.environment_identity(include_freeze=True), f,
                  indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
