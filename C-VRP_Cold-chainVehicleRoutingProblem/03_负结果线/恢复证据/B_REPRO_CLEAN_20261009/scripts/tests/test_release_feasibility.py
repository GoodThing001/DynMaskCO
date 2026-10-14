"""
P0-D1: Release Feasibility Test（强制测试 #1）

验证动态订单的揭示时间满足「必要可服务条件」：
  订单被揭示后，至少存在一条从 depot 直达的服务路径，即
      r_i + τ_i <= b_i    （τ_i = dist(depot, i) / speed，乐观旅行时间下界）

旧数据（42% 未来订单 reveal_time > tw_end）违反该条件，导致 non-anticipatory
policy 结构上无法服务这些订单 —— 是 generator bug，不是自然困难。

本测试对单个 .npz（或 --data_dir 下所有 .npz）检查：
  1. 必要可服务：r_i + τ_i <= b_i（对所有动态客户）
  2. generator 不变式：r_i + τ_i + s_i <= b_i（当前生成器采用的更保守条件）

用法:
    python scripts/tests/test_release_feasibility.py --data <dcc.npz>
    python scripts/tests/test_release_feasibility.py --data_dir data/baseline/50_node/test
"""

import sys, os, argparse, glob
import numpy as np


def check_release_feasibility(data_path, speed=1.0, tol=1e-6, verbose=True):
    """检查单个数据文件的 release feasibility。返回 (ok, stats_dict)。"""
    d = dict(np.load(data_path))
    coords = d['coords'].astype(np.float32)
    tw_end = d['tw_end'].astype(np.float32)
    service_time = d.get('service_time',
                         np.zeros_like(tw_end, dtype=np.float32))
    reveal_time = d.get('reveal_time',
                        np.zeros_like(tw_end, dtype=np.float32))

    # 距离 depot（node 0）的欧氏距离，speed=1.0（与生成器/解码器一致）
    diff = coords - coords[:, 0:1, :]
    dist_to_depot = np.sqrt((diff ** 2).sum(axis=-1))  # (N, nodes)

    dynamic = reveal_time > 0
    n_dynamic = int(dynamic.sum())

    # 条件 1（弱）：揭示时时间窗尚未关闭
    bad_weak = (reveal_time > tw_end) & dynamic

    # 条件 2（必要可服务）：r + τ <= b，即揭示后至少能直达
    tau = dist_to_depot / speed
    bad_service = (reveal_time + tau > tw_end + tol) & dynamic

    # 条件 3（generator 不变式）：r + τ + s <= b（更保守）
    bad_gen = (reveal_time + tau + service_time > tw_end + tol) & dynamic

    ok = (n_dynamic == 0) or (int(bad_service.sum()) == 0)

    if verbose:
        name = os.path.basename(data_path)
        print(f"=== Release Feasibility: {name} ===")
        print(f"  dynamic customers:  {n_dynamic}")
        print(f"  weak violation     (r > b):                {int(bad_weak.sum())}")
        print(f"  service violation  (r + τ > b):           {int(bad_service.sum())}")
        print(f"  generator invariant (r + τ + s > b):      {int(bad_gen.sum())}")
        if n_dynamic > 0 and int(bad_service.sum()) > 0:
            # 举例最差 3 个
            idx = np.argwhere(bad_service)
            print("  worst cases (inst, node, r, τ, s, b):")
            for (bi, ni) in idx[:3]:
                print(f"    inst={bi} node={ni} r={reveal_time[bi, ni]:.2f} "
                      f"τ={tau[bi, ni]:.2f} s={service_time[bi, ni]:.2f} "
                      f"b={tw_end[bi, ni]:.2f}")
        print(f"  => {'PASS' if ok else 'FAIL'}")

    stats = {
        'n_dynamic': n_dynamic,
        'n_weak_bad': int(bad_weak.sum()),
        'n_service_bad': int(bad_service.sum()),
        'n_gen_bad': int(bad_gen.sum()),
    }
    return ok, stats


def main():
    parser = argparse.ArgumentParser(description='P0-D1: Release Feasibility Test')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()

    if args.data is None and args.data_dir is None:
        parser.error('必须提供 --data 或 --data_dir')

    if args.data_dir is not None:
        files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
        if not files:
            print(f"No .npz files under {args.data_dir}")
            sys.exit(1)
    else:
        files = [args.data]

    all_ok = True
    for f in files:
        ok, stats = check_release_feasibility(f, speed=args.speed)
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    print(f"SUMMARY: {'ALL PASS — release feasible' if all_ok else 'FAIL — release infeasible orders exist'}")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
