"""
P0-D1: 重新生成 baseline 数据（release-feasibility 修复版）

背景（导师 D1 决策）：
  旧 baseline 数据存在结构性 release infeasibility —— 约 42~50% 动态订单
  reveal_time > tw_end（订单揭示时时间窗已关闭），non-anticipatory policy
  结构上无法服务，这是 generator bug 而非自然困难。

  生成器 `generate_coldchain_data.py` 已修复（约束 reveal_time 满足
  r_i + τ_i + s_i <= b_i），本脚本据此重新生成 train/val/test，并：
    1. 冻结旧数据到 data/archive/baseline_v1_buggy/
    2. 重新生成 baseline_v2_strict 数据（写入 data/baseline/）
    3. 重新生成 DATA_MANIFEST.sha256

用法:
    python scripts/data/regenerate_baseline.py            # 全量（train+val+test+100-node）
    python scripts/data/regenerate_baseline.py --dry_run  # 只打印计划，不写盘
    python scripts/data/regenerate_baseline.py --skip_archive  # 不移动旧数据（已归档过）

注意：train/val/test 使用不同 seed（无实例重叠，防止 data leakage）。
"""

import sys, os, argparse, hashlib, shutil
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, os.path.join(_BASE))  # 使 generate_coldchain_data 可导入

from generate_coldchain_data import generate_dataset

# 50-node: 9 域（R1/C1/RC1 × EDoD 0.2/0.5/0.8）
TYPES = ['R1', 'C1', 'RC1']
EDODS = [0.2, 0.5, 0.8]
SPLIT_SIZES = {'train': 1280, 'val': 128, 'test': 128}
# 100-node cross-scale test: 3 域 × EDoD 0.5
TYPES_100 = ['R1', 'C1', 'RC1']


def seed_for(type_, edod, split, size=50):
    """确定性且互不重叠的 seed（train/val/test 不同，防止实例泄漏）。"""
    base = {'R1': 100000, 'C1': 200000, 'RC1': 300000}[type_]
    edod_int = {0.2: 0, 0.5: 1, 0.8: 2}[edod]
    split_off = {'train': 0, 'val': 1000, 'test': 2000}[split]
    size_off = 0 if size == 50 else 400000  # 100-node 独立区间
    return base + edod_int * 100 + split_off + size_off


def file_name(type_, edod, split, size=50):
    e = f"{int(round(edod * 10)):02d}"
    t = type_.lower()
    return f"dcc_{size}_{t}_edod{e}_{split}.npz"


def plan_files():
    """返回 [(rel_path, type, edod, split, size, seed)] 列表。"""
    plan = []
    for size in [50]:
        for split, n in SPLIT_SIZES.items():
            for t in TYPES:
                for e in EDODS:
                    rel = f"50_node/{split}/{file_name(t, e, split, 50)}"
                    plan.append((rel, t, e, split, 50, seed_for(t, e, split, 50)))
    # 100-node test (cross-scale)
    for t in TYPES_100:
        e = 0.5
        rel = f"100_node/test/{file_name(t, e, 'test', 100)}"
        plan.append((rel, t, e, 'test', 100, seed_for(t, e, 'test', 100)))
    return plan


def main():
    parser = argparse.ArgumentParser(description='P0-D1: 重新生成 baseline 数据')
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(_CVRPTW, 'data', 'baseline'))
    parser.add_argument('--archive_dir', type=str,
                        default=os.path.join(_CVRPTW, 'data', 'archive', 'baseline_v1_buggy'))
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--skip_archive', action='store_true',
                        help='已归档过旧数据时跳过移动步骤')
    args = parser.parse_args()

    plan = plan_files()
    print(f"=== D1: 重新生成 baseline 数据（release-feasibility 修复）===")
    print(f"  数据根: {args.data_root}")
    print(f"  归档旧数据到: {args.archive_dir}")
    print(f"  文件数: {len(plan)}")

    if args.dry_run:
        for rel, t, e, split, size, seed in plan:
            print(f"    [DRY] {rel}  seed={seed}")
        return

    # 1. 归档旧数据（移动，不删除）
    if not args.skip_archive:
        os.makedirs(args.archive_dir, exist_ok=True)
        for rel, *_ in plan:
            src = os.path.join(args.data_root, rel)
            if os.path.exists(src):
                dst = os.path.join(args.archive_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                print(f"  [archive] {rel}")
        print("  旧数据已归档。")

    # 2. 重新生成
    for rel, t, e, split, size, seed in plan:
        out_path = os.path.join(args.data_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"  [generate] {rel} (seed={seed})", flush=True)
        ds = generate_dataset(
            num_instances=SPLIT_SIZES[split] if size == 50 else 1280,
            num_customers=size,
            instance_type=t,
            capacity=args.capacity,
            max_route_len=200,
            seed=seed,
            edod=e,
        )
        np.savez_compressed(out_path, **ds)

    # 3. 重新生成 manifest
    manifest_lines = []
    for rel, *_ in plan:
        p = os.path.join(args.data_root, rel)
        with open(p, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        manifest_lines.append(f"{sha}  {rel}")
    manifest_path = os.path.join(args.data_root, 'DATA_MANIFEST.sha256')
    with open(manifest_path, 'w') as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"  已写入 manifest: {manifest_path}")

    # 4. 快速 self-check：release feasibility
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'tests'))
    from test_release_feasibility import check_release_feasibility
    print("\n=== release feasibility self-check ===")
    all_ok = True
    for rel, *_ in plan:
        p = os.path.join(args.data_root, rel)
        ok, stats = check_release_feasibility(p, verbose=False)
        all_ok = all_ok and ok
        print(f"  {rel}: {'PASS' if ok else 'FAIL'} "
              f"(dyn={stats['n_dynamic']}, service_bad={stats['n_service_bad']})")
    print(f"\n  => {'ALL RELEASE-FEASIBLE ✓' if all_ok else 'FAIL — 存在 release-infeasible 订单'}")


if __name__ == '__main__':
    main()
