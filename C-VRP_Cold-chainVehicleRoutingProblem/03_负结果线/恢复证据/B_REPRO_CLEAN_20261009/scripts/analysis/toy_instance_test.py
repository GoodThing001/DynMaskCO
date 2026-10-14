"""
③ controlled toy instance 验证 — 证明品质项 causally 改变路由

设计：
- 对称距离（任意服务顺序的距离相同），隔离「距离」的影响
- 唯一差异是 temp_class（K_i：常温 K=0.01 >> 冷冻 K=0.0002）
- edge_logits 设零（均匀，隔离「模型」的影响）
- 对比 λ_q=0（距离优先，无偏好）vs λ_q 大（品质优先，应高 K 先服务）

预期：λ_q 增大后，常温（高 K）节点稳定排在前面（rank 更小），
证明 weighted completion time（∑ K_i·t_i）causally 改变路由。

用法：
  python analysis/toy_instance_test.py [--num_customers 8]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decoding'))
from resource_beam import ResourceBeamSearcher


def build_symmetric_toy(num_customers=8):
    """构造对称距离 toy：depot 中心 + 客户在圆上，dist_mat 手动对称。"""
    total = num_customers + 1

    # coords（仅用于 ResourceBeamSearcher 构造，dist_mat 会 override）
    coords = np.zeros((total, 2), dtype=np.float32)
    coords[0] = [0.5, 0.5]
    for i in range(1, total):
        angle = 2 * np.pi * (i - 1) / num_customers
        coords[i] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # 对称 dist_mat：depot-客户 = 1.0，客户-客户 = 2.0
    # → 任意服务顺序的总距离 = 2.0 × num_customers（完全相同）
    dist = np.full((total, total), 2.0, dtype=np.float32)
    dist[0, :] = 1.0
    dist[:, 0] = 1.0
    np.fill_diagonal(dist, 0.0)

    tw_start = np.zeros(total, dtype=np.float32)
    tw_end = np.full(total, 100.0, dtype=np.float32)   # 宽 TW（无约束）
    service_time = np.full(total, 0.1, dtype=np.float32)  # 相同 service
    demands = np.ones(total, dtype=np.float32)
    demands[0] = 0.0

    # 唯一差异：temp_class（前一半冷冻低K，后一半常温高K）
    # 关键设计：编号顺序 ≠ K 序，才能区分「距离优先（编号顺序）」vs「品质优先（K序）」
    temp_class = np.zeros(total, dtype=np.int32)
    half = num_customers // 2
    temp_class[1:1 + half] = 2     # 冷冻 K=0.0002（前一半，编号小）
    temp_class[1 + half:] = 0      # 常温 K=0.01（后一半，编号大）

    return coords, dist, tw_start, tw_end, service_time, demands, temp_class


def run_toy(num_customers=8):
    coords, dist, tw_start, tw_end, service_time, demands, temp_class = \
        build_symmetric_toy(num_customers)
    total = num_customers + 1
    half = num_customers // 2

    # edge_logits 设零：均匀，隔离模型偏好
    edge_logits = np.zeros((total, total), dtype=np.float32)

    print("=" * 70)
    print(f"③ controlled toy instance：{num_customers} 客户")
    print(f"   前 {half} 个 = 冷冻(低K=0.0002)，后 {half} 个 = 常温(高K=0.01)")
    print("   （编号顺序 = 冷冻先，K序 = 常温先，两者相反以区分「无偏好 vs 品质优先」）")
    print("   对称距离：任意服务顺序的总距离完全相同（隔离距离因素）")
    print("   edge_logits 全零：隔离模型因素")
    print("=" * 70)

    LAMBDA_QS = [0.0, 0.5, 2.0]
    for lq in LAMBDA_QS:
        searcher = ResourceBeamSearcher(
            coords, tw_start, tw_end, service_time, demands,
            capacity=50, speed=1.0, K=16, tw_margin=0.0,
            enable_quality=True, lambda_q=lq, temp_class=temp_class,
            quality_salable_threshold=0.02, dist_mat=dist)
        route, score = searcher.generate(edge_logits, max_steps=200)

        # 提取服务顺序（去 depot）
        seq = [n for n in route if n != 0]
        ranks = {n: i for i, n in enumerate(seq)}
        frozen = [n for n in seq if 1 <= n <= half]        # 前一半冷冻（编号小）
        ambient = [n for n in seq if half < n <= num_customers]  # 后一半常温（编号大）
        ambient_rank = np.mean([ranks[n] for n in ambient]) if ambient else float('nan')
        frozen_rank = np.mean([ranks[n] for n in frozen]) if frozen else float('nan')

        print(f"\nλ_q={lq}: 服务顺序 = {seq}")
        print(f"   常温平均 rank = {ambient_rank:.2f} | 冷冻平均 rank = {frozen_rank:.2f}")
        if ambient_rank < frozen_rank:
            print(f"   ✅ 常温(高K)先服务（品质优先生效）")
        elif ambient_rank > frozen_rank:
            print(f"   ➖ 冷冻先（λ_q=0 的编号顺序，无品质偏好）")
        else:
            print(f"   ➖ 无偏好（rank 相同）")

    print("\n" + "=" * 70)
    print("判定标准：")
    print("  λ_q=0 时 rank 差异小（距离无偏好，beam 取编号顺序）")
    print("  λ_q 增大后，常温(高K)的 rank 应显著小于冷冻(低K)的 rank")
    print("  → 证明 weighted completion time（∑ K_i·t_i）causally 改变路由")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=8,
                        help='toy 客户数（6-12 推荐）')
    args = parser.parse_args()
    run_toy(args.num_customers)
