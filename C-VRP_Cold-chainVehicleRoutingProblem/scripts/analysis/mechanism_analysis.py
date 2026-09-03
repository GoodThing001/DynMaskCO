"""
② 机制验证 — 分温区统计 Mean service rank + Mean delivery time

验证品质感知的机制：weighted completion time（∑ K_i·t_i）意味着
高 K（常温）货物应更早服务（rank 小）+ 更早送达（delivery time 小），
即 K 序 ↔ t 序反相关：
  K_ambient(0.01) > K_refrig(0.002) > K_frozen(0.0002)
  →  t_ambient < t_refrig < t_frozen

用法：
  python analysis/mechanism_analysis.py \
      --routes /tmp/routes_lq0.npz --data <test.npz> [--routes_lq /tmp/routes_lq2.npz]
"""

import sys, os, argparse
import numpy as np

K_TEMP = np.array([0.01, 0.002, 0.0002], dtype=np.float32)
CLASS_NAMES = ['常温(0.01)', '冷藏(0.002)', '冷冻(0.0002)']


def analyze_routes(routes, data, label=''):
    """从最终路线计算分温区的 service rank + delivery time。"""
    coords = data['coords']
    service_time = data['service_time']
    temp_class = data['temp_class']
    dist_mat = data.get('dist_mat', None)
    if dist_mat is None:
        # 若无 dist_mat，用 coords 算欧氏距离
        dist_mat = np.sqrt(((coords[:, :, None, :] - coords[:, None, :, :]) ** 2).sum(-1))

    N = routes.shape[0]
    # 分温区累计 rank 和 delivery time
    rank_sum = np.zeros(3)
    dt_sum = np.zeros(3)
    count = np.zeros(3, dtype=int)

    for b in range(N):
        route = routes[b]
        rank = 0
        cum = 0.0
        prev = 0
        for x in route:
            node = int(x)
            if node == 0:
                cum = 0.0; prev = 0; continue
            if node >= dist_mat.shape[1]:
                continue
            cum += dist_mat[b, prev, node] + service_time[b, prev]
            tc = int(temp_class[b, node])
            if 0 <= tc <= 2:
                rank_sum[tc] += rank
                dt_sum[tc] += cum
                count[tc] += 1
            prev = node
            rank += 1

    print(f"\n{'='*60}")
    print(f"{label}（{N} 实例）")
    print(f"{'='*60}")
    print(f"{'温区':<14} {'K_i':>8} {'Mean rank':>10} {'Mean delivery time':>18} {'数量':>6}")
    for tc in range(3):
        if count[tc] > 0:
            print(f"{CLASS_NAMES[tc]:<14} {K_TEMP[tc]:>8.4f} "
                  f"{rank_sum[tc]/count[tc]:>10.2f} {dt_sum[tc]/count[tc]:>18.4f} {count[tc]:>6}")
    return rank_sum / np.maximum(count, 1), dt_sum / np.maximum(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str, required=True, help='保存的路线 .npz（--save_routes 输出）')
    parser.add_argument('--data', type=str, required=True, help='原始数据 .npz')
    parser.add_argument('--routes_lq', type=str, default=None,
                        help='可选：第二个 λ_q 的路线 .npz，用于对比')
    args = parser.parse_args()

    data = dict(np.load(args.data))
    routes = np.load(args.routes)['routes']
    rank0, dt0 = analyze_routes(routes, data, label=os.path.basename(args.routes))

    if args.routes_lq:
        routes2 = np.load(args.routes_lq)['routes']
        rank1, dt1 = analyze_routes(routes2, data, label=os.path.basename(args.routes_lq))
        print(f"\n{'='*60}")
        print("机制验证（K 序 ↔ t 序反相关）")
        print(f"{'='*60}")
        print("预期：λ_q 增大后，高 K（常温）的 rank 和 delivery time 显著下降")
        print("      （品质优先生效：高 K 货物更早服务/送达）")


if __name__ == '__main__':
    main()
