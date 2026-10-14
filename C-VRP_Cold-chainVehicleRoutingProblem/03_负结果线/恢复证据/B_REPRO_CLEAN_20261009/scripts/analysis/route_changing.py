"""
① Route-changing test — λ_q 扫描的路线变化分析

验证 λ_q 增大后路线是否发生系统性变化。报告：
- edge overlap（相邻 λ_q 路线边的重合率）
- edit distance（路线序列差异）
- distance / quality loss / weighted service time / feasibility

用法：
  python analysis/route_changing.py \
      --data <test.npz> \
      --routes "lq0:/tmp/routes_lq0.npz,lq0.5:/tmp/routes_lq05.npz,lq2:/tmp/routes_lq2.npz"
"""

import sys, os, argparse
import numpy as np

K_TEMP = np.array([0.01, 0.002, 0.0002], dtype=np.float32)


def route_to_edges(route, num_nodes):
    edges = set()
    prev = 0
    for x in route:
        node = int(x)
        if node == 0:
            prev = 0
            continue
        if node >= num_nodes:
            continue
        edges.add((prev, node))
        prev = node
    return edges


def route_metrics(route, dist_mat, service_time, temp_class):
    """单条路线的 distance / Q / weighted service time。"""
    num_nodes = dist_mat.shape[0]
    dist = 0.0
    q = 0.0
    weighted_t = 0.0
    cum = 0.0
    prev = 0
    for x in route:
        node = int(x)
        if node == 0:
            cum = 0.0; prev = 0; continue
        if node >= num_nodes:
            continue
        cum += dist_mat[prev, node] + service_time[prev]
        dist += dist_mat[prev, node]
        k = K_TEMP[int(temp_class[node])]
        q += k * cum            # 一阶近似品质损耗 = K_i · t_i
        weighted_t += k * cum   # weighted completion time
        prev = node
    return dist, q, weighted_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--routes', type=str, required=True,
                        help='逗号分隔的 label:path 对，如 "lq0:a.npz,lq2:b.npz"')
    args = parser.parse_args()

    data = dict(np.load(args.data))
    dist_mat = data.get('dist_mat', None)
    if dist_mat is None:
        coords = data['coords']
        dist_mat = np.sqrt(((coords[:, :, None, :] - coords[:, None, :, :]) ** 2).sum(-1))
    service_time = data['service_time']
    temp_class = data['temp_class']

    # 解析 label:path
    entries = []
    for item in args.routes.split(','):
        label, path = item.split(':', 1)
        entries.append((label, np.load(path)['routes']))

    num_nodes = dist_mat.shape[1]
    N = entries[0][1].shape[0]

    # 每个 λ_q 的指标
    print(f"\n{'='*70}")
    print(f"① route-changing test（{N} 实例，{num_nodes-1} 节点）")
    print(f"{'='*70}")
    print(f"{'λ_q':<8} {'distance':>10} {'quality(Q)':>12} {'weighted time':>13}")
    for label, routes in entries:
        d_sum = q_sum = w_sum = 0.0
        for b in range(N):
            d, q, w = route_metrics(routes[b], dist_mat[b], service_time[b], temp_class[b])
            d_sum += d; q_sum += q; w_sum += w
        print(f"{label:<8} {d_sum/N:>10.4f} {q_sum/N:>12.4f} {w_sum/N:>13.4f}")

    # 相邻 λ_q 的 edge overlap / edit distance
    print(f"\n{'='*70}")
    print("路线变化（相邻 λ_q 对比）")
    print(f"{'='*70}")
    print(f"{'对比':<20} {'edge overlap':>13} {'edit distance':>13}")
    for i in range(1, len(entries)):
        l1, r1 = entries[i-1]
        l2, r2 = entries[i]
        overlap_sum = 0.0
        edit_sum = 0.0
        for b in range(N):
            e1 = route_to_edges(r1[b], num_nodes)
            e2 = route_to_edges(r2[b], num_nodes)
            inter = len(e1 & e2)
            denom = max(len(e1), len(e2), 1)
            overlap_sum += inter / denom
            # 简化 edit distance：两路线共有边之外的部分比例
            edit_sum += 1.0 - inter / denom
        print(f"{l1} → {l2:<15} {overlap_sum/N:>13.4f} {edit_sum/N:>13.4f}")

    print(f"\n解读：")
    print("  若 λ_q 增大后 edge overlap 显著 < 1（edit distance 显著 > 0），")
    print("  说明品质项 causally 改变了路线。")
    print("  若 overlap ≈ 1（edit distance ≈ 0），说明路线几乎不变，品质感知无效果。")


if __name__ == '__main__':
    main()
