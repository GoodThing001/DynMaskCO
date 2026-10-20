"""
方向三：Zone Density 特征生成。

为现有数据集的每个实例添加 zone_density 特征（第 8 维）。

zone_density[node] = 该节点所在空间区域的"历史订单到达概率" (0-1)

用法:
    python data/add_zone_density.py \
        --input dcc_50_r1_edod05_train.npz \
        --output dcc_50_r1_edod05_8d_train.npz \
        --n_clusters 8
"""

import sys, os, argparse, numpy as np
from sklearn.cluster import KMeans


def add_zone_density(dataset, n_clusters=8):
    """
    对数据集中的所有实例统一做空间聚类，为每个节点添加 zone_density 特征。

    zone_density = 该 cluster 中"会被揭示的订单"比例
    （模拟该区域的动态订单密度）
    """
    coords = dataset['coords']  # (N, nodes, 2)
    reveal_time = dataset.get('reveal_time',
        np.zeros_like(dataset['demands'], dtype=np.float32))
    num_instances, num_nodes = coords.shape[:2]

    # 1. 收集所有非 depot 节点的坐标（跨实例）
    all_coords = coords[:, 1:, :].reshape(-1, 2)  # (N*(nodes-1), 2)

    # 2. K-means 聚类
    print(f"  Clustering {all_coords.shape[0]} points into {n_clusters} zones...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_coords)  # (N*(nodes-1),)

    # 3. 统计每个 cluster 的"动态订单比例"
    all_reveal = reveal_time[:, 1:].reshape(-1)  # (N*(nodes-1),)
    dynamic_frac = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = labels == c
        dynamic_frac[c] = (all_reveal[mask] > 0).mean()

    # 归一化到 [0, 1]
    dynamic_frac = dynamic_frac / (dynamic_frac.max() + 1e-8)
    print(f"  Zone density: {dict(enumerate(dynamic_frac.round(3)))}")

    # 4. 为每个实例的每个节点分配 zone_density
    zone_density = np.zeros_like(reveal_time)
    labels_2d = labels.reshape(num_instances, num_nodes - 1)
    for i in range(num_instances):
        for j in range(num_nodes - 1):
            c = labels_2d[i, j]
            zone_density[i, j + 1] = dynamic_frac[c]  # depot (j=0) stays 0

    dataset['zone_density'] = zone_density.astype(np.float32)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=8)
    args = parser.parse_args()

    print(f"Adding zone_density to {args.input}...")
    data = dict(np.load(args.input))
    data = add_zone_density(data, args.n_clusters)

    np.savez_compressed(args.output, **data)
    print(f"Saved: {args.output}")
    for k, v in data.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == '__main__':
    main()
