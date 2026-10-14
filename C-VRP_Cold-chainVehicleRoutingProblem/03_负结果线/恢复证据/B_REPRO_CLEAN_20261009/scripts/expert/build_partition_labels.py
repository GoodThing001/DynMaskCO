"""
Partition labels —— 从 OR assignment 生成 pairwise grouping 监督 G_ij（导师 2026-09-01）。

G_ij = 1[i, j 属于同一条 route]。这是 permutation-invariant 的 partition 监督，完全不涉及
vehicle ID（不需要知道 customer 属于 vehicle 3 还是 17），天然吸收 vehicle symmetry。

训练数据来源：OR-joint 的 assignment（teacher_vehicle[s, j] = customer j 的 vehicle）。两个客户
同 route ⟺ teacher_vehicle 相同（且都 != -1）。

后续 Hierarchical Fleet–Route 的 coarse/fine 关系：G_ij（same route，coarse）→ A_ij（adjacent，
fine），且 A_ij ≤ G_ij。
"""
import numpy as np


def build_partition_labels(teacher_vehicle):
    """teacher_vehicle: [S, N]（customer j 的 vehicle id，-1 = 未服务/未标记）。

    返回 G: [S, N, N] int8，G[s,i,j] = G[s,j,i] = 1 if i,j 同 route（且都 labeled）。
    depot（node 0）不参与 grouping（恒 0）。
    """
    teacher = np.asarray(teacher_vehicle)
    S, N = teacher.shape
    G = np.zeros((S, N, N), dtype=np.int8)
    for s in range(S):
        for i in range(1, N):
            ki = int(teacher[s, i])
            if ki == -1:
                continue
            for j in range(i + 1, N):
                if int(teacher[s, j]) == ki:
                    G[s, i, j] = G[s, j, i] = 1
    return G


def build_partition_labels_sparse(teacher_vehicle):
    """稀疏版：返回 {(s, i, j): 1} 或按 route 分组的 customer 集合（供后续 slot-level 监督）。

    返回 routes: list[dict]，每个 state 一个 {vehicle_id: [customers]}（仅 labeled customers）。
    """
    teacher = np.asarray(teacher_vehicle)
    S, N = teacher.shape
    routes = []
    for s in range(S):
        r = {}
        for j in range(1, N):
            k = int(teacher[s, j])
            if k == -1:
                continue
            r.setdefault(k, []).append(j)
        routes.append(r)
    return routes


if __name__ == '__main__':
    # 冒烟：2 states，5 customers（depot + 4），vehicle 0 服务 {1,3}、vehicle 1 服务 {2}
    teacher = np.array([
        [-1, 0, 1, 0, -1],
        [-1, 0, 0, 1, -1],
    ], dtype=np.int32)
    G = build_partition_labels(teacher)
    assert G[0, 1, 3] == 1 and G[0, 1, 2] == 0, "same-route 应 1，cross-route 应 0"
    assert G[1, 1, 2] == 1 and G[1, 1, 3] == 0
    print("PASS: build_partition_labels（G_ij 同 route=1，跨 route=0，permutation-invariant）")
