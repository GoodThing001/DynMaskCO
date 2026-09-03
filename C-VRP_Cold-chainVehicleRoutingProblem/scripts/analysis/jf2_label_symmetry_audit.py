"""
JF2 M0 诊断 — OR 标签对称性审计（主控文档 §12.3 / §25.2）。

M0 null（disagreement top-1 ≈ 随机）的最可能根因：同质车辆（idle@depot）等价，OR-Tools 在
等价车之间任意选一个，teacher_vehicle 是噪声。

本脚本（纯 NumPy，读已算好的 states + teacher）统计：在 teacher ≠ min-travel 的分歧对上，
有多少是「teacher 的车辆与 min-travel 车辆等价（同 anchor node/time/load）」——即分歧只是
对称性导致的任意选择，而非 OR 真的偏好另一辆不同的车。

判读：
  - 若「对称分歧」占比高（如 >70%）→ CE label 被对称性噪声主导 → 转 equivalence-aware /
    preference / regret learning（主控文档 §27.4）。
  - 若「非对称分歧」占比高 → 分歧是真实的（OR 偏好不同的车），M0 学不到是特征/模型问题。

用法：
    python scripts/analysis/jf2_label_symmetry_audit.py \
        --states results/jf2/data/r1_edod05/states_train_0.npz \
        --teacher results/jf2/data/r1_edod05/teacher_vehicle.npz
"""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True)
    parser.add_argument('--teacher', required=True)
    args = parser.parse_args()

    a = dict(np.load(args.states))
    teacher = np.load(args.teacher)['teacher_vehicle']
    anchor_ids = a['anchor_ids']          # [S,K] anchor node
    veh_feat = a['veh_feat']              # [S,K,4] = [t/T, load/Q, rem_cap/Q, status]
    base_score = a['base_score']          # [S,K,N]
    cand_mask = a['candidate_mask']       # [S,K,N] bool
    status = a['vehicle_status']          # [S,K]

    S, K, N = base_score.shape
    # min-travel 车 = argmax base_score（不可行处置 -1e9）
    min_travel = np.argmax(np.where(cand_mask, base_score, -1e9), axis=1)  # [S,N]

    n_labeled = 0
    n_agree = 0          # teacher == min-travel
    n_disagr = 0         # teacher != min-travel
    n_sym = 0            # 分歧且 teacher ≡ min-travel（同 anchor node/time/load）
    n_genuine = 0        # 分歧且 teacher 与 min-travel 车不等价

    # 按「分歧时的等价类型」细分
    sym_types = {'same_idle_depot': 0, 'same_ready_node': 0, 'same_time_load': 0}

    eps = 1e-4
    for s in range(S):
        for j in range(1, N):
            kt = int(teacher[s, j])
            if kt == -1:
                continue
            n_labeled += 1
            km = int(min_travel[s, j])
            if kt == km:
                n_agree += 1
                continue
            n_disagr += 1
            same_node = (anchor_ids[s, kt] == anchor_ids[s, km])
            same_time = abs(veh_feat[s, kt, 0] - veh_feat[s, km, 0]) < eps
            same_load = abs(veh_feat[s, kt, 1] - veh_feat[s, km, 1]) < eps
            if same_node and same_time and same_load:
                n_sym += 1
                if int(status[s, kt]) == 0 and int(status[s, km]) == 0:
                    sym_types['same_idle_depot'] += 1
                elif same_node != 0:
                    sym_types['same_ready_node'] += 1
                else:
                    sym_types['same_time_load'] += 1
            else:
                n_genuine += 1

    print(f"=== JF2 OR-label symmetry audit ===")
    print(f"labeled customers: {n_labeled}")
    print(f"  teacher == min-travel: {n_agree} ({n_agree/max(1,n_labeled):.1%})")
    print(f"  teacher != min-travel: {n_disagr} ({n_disagr/max(1,n_labeled):.1%})")
    print(f"    ├─ 对称分歧（teacher ≡ min-travel 车）: {n_sym} ({n_sym/max(1,n_disagr):.1%})")
    print(f"    │    ├─ 同为 idle@depot: {sym_types['same_idle_depot']}")
    print(f"    │    ├─ 同为 ready@某节点: {sym_types['same_ready_node']}")
    print(f"    │    └─ 其他等价: {sym_types['same_time_load']}")
    print(f"    └─ 真实分歧（teacher 是不同的车）: {n_genuine} ({n_genuine/max(1,n_disagr):.1%})")
    print()
    if n_sym / max(1, n_disagr) > 0.7:
        print("结论：分歧主要由对称性主导 → CE label 噪声，应转 equivalence-aware/preference learning")
    elif n_genuine / max(1, n_disagr) > 0.7:
        print("结论：分歧是真实的 → M0 学不到是特征/模型问题，不是对称性")
    else:
        print("结论：对称与真实分歧混合，需进一步拆解")


if __name__ == '__main__':
    main()
