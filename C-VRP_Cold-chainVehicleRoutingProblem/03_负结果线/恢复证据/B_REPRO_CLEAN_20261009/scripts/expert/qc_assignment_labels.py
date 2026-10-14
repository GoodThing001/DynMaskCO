"""
JF2 Step 13 QC — OR-joint teacher label 审计（主控文档 §13 / §15）。

检查每个 state 的 teacher_vehicle：
  1. teacher ∈ candidate_mask（teacher 车辆必须在 sound candidate set 内，否则 candidate/teacher
     语义不一致，CE loss 无法在 feasible vehicles 上算）。
  2. teacher 车辆是 idle/ready（不是 committed/closed）。
  3. teacher 客户 ∈ mutable_mask（teacher 只标 mutable 客户）。
  4. 无 duplicate（一个客户最多被一辆车服务）。

用法（服务器，需 states .npz + teacher_vehicle.npz）：
    python scripts/expert/qc_assignment_labels.py \
        --states results/jf2/data/r1_edod05/states_train_0.npz \
        --teacher results/jf2/data/r1_edod05/teacher_vehicle.npz
"""
import sys, os, argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True)
    parser.add_argument('--teacher', required=True)
    args = parser.parse_args()

    a = dict(np.load(args.states))
    t = np.load(args.teacher)['teacher_vehicle']
    S, N = t.shape
    K = a['candidate_mask'].shape[1]

    n_labeled = 0
    n_not_candidate = 0
    n_not_active = 0
    n_not_mutable = 0
    n_dup = 0
    # 每 state 每车辆被标客户数（查 duplicate）
    for s in range(S):
        labeled = [(j, int(t[s, j])) for j in range(N) if t[s, j] != -1]
        n_labeled += len(labeled)
        for j, k in labeled:
            if not a['candidate_mask'][s, k, j]:
                n_not_candidate += 1
            if int(a['vehicle_status'][s, k]) not in (0, 1):  # idle/ready
                n_not_active += 1
            if not a['mutable_mask'][s, j]:
                n_not_mutable += 1
        # duplicate：同 state 同 vehicle 被多客户指向不违反 exact-once；真正的 dup 是同客户被多车服务
        # （t[s,j] 是标量，天然无 dup）。这里查「同 vehicle 客户是否重复」由 t 的构造保证，跳过。

    print(f"=== OR-joint teacher QC（{S} states）===")
    print(f"labeled customers: {n_labeled}")
    print(f"  teacher ∉ candidate_mask : {n_not_candidate}  ({n_not_candidate/max(1,n_labeled):.3%})")
    print(f"  teacher 非 idle/ready     : {n_not_active}    ({n_not_active/max(1,n_labeled):.3%})")
    print(f"  teacher 客户 ∉ mutable    : {n_not_mutable}   ({n_not_mutable/max(1,n_labeled):.3%})")
    print()
    if n_not_candidate == 0:
        print("PASS: teacher 全部落在 sound candidate set 内（candidate/teacher 语义一致）")
    else:
        print("FAIL: 存在 teacher ∉ candidate —— candidate/teacher 语义不一致，需调查")


if __name__ == '__main__':
    main()
