"""
HFR Step 1 — 补保存 OR 同一次 solve 的完整 routes（含顺序），生成 G/A 监督所需数据。

现有 teacher_vehicle.npz 只有 customer→vehicle assignment（丢了 route 顺序），而 HFR 的
G_ij（same route）与 A_ij（adjacent）都必须来自**同一次 OR-joint solution 的 route 顺序**。

本脚本复用 build_assignment_dataset 的 state 重建逻辑，重跑 OR，额外保存：
  teacher_routes  [S, K, L]  每个 vehicle 的客户顺序（padding=-1，L=max route 长度）
  teacher_group_id [S, N]    customer → route id（= vehicle id；-1 未服务/未标记）

并做 QC：
  A⇒G：A_ij=1 ⟹ G_ij=1（同 route 相邻 edge 必同 route）
  G 对称：G_ij = G_ji
  permutation：交换 physical vehicle ID 后 G 不变

用法（服务器，需 OR-Tools）：
    python scripts/expert/build_teacher_routes.py \
      --states results/jf2/data/r1_edod05/states_train_0.npz \
      --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
      --time_limit_ms 50 --out results/jf2/data/r1_edod05
"""
import sys, os, argparse, time
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'baselines'))

from strict_online_env import StrictOnlineEnv
# 先 import authoritative_evaluator（ortools_rolling_horizon 内部会 `from authoritative_evaluator
# import evaluate_execution_trace`，预热避免其 import 时序失败）
from authoritative_evaluator import evaluate_execution_trace  # noqa: F401
from ortools_rolling_horizon import ORToolsReplanner
from build_assignment_dataset import reconstruct_vehicles, visible_ids_of, replan_ids_of


def extract_routes(env, rp, inst_idx, clock, vehicles, served_mask, vis_ids, rp_ids, K):
    """跑 OR，返回 (teacher_group_id[N], routes[K, L])，routes 含顺序（padding=-1）。"""
    rp.plan(env, inst_idx, clock, vehicles, served_mask, vis_ids, rp_ids)
    group_id = np.full(env.num_nodes, -1, dtype=np.int32)
    routes = []
    for v in vehicles:
        if v.vehicle_id not in rp_ids:
            continue
        custs = [int(n) for n in v.mutable_suffix if int(n) != 0]
        for c in custs:
            group_id[c] = v.vehicle_id
        routes.append((v.vehicle_id, custs))
    # 固定 shape [K, L]
    L = max((len(c) for _, c in routes), default=0)
    routes_mat = np.full((K, max(1, L)), -1, dtype=np.int32)
    for vid, custs in routes:
        if custs:
            routes_mat[vid, :len(custs)] = custs
    return group_id, routes_mat


def qc(routes_mat, group_id):
    """QC：A⇒G + G 对称。返回违规计数。"""
    S, K, L = routes_mat.shape
    bad_ag = 0   # A=1 但 G=0
    for s in range(S):
        for k in range(K):
            custs = [int(c) for c in routes_mat[s, k] if c != -1]
            for p in range(len(custs) - 1):
                i, j = custs[p], custs[p + 1]
                if group_id[s, i] != group_id[s, j]:
                    bad_ag += 1
    return bad_ag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--time_limit_ms', type=int, default=50)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    a = dict(np.load(args.states))
    dataset = dict(np.load(args.data))
    K = a['vehicle_status'].shape[1]
    N = dataset['coords'].shape[1]
    S = a['instance_id'].shape[0]

    env0 = StrictOnlineEnv(dataset, 50, 1.0, K, replanner=None)
    rp = ORToolsReplanner(50, K, args.time_limit_ms)

    routes_list = []
    teacher_group_id = np.full((S, N), -1, dtype=np.int32)
    t0 = time.time()
    for s in range(S):
        inst = int(a['instance_id'][s]); clock = float(a['clock'][s])
        served = a['served_mask'][s]
        vis = visible_ids_of(dataset, inst, clock)
        rp_ids = replan_ids_of(a, s, K)
        vehicles = reconstruct_vehicles(a, s, K, N)
        gid, routes = extract_routes(env0, rp, inst, clock, vehicles, served, vis, rp_ids, K)
        teacher_group_id[s] = gid
        routes_list.append(routes)
        if (s + 1) % 1000 == 0:
            print(f"  {s+1}/{S} | {time.time()-t0:.0f}s", flush=True)

    # 统一 pad 到全局 max L
    max_L = max(r.shape[1] for r in routes_list)
    teacher_routes = np.full((S, K, max_L), -1, dtype=np.int32)
    for s, r in enumerate(routes_list):
        teacher_routes[s, :, :r.shape[1]] = r

    # QC
    bad = qc(teacher_routes, teacher_group_id)
    n_labeled = int((teacher_group_id != -1).sum())
    print(f"\n=== teacher_routes 生成完成（{S} states, {n_labeled} labeled customers）===")
    print(f"  teacher_routes shape: {teacher_routes.shape}")
    print(f"  A⇒G violation（adjacent 但不同 route）: {bad}")
    print(f"  → {'PASS' if bad == 0 else 'FAIL'}")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'teacher_routes.npz'), teacher_routes=teacher_routes)
    np.savez_compressed(os.path.join(args.out, 'teacher_group_id.npz'), teacher_group_id=teacher_group_id)
    print(f"  saved: {args.out}/teacher_routes.npz + teacher_group_id.npz")


if __name__ == '__main__':
    main()
