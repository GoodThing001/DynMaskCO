"""
JF2 Step 12-13 — OR-joint assignment teacher 标注。

读 collect_assignment_states.py 采的 state（.npz），对每个 state 重建 strict-online 状态，
用 ORToolsReplanner（OR-joint，frozen prefix + committed leg 冻结 + visible mutable 跨车重分配）
求解，抽出 teacher_vehicle[j]（customer j 由哪辆车服务；-1 = OR 未能服务）。

Step 12（budget calibration）：--num_states 200 扫 --time_limit_ms {50,100,250,500,1000}，
比较 completion / objective / runtime / assignment stability，选 quality-time knee。
Step 13（full labels）：用选定 budget 跑全部 state，把 teacher_vehicle 写回 .npz。

需 OR-Tools，服务器跑：

    # Step 12 calibration
    for ms in 50 100 250 500 1000; do
      python scripts/expert/build_assignment_dataset.py \
        --states results/jf2/data/r1_edod05/states_train_0.npz \
        --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
        --num_states 200 --time_limit_ms $ms --out results/jf2/data/r1_edod05/calib_$ms
    done

    # Step 13 full labels
    python scripts/expert/build_assignment_dataset.py \
      --states results/jf2/data/r1_edod05/states_train_0.npz \
      --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
      --time_limit_ms 250 --out results/jf2/data/r1_edod05
"""
import sys, os, argparse, time, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'baselines'))

from strict_online_env import StrictOnlineEnv, VehicleState
from ortools_rolling_horizon import ORToolsReplanner

STATUS_CODE_INV = {0: 'idle', 1: 'ready', 2: 'committed', 3: 'returning', 4: 'closed'}


def reconstruct_vehicles(a, s, K, N):
    vehicles = []
    for k in range(K):
        tail = [int(j) for j in range(N) if a['tail_mask'][s, k, j]]
        vehicles.append(VehicleState(
            vehicle_id=k,
            status=STATUS_CODE_INV.get(int(a['vehicle_status'][s, k]), 'closed'),
            current_node=int(a['vehicle_node'][s, k]),
            ready_time=float(a['vehicle_ready'][s, k]),
            current_load=float(a['vehicle_load'][s, k]),
            committed_next=(None if int(a['vehicle_comm_next'][s, k]) == -1
                            else int(a['vehicle_comm_next'][s, k])),
            committed_finish=(None if a['vehicle_comm_finish'][s, k] == -1.0
                              else float(a['vehicle_comm_finish'][s, k])),
            mutable_suffix=tail,
        ))
    return vehicles


def visible_ids_of(dataset, inst_idx, clock):
    N = dataset['coords'].shape[1]
    return [i for i in range(1, N)
            if dataset['demands'][inst_idx, i] > 0
            and dataset['reveal_time'][inst_idx, i] <= clock + 1e-6]


def replan_ids_of(a, s, K):
    return {k for k in range(K) if a['replan_mask'][s, k]}


def label_one(env, rp, inst_idx, clock, vehicles, served_mask, vis_ids, rp_ids, N):
    rp.plan(env, inst_idx, clock, vehicles, served_mask, vis_ids, rp_ids)
    teacher = np.full(N, -1, dtype=np.int32)
    # 只读 active 车辆（rp_ids）的 OR 分配；committed 车辆的 mutable_suffix 是冻结 tail（reserved），
    # 不是 OR assignment，不能标成 teacher。
    for v in vehicles:
        if v.vehicle_id not in rp_ids:
            continue
        for n in v.mutable_suffix:
            if int(n) != 0:
                teacher[int(n)] = v.vehicle_id
    return teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_states', type=int, default=-1, help='-1 = 全部')
    parser.add_argument('--time_limit_ms', type=int, default=50)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    a = dict(np.load(args.states))
    dataset = dict(np.load(args.data))
    N = dataset['coords'].shape[1]
    S = a['instance_id'].shape[0]
    idx = np.arange(S) if args.num_states < 0 else np.arange(min(args.num_states, S))

    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles, replanner=None)
    rp = ORToolsReplanner(args.capacity, args.num_vehicles, args.time_limit_ms)

    teacher = np.full((S, N), -1, dtype=np.int32)
    n_unserved = 0
    t0 = time.time()
    for s in idx:
        inst_idx = int(a['instance_id'][s])
        clock = float(a['clock'][s])
        vehicles = reconstruct_vehicles(a, s, args.num_vehicles, N)
        served_mask = a['served_mask'][s]
        vis_ids = visible_ids_of(dataset, inst_idx, clock)
        rp_ids = replan_ids_of(a, s, args.num_vehicles)
        teacher[s] = label_one(env, rp, inst_idx, clock, vehicles, served_mask,
                               vis_ids, rp_ids, N)
        # mutable customers 但 teacher=-1 → OR 未服务
        for j in np.where(a['mutable_mask'][s])[0]:
            if teacher[s, j] == -1:
                n_unserved += 1
        if (s + 1) % 100 == 0:
            print(f"  labeled {s+1}/{len(idx)} | {time.time()-t0:.0f}s", flush=True)

    # 汇总
    n_mutable = int(a['mutable_mask'][idx].sum())
    served_rate = 1.0 - n_unserved / max(1, n_mutable)
    print(f"=== OR-joint labeling（{len(idx)} states, {args.time_limit_ms}ms）===")
    print(f"  mutable customers served by OR: {served_rate:.1%}  ({n_mutable-n_unserved}/{n_mutable})")
    print(f"  total time: {time.time()-t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    # 保存 teacher_vehicle（完整 S 长度，未标的 state 保持 -1）
    np.savez_compressed(os.path.join(args.out, 'teacher_vehicle.npz'),
                        teacher_vehicle=teacher)
    with open(os.path.join(args.out, 'label_summary.json'), 'w') as f:
        json.dump({'num_states_labeled': int(len(idx)), 'time_limit_ms': args.time_limit_ms,
                   'mutable_customers': int(n_mutable), 'served': int(n_mutable - n_unserved),
                   'served_rate': float(served_rate)}, f, indent=2)
    print(f"  saved: {args.out}/teacher_vehicle.npz + label_summary.json")


if __name__ == '__main__':
    main()
