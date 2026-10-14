"""
Pair Feasibility false-negative audit（Phase 0 Closure Step 3）。

回答主控文档 §6/Step 3 的问题：当前 `pair_feasible` 的每个 reject reason
（capacity / eta_tw / return_depot）是不是 sound 的 necessary condition？

方法：
  1. 用 JF1-H（heuristic joint）在 strict-online 环境 rollout VAL 数据，在每次真正
     replan 的 recourse point snapshot FleetState（recording replanner）。
  2. 对每个 snapshot，枚举 (active vehicle × unserved-visible customer)，用
     `pair_decision()` 计算 structured 判定 + reject reason。
  3. false-negative oracle：以最终 execution trace 为准 —— 若 pair_decision 判某 (k,j)
     infeasible，但 JF1-H 实际执行里 k 真的服务了 j，则记为 false negative。

纯 NumPy，无 JAX / model，可本地跑：

    python scripts/analysis/pair_feasibility_audit.py \
        --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
        --num_instances 128
"""
import sys, os, argparse
import numpy as np
from dataclasses import dataclass

_BASE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(os.path.dirname(_BASE), 'simulation')
sys.path.insert(0, _SIM)

from strict_online_env import StrictOnlineEnv, Replanner, VehicleState
from joint_fleet import (get_fleet_anchors, get_global_pool, compute_anchor_info,
                         pair_decision, JointAssignmentReplanner)


class RecordingReplanner(Replanner):
    """包装 JF1-H，在每次 plan() 前 snapshot FleetState，然后委托给 inner。"""

    def __init__(self):
        self.inner = JointAssignmentReplanner('heuristic')
        self.snapshots = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        snap = {
            'inst_idx': int(inst_idx),
            'clock': float(clock),
            'vehicles': [self._clone(v) for v in vehicles],
            'served_mask': served_mask.copy(),
            'visible_ids': list(visible_ids),
            'replan_ids': set(replan_ids) if replan_ids is not None else None,
        }
        self.snapshots.append(snap)
        self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)

    @staticmethod
    def _clone(v):
        return VehicleState(
            vehicle_id=v.vehicle_id, status=v.status, current_node=int(v.current_node),
            ready_time=float(v.ready_time), current_load=float(v.current_load),
            committed_next=v.committed_next, committed_arrive=v.committed_arrive,
            committed_finish=v.committed_finish, mutable_suffix=list(v.mutable_suffix),
            served_route=list(v.served_route), dispatch_time=v.dispatch_time,
            return_finish=v.return_finish, needs_replan=v.needs_replan,
            replan_reason=v.replan_reason,
        )


def executed_map(traces):
    """从 execution trace 抽 (vehicle_id, customer) -> True 的 map。"""
    exe = {}
    for tr in traces:
        for sr in tr.services:
            if sr.node != 0:
                exe[(tr.vehicle_id, int(sr.node))] = True
    return exe


def run_audit(dataset, capacity, num_instances):
    tw_speed = 1.0
    num_vehicles = 25
    total_pairs = 0
    reason_count = {'capacity': 0, 'eta_tw': 0, 'return_depot': 0}
    false_neg = {'capacity': 0, 'eta_tw': 0, 'return_depot': 0}
    # 记录每个 reason 的 (inst, vehicle, customer, exe_feasible) 样例，便于抽查
    fn_samples = {r: [] for r in reason_count}

    n = min(num_instances, dataset['coords'].shape[0])
    for inst_idx in range(n):
        rec = RecordingReplanner()
        env = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles, replanner=rec)
        traces, _ = env.run(inst_idx)
        exe = executed_map(traces)

        for snap in rec.snapshots:
            vehicles = snap['vehicles']
            anchors = get_fleet_anchors(vehicles)
            pool, _locked = get_global_pool(vehicles, snap['served_mask'],
                                            snap['visible_ids'], snap['replan_ids'])
            active_ids = {a.vehicle_id for a in anchors
                          if a.status in ('idle', 'ready')
                          and (snap['replan_ids'] is None or a.vehicle_id in snap['replan_ids'])}
            active_anchors = [a for a in anchors if a.vehicle_id in active_ids]
            if not active_anchors or not pool:
                continue
            anchor_info = compute_anchor_info(env, inst_idx, active_anchors)
            for j in pool:
                for a in active_anchors:
                    vid = a.vehicle_id
                    d = pair_decision(env, inst_idx, anchor_info, vid, j, 0.0, capacity)
                    total_pairs += 1
                    if d.feasible:
                        continue
                    reason_count[d.reason] += 1
                    if exe.get((vid, int(j)), False):
                        # pair 被判 infeasible，但 JF1-H 实际执行里 k 服务了 j → false negative
                        false_neg[d.reason] += 1
                        fn_samples[d.reason].append((inst_idx, vid, int(j), d))

    return total_pairs, reason_count, false_neg, fn_samples


def main():
    parser = argparse.ArgumentParser(description='Pair Feasibility false-negative audit')
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_instances', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=50)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    total, reason_count, false_neg, fn_samples = run_audit(dataset, args.capacity,
                                                           args.num_instances)

    print(f"=== Pair Feasibility Audit（{args.num_instances} instances）===")
    print(f"total (active vehicle, unserved customer) pairs evaluated: {total}")
    print()
    print(f"{'reason':<14} {'reject N':>9} {'false-neg':>10} {'FN rate':>9} {'sound?':>8}")
    print('-' * 54)
    for r in ('capacity', 'eta_tw', 'return_depot'):
        n = reason_count[r]
        fn = false_neg[r]
        rate = (fn / n) if n else 0.0
        sound = 'YES' if fn == 0 else 'NO'
        print(f"{r:<14} {n:>9} {fn:>10} {rate:>9.2%} {sound:>8}")
    print()
    print("结论：false-negative=0 的 reason 才是 sound necessary condition，可作 production hard mask；")
    print("      否则只能作 soft feature（尤其 return_depot，主控文档默认先作 feature）。")
    for r in ('capacity', 'eta_tw', 'return_depot'):
        if fn_samples[r]:
            print(f"\n[{r}] false-negative 样例（inst, vehicle, customer, decision）：")
            for s in fn_samples[r][:5]:
                print("  ", s)


if __name__ == '__main__':
    main()
