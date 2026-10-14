"""
JF2 Step 10 — JF1-H rollout state 采集（student states，供 OR-joint teacher 标注）。

在 strict-online 环境下用 JF1-H（heuristic joint）rollout train 数据，在每次真正 replan 的
recourse point snapshot FleetState + build_jf2_features，保存固定 shape 数组 + audit JSONL。

输出（主控文档 §10.3 / Part XX）：
  states_<split>_<shard>.npz   固定 shape 特征张量（teacher_vehicle 占位 -1，Step 13 填）
  states_<split>_audit.jsonl   每 state 元数据（instance/event/clock/replan_reason/customer_order/feasible）
  manifest.json                schema version + split + policy + 计数 + strata 汇总

纯 NumPy，可本地跑：

    python scripts/expert/collect_assignment_states.py \
        --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
        --num_instances 256 --split train --out_dir results/jf2/data/r1_edod05
"""
import sys, os, argparse, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))

from strict_online_env import StrictOnlineEnv, Replanner, VehicleState
from joint_fleet import JointAssignmentReplanner
from fleet_features import build_jf2_features, Fv, Fp

STATUS_CODE = {'idle': 0, 'ready': 1, 'committed': 2, 'returning': 3, 'closed': 4}


class RecordingReplanner(Replanner):
    """包装 JF1-H，在每次 plan() 前 snapshot FleetState，然后委托 inner。"""

    def __init__(self):
        self.inner = JointAssignmentReplanner('heuristic')
        self.snapshots = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        snap = {
            'inst_idx': int(inst_idx),
            'clock': float(clock),
            'event_id': int(getattr(env, 'event_id', -1)),
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


def _replan_reason(snap):
    for v in snap['vehicles']:
        if v.vehicle_id in (snap['replan_ids'] or set()):
            if v.replan_reason is not None:
                return v.replan_reason
    return 'unknown'


def collect(dataset, num_instances):
    """跑 JF1-H rollout，返回 (snapshots, features_list, metadata_list)。"""
    capacity = 50
    K = 25
    snaps = []
    metas = []
    feats = []
    n = min(num_instances, dataset['coords'].shape[0])
    for inst_idx in range(n):
        rec = RecordingReplanner()
        env = StrictOnlineEnv(dataset, capacity, 1.0, K, replanner=rec)
        env.run(inst_idx)
        for snap in rec.snapshots:
            if not snap['replan_ids']:
                continue
            f = build_jf2_features(env, snap['inst_idx'], snap['clock'], snap['vehicles'],
                                   snap['served_mask'], snap['visible_ids'], snap['replan_ids'])
            if not f.active_vehicle_ids or not f.customer_order:
                continue
            reason = _replan_reason(snap)
            snaps.append(snap)
            feats.append(f)
            metas.append({
                'instance_id': int(snap['inst_idx']), 'event_id': int(snap['event_id']),
                'clock': float(snap['clock']), 'replan_reason': reason,
                'customer_order': list(f.customer_order),
                'feasible_vehicles': {str(j): [int(v) for v in f.candidate.feasible_vehicles[j]]
                                      for j in f.customer_order},
            })
    return snaps, feats, metas


def build_arrays(snaps, feats, K, N):
    S = len(feats)
    out = {
        'instance_id': np.zeros(S, np.int32),
        'event_id': np.zeros(S, np.int32),
        'clock': np.zeros(S, np.float32),
        'served_mask': np.zeros((S, N), bool),
        'mutable_mask': np.zeros((S, N), bool),
        'vehicle_status': np.zeros((S, K), np.int8),
        'vehicle_node': np.zeros((S, K), np.int32),
        'vehicle_ready': np.zeros((S, K), np.float32),
        'vehicle_load': np.zeros((S, K), np.float32),
        'vehicle_comm_next': np.full((S, K), -1, np.int32),
        'vehicle_comm_finish': np.full((S, K), -1.0, np.float32),
        'anchor_ids': np.zeros((S, K), np.int32),
        'veh_feat': np.zeros((S, K, Fv), np.float32),
        'pair_feat': np.zeros((S, K, N, Fp), np.float32),
        'candidate_mask': np.zeros((S, K, N), bool),
        'base_score': np.zeros((S, K, N), np.float32),
        'teacher_vehicle': np.full((S, N), -1, np.int32),
        'replan_mask': np.zeros((S, K), bool),      # 该 state 里哪些车在 replan_ids（active）
        'tail_mask': np.zeros((S, K, N), bool),     # 该 state 里每辆车的 mutable_suffix tail（非 0 客户）
    }
    for s, (snap, f) in enumerate(zip(snaps, feats)):
        out['instance_id'][s] = snap['inst_idx']
        out['event_id'][s] = snap['event_id']
        out['clock'][s] = snap['clock']
        out['served_mask'][s] = snap['served_mask']
        for j in f.customer_order:
            out['mutable_mask'][s, int(j)] = True
        rp = snap['replan_ids'] or set()
        for k in range(K):
            out['replan_mask'][s, k] = (k in rp)
        for v in snap['vehicles']:
            k = v.vehicle_id
            out['vehicle_status'][s, k] = STATUS_CODE.get(v.status, 4)
            out['vehicle_node'][s, k] = v.current_node
            out['vehicle_ready'][s, k] = v.ready_time
            out['vehicle_load'][s, k] = v.current_load
            out['vehicle_comm_next'][s, k] = -1 if v.committed_next is None else v.committed_next
            out['vehicle_comm_finish'][s, k] = -1.0 if v.committed_finish is None else v.committed_finish
            for n in v.mutable_suffix:
                if int(n) != 0:
                    out['tail_mask'][s, k, int(n)] = True
        for i, vid in enumerate(f.active_vehicle_ids):
            out['anchor_ids'][s, vid] = f.anchor_ids[i]
            out['veh_feat'][s, vid] = f.veh_feat[i]
            out['pair_feat'][s, vid] = f.pair_feat[i]
            out['candidate_mask'][s, vid] = f.candidate_mask[i]
            out['base_score'][s, vid] = f.base_score[i]
    return out


def strata_summary(metas, K, N):
    reasons = {}
    n_pending_bins = {}
    for m in metas:
        reasons[m['replan_reason']] = reasons.get(m['replan_reason'], 0) + 1
        np_ = len(m['customer_order'])
        b = '0-10' if np_ <= 10 else '11-20' if np_ <= 20 else '21-30' if np_ <= 30 else '31+'
        n_pending_bins[b] = n_pending_bins.get(b, 0) + 1
    return {'n_states': len(metas), 'replan_reason': reasons, 'n_pending_bins': n_pending_bins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_instances', type=int, default=256)
    parser.add_argument('--split', default='train')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    K, N = args.num_vehicles, dataset['coords'].shape[1]

    snaps, feats, metas = collect(dataset, args.num_instances)

    os.makedirs(args.out_dir, exist_ok=True)
    arrs = build_arrays(snaps, feats, K, N)
    prefix = f"states_{args.split}_{args.shard}"
    np.savez_compressed(os.path.join(args.out_dir, prefix + '.npz'), **arrs)
    with open(os.path.join(args.out_dir, prefix + '_audit.jsonl'), 'w') as f:
        for m in metas:
            f.write(json.dumps(m) + '\n')

    summary = strata_summary(metas, K, N)
    manifest = {
        'schema_version': 'jf2_state_v1',
        'data_split': args.split,
        'policy': 'JF1-H',
        'candidate_semantics': 'jf2_sound_v1',
        'num_vehicles': K,
        'capacity': args.capacity,
        'num_instances_rolled': min(args.num_instances, dataset['coords'].shape[0]),
        'data': os.path.basename(args.data),
        'summary': summary,
    }
    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"=== collect_assignment_states ===")
    print(f"states={summary['n_states']}  instances={manifest['num_instances_rolled']}")
    print(f"replan_reason={summary['replan_reason']}")
    print(f"n_pending_bins={summary['n_pending_bins']}")
    print(f"saved: {prefix}.npz + {prefix}_audit.jsonl + manifest.json")


if __name__ == '__main__':
    main()
