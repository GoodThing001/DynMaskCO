"""N1 teacher 导出：R 轨迹 + 较充分同目标搜索 → 局部修复步骤标签。

从带预留保护的 R 轨迹采集状态，在「prepare + JF1-H-F 形成 P0 + 保护集合之后」运行较充分
regret-2 搜索（8 轮、4s、64 次尝试），逐条导出 `P_before→mask→P_partial→steps→P_target`。

用法（本地/服务器）：
    python scripts/expert/export_repair_teacher.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 64 --max-events-per-instance 6 --teacher-rounds 8 --budget 4.0 \
        --out results/m0_scale/repair_teacher
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from action_contract import build_vehicle_plans
from repair_teacher import collect_event_records


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(
        name=data['name'], distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']), energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']), lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'), dev_statistics=data.get('dev_statistics'))


class TeacherReplanner(JF1HRepairReplanner):
    """JF1-H-F + 预留保护 + 较充分 teacher 搜索，逐事件写回最佳计划并采集步骤标签。"""

    def __init__(self, contract, budget_s=4.0, seed=0, n_rounds=8, collect=True):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.n_rounds = int(n_rounds)
        self.collect = collect
        self.records = []
        self.denom = {'all_events': 0, 'eligible_events': 0, 'records': 0}

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved_ids = set()
        idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
        reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in reserved_ids:
            v = vehicles[vid]
            p = P0.get(vid)
            if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                protected.add(vid)
        self.denom['all_events'] += 1
        records, _stats, best, best_J = collect_event_records(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, self.contract, protected, self.budget_s, self.seed,
            self.n_rounds)
        # 写回最佳计划（与 R 相同 WAIT/RETURN 语义，保护车不改写）
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = best.get(v.vehicle_id)
            if p is None:
                continue
            v.mutable_suffix = (list(p.suffix) + [0] if p.suffix
                                else ([] if (p.anchor_node != 0 and has_future) else [0]))
        self.sync_deferred_from_vehicles(vehicles)
        if records:
            self.denom['eligible_events'] += 1
            self.denom['records'] += len(records)
        if self.collect:
            self.records.extend(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=64)
    ap.add_argument('--max-events-per-instance', type=int, default=6)
    ap.add_argument('--teacher-rounds', type=int, default=8)
    ap.add_argument('--budget', type=float, default=4.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None

    all_records = []
    for inst in range(n):
        rep = TeacherReplanner(eff, budget_s=args.budget, seed=args.seed,
                               n_rounds=args.teacher_rounds, collect=True)
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=rep, coldchain_contract=eff)
        env.run(inst)
        inst_recs = [r for r in rep.records if r['inst'] == inst]
        # 每实例最多 sampled max_events_per_instance 条（按记录顺序 linspace）
        if len(inst_recs) > args.max_events_per_instance:
            idx = np.unique(np.linspace(0, len(inst_recs) - 1, args.max_events_per_instance)
                            .round().astype(int))
            inst_recs = [inst_recs[int(j)] for j in idx]
        all_records.extend(inst_recs)
        print(f"  [inst {inst}] n_records={len(rep.records)} sampled={len(inst_recs)} "
              f"denom={rep.denom}", flush=True)

    summary = {
        'n_instances': n, 'n_records': len(all_records), 'schema': 'repair_teacher_v1',
        'teacher_rounds': args.teacher_rounds, 'budget': args.budget, 'seed': args.seed,
        'guard_reserved': True,
        'objective': args.objective,
        'contract_hash': eff.contract_hash if eff is not None else None,
        'profile_hash': profile.profile_hash if profile is not None else None,
    }
    with open(os.path.join(args.out, 'records.jsonl'), 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  n_records={len(all_records)}")
    print(f"saved: {args.out}/records.jsonl + summary.json")


if __name__ == '__main__':
    main()
