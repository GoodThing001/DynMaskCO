"""mask 范围审计：重新采集训练决策点（不训练），统计越界 mask。

回答点 #4：训练 / 固定状态脚本从「所有可见未服务客户」选 mask
（pool = visible & unserved），是否越界选了 committed / 非可变车 tail 客户——即偏离
decision_pool_from_vehicles 口径（后者额外排除 committed 客户与 non-replan 车 tail 客户）。

只采集 + 审计，不做任何训练。默认复现训练 CollectProbe（max_per_instance=4, 64 实例）。
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile)
from train_mpre_reinforce import CollectProbe
from run_fixed_state_quality import _audit_mask_scope


def _load_profile(path):
    with open(path) as f:
        d = json.load(f)
    if d.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                            quality_scale=float(d['quality_scale']),
                            energy_scale=float(d['energy_scale']),
                            lambda_quality=float(d['lambda_quality']),
                            lambda_energy=float(d['lambda_energy']),
                            scale_source=d.get('scale_source', 'pilot'),
                            dev_statistics=d.get('dev_statistics'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=64)
    ap.add_argument('--max-per-instance', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    eff = apply_objective_profile(default_pilot_contract(), profile)

    collector = CollectProbe(eff, args.capacity, max_per_instance=args.max_per_instance)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=collector, coldchain_contract=eff)
        env.run(inst)
    pool = collector.states
    print(f"collected {len(pool)} states over {args.max_instances} instances", flush=True)

    rows = []
    for st in pool:
        audit = _audit_mask_scope(st['vehicles'], st['served_mask'], st['visible_ids'],
                                  st['P0'], list(st['mask_set']), st['mutable_ids'], set())
        rows.append({'inst': st['inst'], 'event': st['event'], 'audit': audit})

    n_states_out = 0
    n_cust_out = 0
    reasons = Counter()
    by_event = {}
    for r in rows:
        a = r['audit']
        ev = int(r['event'])
        by_event.setdefault(ev, {'n_states': 0, 'n_out': 0, 'n_cust_out': 0})
        by_event[ev]['n_states'] += 1
        if a['n_out_of_pool'] > 0:
            n_states_out += 1
            n_cust_out += a['n_out_of_pool']
            by_event[ev]['n_out'] += 1
            by_event[ev]['n_cust_out'] += a['n_out_of_pool']
        for _c, info in a['per_customer'].items():
            if not info['in_decision_pool']:
                reasons['out_of_decision_pool'] += 1
            if info['is_committed_next']:
                reasons['is_committed_next'] += 1
            if info['in_non_mutable_tail']:
                reasons['in_non_mutable_tail'] += 1
            if info['in_protected_tail']:
                reasons['in_protected_tail'] += 1
            if not info['in_mutable_vehicle']:
                reasons['not_in_mutable_vehicle'] += 1

    inst_dist = Counter(r['inst'] for r in rows)
    summary = {
        'n_states': len(rows),
        'n_instances_covered': len(inst_dist),
        'inst_dist': dict(sorted(inst_dist.items())),
        'n_states_with_out_of_pool': n_states_out,
        'n_out_of_pool_customers': n_cust_out,
        'out_of_pool_state_rate': float(n_states_out / max(len(rows), 1)),
        'reason_counts': dict(reasons),
        'by_event': by_event,
    }
    with open(os.path.join(args.out, 'mask_audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'mask_audit_per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out}/mask_audit_summary.json")


if __name__ == '__main__':
    main()
