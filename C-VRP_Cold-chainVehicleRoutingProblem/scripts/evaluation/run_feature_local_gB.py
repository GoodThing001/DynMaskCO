"""ĝ vs g_B 反事实归因（⑤ 归因诊断第二步）。

对每条 FULL 轨迹取第 1/5/10 次通过认证的接受（存在才采样），恢复该决策点的完整状态与
P_before/P_after，两条分支都接同一 JF1-H-F continuation 到终局：

    g_B = J(KEEP 后接 baseline) − J(ACTION 后接 baseline)

比较 ĝ（评分器预测 s·(f(a)−f(keep))）与 g_B，记录原始/归一化 D/Q/E 差。

关键：对第 5/10 次修改，P_before 已含此前接受的修改，**不能**从事件原始 snapshot 调
rollout_baseline()（会重建原 incumbent 抹掉之前修改）；必须显式安装记录下的 P_before/P_after
（_snapshot_with_force 写 force_suffix）。

用法（服务器，CPU）：
    python scripts/evaluation/run_feature_local_gB.py \
        --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --deindex --tau 0.005 --max-instances 4 --out results/m0_scale/gB_F_H_s42
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from feature_local_replanner import (FeatureLocalReplanner, PrePrepareContextCollector,
                                     load_feature_local_scorer)
from counterfactual_teacher import (_snapshot_with_force, _mutable_ids_from_snapshot,
                                    rollout_baseline)
from action_contract import VehiclePlan
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)


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


def _dict_to_plans(d):
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), 0.0, 0.0,
                                  tuple(int(x) for x in v['suffix']))
            for vid, v in d.items()}


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=4)
    ap.add_argument('--tau', type=float, default=0.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--sample-ranks', type=str, default='1,5,10')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff_contract = (apply_objective_profile(contract, profile)
                    if args.objective == 'coldchain' else None)
    scorer = load_feature_local_scorer(args.ckpt)
    ranks = [int(x) for x in args.sample_ranks.split(',')]

    # 独立 roll-out env（含 coldchain contract）
    roll_env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff_contract)

    rows = []
    for i in range(n):
        collector = PrePrepareContextCollector(deindex=args.deindex, save_snapshot=True)
        replanner = FeatureLocalReplanner(scorer=scorer, K=1, tau=args.tau,
                                          capacity=args.capacity, deindex=args.deindex,
                                          context_source=collector, max_total_accepts=None)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff_contract)
        menv.snapshot_hook = collector.hook
        menv.run(i)

        accepts = [d for d in replanner.decision_log if d['result'] == 'accept']
        for rank in ranks:
            if len(accepts) < rank:
                continue
            d = accepts[rank - 1]
            key = (d['instance'], d['event'], d['clock'])
            if key not in collector.snapshot_cache:
                rows.append({'instance': i, 'rank': rank, 'error': '缺快照', 'g_hat': d['g_hat']})
                continue
            snapshot = collector.snapshot_cache[key]
            P_before = _dict_to_plans(d['P_before'])
            P_after = _dict_to_plans(d['P_after'])
            mutable_ids = _mutable_ids_from_snapshot(snapshot)
            try:
                snap_keep = _snapshot_with_force(roll_env, snapshot, P_before,
                                                 mutable_ids=mutable_ids)
                keep = rollout_baseline(roll_env, snap_keep, args.objective)
                snap_action = _snapshot_with_force(roll_env, snapshot, P_after,
                                                   mutable_ids=mutable_ids)
                act = rollout_baseline(roll_env, snap_action, args.objective)
                g_B = _cost(keep, args.objective) - _cost(act, args.objective)
                # 断言：force_suffix 是否区分了 KEEP/ACTION（重建正确性）
                fs_diff = snap_keep['force_suffix'] != snap_action['force_suffix']
                # 断言：两条分支在决策点写回的计划是否不同（force_suffix 是否生效）
                rows.append({'instance': i, 'rank': rank, 'event': d['event'], 'seq': d['seq'],
                             'customer': d['customer'], 'g_hat': d['g_hat'], 'g_B': g_B,
                             'force_suffix_diff': bool(fs_diff),
                             'dD': float(keep['distance_cost'] - act['distance_cost']),
                             'dQ': float(keep['quality_loss'] - act['quality_loss']),
                             'dE': float(keep['energy_kwh'] - act['energy_kwh']),
                             'keep_service': bool(keep.get('complete')),
                             'action_service': bool(act.get('complete'))})
                print(f"  [inst {i} rank {rank}] ĝ={d['g_hat']:+.4f} g_B={g_B:+.4f} "
                      f"dD={rows[-1]['dD']:+.3f} dQ={rows[-1]['dQ']:+.3f} "
                      f"dE={rows[-1]['dE']:+.2f}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({'instance': i, 'rank': rank, 'error': repr(e),
                             'g_hat': d['g_hat']})

    summary = {'rows': rows, 'n': len(rows),
               'n_error': sum(1 for r in rows if 'error' in r),
               'g_hat_mean': float(np.mean([r['g_hat'] for r in rows if 'error' not in r])),
               'g_B_mean': float(np.mean([r['g_B'] for r in rows if 'error' not in r]))}
    with open(os.path.join(args.out, 'gB.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ĝ_mean={summary['g_hat_mean']:+.4f} g_B_mean={summary['g_B_mean']:+.4f} "
          f"n={summary['n']} err={summary['n_error']}")
    print(f"saved: {args.out}/gB.json")


if __name__ == '__main__':
    main()
