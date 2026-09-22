"""⑤ 执行轨迹核查（Step A）：追踪 KEEP/ACTION 两分支计划差异在哪一步消失。

对选定采样点（inst:rank），恢复决策点，跑 KEEP（P_before）与 ACTION（P_after）两条分支，
在每次决策点记录 (event_id, is_reveal, plan_hash)，找出两分支计划 hash 第一次收敛的事件，
并记录该事件是否为 reveal（reveal 清尾 vs baseline 重规划）。

用法（服务器，CPU）：
    python scripts/evaluation/run_feature_local_trace.py \
        --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --deindex --tau 0.005 --select 0:1,0:5,1:1 --out results/m0_scale/trace_F_H_s42
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
from action_contract import VehiclePlan, build_vehicle_plans, plan_hash
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


def _trace_events(dataset, capacity, num_vehicles, eff, inst, snap):
    """跑 run_resumed，逐决策点记录 (event, reveal, plan_hash)。每分支新建 env，避免状态泄漏。"""
    env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                          replanner=make_continuation(), coldchain_contract=eff)
    events = []
    def hook(e, inst_idx, clk, eid, rid, veh, tr, sm, ac):
        is_reveal = any(abs(float(e.reveal_time[inst_idx, c]) - float(clk)) < 1e-6
                        for c in ac if float(e.reveal_time[inst_idx, c]) > 0)
        ph = plan_hash(build_vehicle_plans(e, inst_idx, veh))
        events.append({'event': int(eid), 'reveal': bool(is_reveal), 'plan_hash': ph})
    env.snapshot_hook = hook
    env.run_resumed(snap)
    env.snapshot_hook = None
    return events


def _first_convergence(keep_ev, act_ev):
    n = min(len(keep_ev), len(act_ev))
    first_diff = None
    first_same = None
    for j in range(n):
        if keep_ev[j]['plan_hash'] != act_ev[j]['plan_hash']:
            if first_diff is None:
                first_diff = j
        else:
            if first_same is None and first_diff is not None:
                first_same = j
    return first_diff, first_same, keep_ev, act_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--tau', type=float, default=0.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--select', required=True, help='逗号分隔 inst:rank 对，如 0:1,0:5,1:1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None
    scorer = load_feature_local_scorer(args.ckpt)
    sel = [tuple(int(x) for x in p.split(':')) for p in args.select.split(',')]

    roll_env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)

    rows = []
    # 按实例分组跑 FULL（每个实例只跑一次）
    insts = sorted(set(i for i, _ in sel))
    for inst in insts:
        collector = PrePrepareContextCollector(deindex=args.deindex, save_snapshot=True)
        replanner = FeatureLocalReplanner(scorer=scorer, K=1, tau=args.tau,
                                          capacity=args.capacity, deindex=args.deindex,
                                          context_source=collector, max_total_accepts=None)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        menv.run(inst)
        accepts = [d for d in replanner.decision_log if d['result'] == 'accept']
        for i, rank in [(i, r) for i, r in sel if i == inst]:
            if len(accepts) < rank:
                rows.append({'inst': inst, 'rank': rank, 'error': '缺 accept'})
                continue
            d = accepts[rank - 1]
            key = (d['instance'], d['event'], d['clock'])
            if key not in collector.snapshot_cache:
                rows.append({'inst': inst, 'rank': rank, 'error': '缺快照'})
                continue
            snapshot = collector.snapshot_cache[key]
            P_before = _dict_to_plans(d['P_before'])
            P_after = _dict_to_plans(d['P_after'])
            mutable_ids = _mutable_ids_from_snapshot(snapshot)
            snap_keep = _snapshot_with_force(roll_env, snapshot, P_before, mutable_ids=mutable_ids)
            snap_action = _snapshot_with_force(roll_env, snapshot, P_after, mutable_ids=mutable_ids)
            keep_ev = _trace_events(dataset, args.capacity, args.num_vehicles, eff, inst, snap_keep)
            act_ev = _trace_events(dataset, args.capacity, args.num_vehicles, eff, inst, snap_action)
            first_diff, first_same, ke, ae = _first_convergence(keep_ev, act_ev)
            # 计划差异在哪个事件消失，是否为 reveal
            converge_event = ke[first_same]['event'] if first_same is not None else None
            converge_reveal = ke[first_same]['reveal'] if first_same is not None else None
            rows.append({'inst': inst, 'rank': rank, 'event': d['event'], 'customer': d['customer'],
                         'g_hat': d['g_hat'], 'n_events_keep': len(keep_ev),
                         'n_events_act': len(act_ev),
                         'first_diff_at': first_diff, 'first_same_at': first_same,
                         'converge_event': converge_event, 'converge_is_reveal': converge_reveal,
                         'never_converged': first_same is None})
            print(f"  [inst{inst} rank{rank}] n_events={len(keep_ev)}/{len(act_ev)} "
                  f"first_diff@event{first_diff} converge@event{converge_event} "
                  f"reveal={converge_reveal}", flush=True)

    summary = {'rows': rows}
    with open(os.path.join(args.out, 'trace.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {args.out}/trace.json")


if __name__ == '__main__':
    main()
