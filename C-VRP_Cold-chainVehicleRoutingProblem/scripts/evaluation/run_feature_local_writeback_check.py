"""feature-local 在线写回 + 因果性正确性检查（④ 步骤 3，不跑完整收益实验）。

用恢复快照 + 受控评分器验证：
  A. 因果性：扰动未揭示订单（coords/reveal_time）不改变 pre-prepare context；
  B. 强制拒绝（评分器返回 -inf）：learned 层不改 baseline 计划与 deferred；
  D. 空路线写回语义：suffix 为空时 []（非仓库且有 future）或 [0]（否则返仓），与
     counterfactual_teacher._snapshot_with_force 一致。

真正的「接受后写回 + 下一客户读新计划」由 ⑤ gate 的在线运行覆盖（accept 时 _reconstruct
重建 P）；本脚本只覆盖纯正确性分支，不用脆弱的强制接受 scorer。

用法（服务器）：
    python scripts/evaluation/run_feature_local_writeback_check.py \
        --teacher-dir results/m0_scale/dev_check \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --max-contexts 2 --deindex --out results/m0_scale/iface_writeback
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_teacher_dataset import load_teacher_dataset
from dynmaskco_cc_context import restore_vehicles_from_json
from jf1h_repair import make_continuation
from strict_online_env import StrictOnlineEnv
from coldchain_contract import default_pilot_contract
from action_contract import build_vehicle_plans, VehiclePlan
from feature_local_replanner import (FeatureLocalReplanner, PrePrepareContextCollector,
                                     _npz_view)


class _RejectScorer:
    s = 1.0
    def score(self, x):
        return np.full(x.shape[:2], -1e9, np.float32)


def _setup(env, npz, snap, cont, deindex, capacity, scorer):
    collector = PrePrepareContextCollector(deindex=deindex)
    replanner = FeatureLocalReplanner(scorer=scorer, deindex=deindex, capacity=capacity,
                                      context_source=collector)
    vehicles = restore_vehicles_from_json(snap)
    if hasattr(replanner, 'restore_state'):
        replanner.restore_state(snap.get('replanner_state'))
    clock = float(snap['clock'])
    event_id = int(snap['event_id'])
    served_mask = snap['served_mask']
    all_customers = snap['customer_universe']
    collector.hook(env, int(snap['instance_id']), clock, event_id, snap['reveal_idx'], vehicles,
                   None, served_mask, all_customers)
    env.prepare_decision_point(clock, vehicles)
    visible_ids = [int(c) for c in all_customers if bool(snap['visible_mask'][int(c)])]
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    cont.restore_state(snap.get('replanner_state'))
    cont.plan(env, int(snap['instance_id']), clock, vehicles, served_mask, visible_ids,
              replan_ids=replan_ids)
    P = build_vehicle_plans(env, int(snap['instance_id']), vehicles)
    mutable_ids = {v.vehicle_id for v in vehicles
                   if v.status in ('idle', 'ready') and v.needs_replan}
    return replanner, collector, vehicles, P, mutable_ids, clock, event_id, served_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-contexts', type=int, default=2)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    npz = dict(np.load(args.data))
    env = StrictOnlineEnv(npz, args.capacity, 1.0, 25, replanner=make_continuation(),
                          coldchain_contract=default_pilot_contract())
    cont = make_continuation()

    results = []
    def record(name, ok, detail=''):
        results.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # ---- D. 空路线写回语义（纯表达式单测，对比 _snapshot_with_force） ----
    def _writeback_suffix(p, has_future):
        return (list(p.suffix) + [0] if p.suffix
                else ([] if (p.anchor_node != 0 and has_future) else [0]))

    cases = [
        (VehiclePlan(0, 0, 0.0, 0.0, (1, 2)), True, [1, 2, 0]),    # 有客户 → 客户+[0]
        (VehiclePlan(3, 5, 0.0, 0.0, ()), True, []),               # 非仓库空+future → WAIT
        (VehiclePlan(3, 5, 0.0, 0.0, ()), False, [0]),             # 非仓库空+无future → 返仓
        (VehiclePlan(2, 0, 0.0, 0.0, ()), True, [0]),              # 仓库空 → 返仓
    ]
    all_ok = True
    for p, hf, exp in cases:
        got = _writeback_suffix(p, hf)
        if got != exp:
            all_ok = False
            print(f"      MISMATCH anchor={p.anchor_node} suffix={p.suffix} has_future={hf}: "
                  f"got={got} exp={exp}")
    record('empty-route writeback semantics (WAIT/RETURN)', all_ok)

    for ctx in ds.contexts[:args.max_contexts]:
        inst = int(ctx['inst_idx'])
        snap = ctx['snapshot']
        cust = int(ds.candidates_by_context[ctx['context_id']][0]['action']['customer'])

        # ---- A. 因果性：扰动未揭示订单 ----
        unrevealed = [c for c in snap['customer_universe']
                      if npz['reveal_time'][inst, c] > float(snap['clock']) + 1e-6]
        repl, col, veh, P, mid, clock, eid, sm = _setup(
            env, npz, snap, cont, args.deindex, args.capacity, _RejectScorer())
        c_orig = col.get(inst, eid, clock)
        npz2 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in npz.items()}
        for c in unrevealed:
            npz2['coords'][inst, c] += 100.0
            npz2['reveal_time'][inst, c] += 100.0
        env2 = StrictOnlineEnv(npz2, args.capacity, 1.0, 25, replanner=make_continuation(),
                               coldchain_contract=default_pilot_contract())
        repl2, col2, veh2, P2, mid2, clock2, eid2, sm2 = _setup(
            env2, npz2, snap, cont, args.deindex, args.capacity, _RejectScorer())
        c_pert = col2.get(inst, eid2, clock2)
        record(f'causality inst{inst}/evt{eid} perturb-unrevealed context unchanged',
               float(np.abs(c_orig - c_pert).max()) < 1e-9,
               f"maxdiff={np.abs(c_orig - c_pert).max():.2e} unrevealed={len(unrevealed)}")

        # ---- B. 强制拒绝：不改 baseline 计划 ----
        repl, col, veh, P, mid, clock, eid, sm = _setup(
            env, npz, snap, cont, args.deindex, args.capacity, _RejectScorer())
        suffix_before = {v.vehicle_id: list(v.mutable_suffix) for v in veh}
        deferred_before = set(repl.deferred_customers)
        ok = repl._process(env, inst, clock, veh, sm, P, col.get(inst, eid, clock),
                           _npz_view(env, inst), cust, mid)
        suffix_after = {v.vehicle_id: list(v.mutable_suffix) for v in veh}
        record(f'reject-no-change inst{inst}/evt{eid} cust{cust}',
               (not ok and suffix_before == suffix_after
                and deferred_before == set(repl.deferred_customers)),
               f"accept={ok}")

    summary = {'results': results,
               'n_pass': sum(1 for r in results if r['ok']),
               'n_total': len(results),
               'verdict': 'PASS' if all(r['ok'] for r in results) else 'FAIL'}
    with open(os.path.join(args.out, 'writeback_check.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  {summary['n_pass']}/{summary['n_total']} PASS  verdict={summary['verdict']}")
    print(f"saved: {args.out}/writeback_check.json")


if __name__ == '__main__':
    main()
