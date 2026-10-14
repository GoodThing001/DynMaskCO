"""②a · 动作执行追踪：mask=2 局部修复的修改是被执行还是被后续重规划覆盖。

回答 D3 的机制问题：myopic 头腔（D1/D2 ~0.006–0.013）为何不进入终局（D3 终局头腔 ≈0）？
区分两种机制：
  - M1 动作未执行：修改被后续重规划覆盖（客户没按候选方案被服务）；
  - M2 执行了但无终局收益：修改按候选执行，但未来订单把收益洗掉。

做法：复用 run_headroom_census.Probe（mask=2）采集状态，对每个状态：
  1. 距离贪心修复 P_dg（run_headroom_census._dist_greedy_repair）与穷举修复 P_exh；
  2. _snapshot_with_force + run_resumed 得终局 traces（每车 services[].node=实际服务顺序）；
  3. 比对候选分配 cand_assign（mask_set 客户在候选 plan 里被分配给哪辆车）vs 实际执行
     trace_assign（mask_set 客户实际被哪辆车服务）；
  4. executed（mask_set 全部按候选执行）vs overwritten（至少一个被改派/改序）。

用法（本地，纯 NumPy）：
    PYTHONPATH=scripts python negative_result_paper/scripts/run_execution_trace.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 16 --out negative_result_paper/results/execution_trace_cal
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from recourse_snapshot import capture_recourse_snapshot, resume_from_snapshot
from counterfactual_teacher import _eval, _snapshot_with_force
from coldchain_contract import default_pilot_contract, apply_objective_profile
from run_headroom_census import Probe, _load_profile, _dist_greedy_repair, _exhaustive_repair


def _terminal_baseline(renv, snapshot):
    traces, served = renv.run_resumed(snapshot)
    return _eval(renv, int(snapshot['instance_id']), traces, 'coldchain', served_mask=served)


def _terminal_plan_with_traces(renv, snapshot, plan, mutable_ids):
    snap2 = _snapshot_with_force(renv, snapshot, plan, mutable_ids=mutable_ids)
    traces, served = renv.run_resumed(snap2)
    out = _eval(renv, int(snapshot['instance_id']), traces, 'coldchain', served_mask=served)
    return out, traces


def _candidate_assignment(plan):
    assign = {}
    for vid, p in plan.items():
        for c in p.suffix:
            if int(c) > 0:
                assign[int(c)] = int(vid)
    return assign


def _trace_assignment(traces):
    assign = {}
    for vid, tr in enumerate(traces):
        for sr in getattr(tr, 'services', []):
            node = int(sr.node)
            if node > 0:
                assign[node] = int(vid)
    return assign


def _persistence(mask_set, cand_assign, trace_assign):
    pers = {}
    for c in mask_set:
        c = int(c)
        cv = cand_assign.get(c)
        tv = trace_assign.get(c)
        pers[c] = (cv is not None and tv is not None and cv == tv)
    return pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective

    snapshots = {}
    per_inst_cap = {}

    def snap_hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        cnt = per_inst_cap.get(int(inst), 0)
        if cnt >= 2:
            return
        key = (int(inst), int(eid))
        if key not in snapshots:
            snapshots[key] = capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac)
            per_inst_cap[int(inst)] = cnt + 1

    probe = Probe(contract, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.snapshot_hook = snap_hook
        env.run(inst)
    states = probe.states
    print(f'collected {len(states)} states', flush=True)

    rows = []
    for st in states:
        key = (st['inst'], st['event'])
        snap = snapshots.get(key)
        if snap is None:
            continue
        inst_idx = st['inst']
        env = st['env']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']

        P_dg = _dist_greedy_repair(env, inst_idx, partial, mask_set, mutable_ids)
        P_exh, J_exh, _ = _exhaustive_repair(env, inst_idx, partial, mask_set,
                                             mutable_ids, st['vis'], contract, objective)

        renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity, 1.0,
                                    args.num_vehicles, coldchain_contract=contract)
        o0 = _terminal_baseline(renv, snap)
        J0_term = float(o0['coldchain_cost'])

        def _row_for(plan, tag):
            if plan is None:
                return None
            cand_assign = _candidate_assignment(plan)
            cand_pos = {}
            for c in mask_set:
                cv = cand_assign.get(int(c))
                if cv is not None:
                    p = plan.get(cv)
                    if p is not None and int(c) in p.suffix:
                        cand_pos[str(c)] = int(p.suffix.index(int(c)))
            out, traces = _terminal_plan_with_traces(renv, snap, plan, mutable_ids)
            trace_assign = _trace_assignment(traces)
            pers = _persistence(mask_set, cand_assign, trace_assign)
            n_persist = sum(1 for v in pers.values() if v)
            executed = (n_persist == len(mask_set))
            headroom = J0_term - float(out['coldchain_cost'])
            return {
                'n_mask': len(mask_set), 'n_persisted': n_persist,
                'persisted': {str(k): v for k, v in pers.items()},
                'executed': executed,
                'terminal_headroom': headroom,
                'cand_assign': {str(k): v for k, v in cand_assign.items()
                                if k in mask_set},
                'cand_pos': cand_pos,
                'trace_assign': {str(k): v for k, v in trace_assign.items()
                                 if k in mask_set},
            }

        dg = _row_for(P_dg, 'dg')
        ex = _row_for(P_exh, 'exh')
        row = {
            'inst': inst_idx, 'event': st['event'],
            'myopic_J0': st['J0'], 'terminal_J0': J0_term,
            'J_exhaustive_myopic': J_exh,
            'dg': dg, 'exh': ex,
        }
        rows.append(row)
        dg_s = 'NA' if dg is None else f"persist={dg['n_persisted']}/{dg['n_mask']} headroom={dg['terminal_headroom']:+.4f}"
        print(f"  [inst {inst_idx}/evt {st['event']}] dg {dg_s}", flush=True)

    def _clustered(vals):
        per = {}
        for r in rows:
            for v in vals(r):
                if v is None:
                    continue
                per.setdefault(r['inst'], []).append(v)
        return [float(np.mean(x)) for x in per.values()]

    def _ci(vals, n_boot=2000, seed=0):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(seed)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(n_boot)]
        return {'mean': mean, 'ci_lo': float(np.percentile(b, 2.5)),
                'ci_hi': float(np.percentile(b, 97.5))}

    def _stat(tag):
        executed = [r[tag]['executed'] for r in rows if r.get(tag)]
        n_exec = sum(1 for x in executed if x)
        headroom_all = [r[tag]['terminal_headroom'] for r in rows if r.get(tag)]
        headroom_exec = [r[tag]['terminal_headroom'] for r in rows
                         if r.get(tag) and r[tag]['executed']]
        headroom_over = [r[tag]['terminal_headroom'] for r in rows
                         if r.get(tag) and not r[tag]['executed']]
        return {
            'n_states': len(executed),
            'n_executed': n_exec,
            'executed_frac': (n_exec / len(executed)) if executed else None,
            'headroom_all': _ci(headroom_all),
            'headroom_executed': _ci(headroom_exec),
            'headroom_overwritten': _ci(headroom_over),
        }

    summary = {
        'n_states': len(rows), 'n_instances': len(set(r['inst'] for r in rows)),
        'dist_greedy': _stat('dg'),
        'exhaustive': _stat('exh'),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n=== execution-trace persistence (mask=2 repair) ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
