"""D7-a · 易腐排序头腔（myopic，无训练/无模型/纯 NumPy）。

回答：当前合同下，把每辆车 suffix 从"距离贪心顺序"重排为"按易腐权重升序（易腐订单后取，
缩短其车载时间）"，能否在 myopic J_vis 上产生正收益。若连 myopic 都 ≈0（甚至负），说明
当前品质/能耗项的物理参数（衰减率 ~0.001–0.0035/h）太弱、不足以奖励冷链排序——结构改动
（分舱指派）也不会奏效，必须先解决"参数太弱"这个前置问题。

易腐权重（决策时可见）：w(c) = product_value[class] × quantity × reference_rate_per_hour[class]。
pickup-to-depot 下"易腐后取"：suffix 按 w 升序（低易腐先取、高易腐靠近返仓）。

用法：
    python scripts/evaluation/run_perishability_headroom.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --split train --out results/m0_scale/perishability_headroom_train
"""
import argparse
import copy
import json
import os
import sys
from dataclasses import replace as dc_replace

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from action_contract import build_vehicle_plans, apply_action
from cc_lns_replanner import _plan_suffixes
from mpre_policy import enumerate_legal_actions
from visible_state import evaluate_visible_plan
from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from run_headroom_census import Probe, _load_profile, _plan_copy, _eval_plan


def _plan_from_suffixes(partial_template, suffixes):
    """用 per-vid 的客户列表重建 plan（复用 partial 的锚点/时间/载重）。"""
    out = {}
    for vid, p in partial_template.items():
        suf = tuple(int(x) for x in suffixes.get(vid, ()))
        out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, suf)
    return out


def scale_quality_physics(contract, rate_scale=1.0, value_scale=1.0):
    """按因子缩放品质物理（衰减率/product_value），放宽对应 provenance 区间以便 validate。

    这是敏感性扫描用：pilot 占位参数（衰减率 ~0.001–0.0035/h、product_value 1.0/1.5/2.0）
    太慢，无法支撑冷链排序。此处不碰 lambda_quality/lambda_energy、不碰任何硬约束。
    """
    q = contract.quality
    new_q = dc_replace(
        q,
        reference_rate_per_hour=tuple(r * rate_scale for r in q.reference_rate_per_hour),
        product_value=tuple(v * value_scale for v in q.product_value),
    )
    widen = {'quality.reference_rate_per_hour', 'quality.product_value'}
    new_pp = tuple(
        dc_replace(p, sensitivity_low=0.0, sensitivity_high=1e9)
        if p.parameter_path in widen else p
        for p in contract.parameter_provenance
    )
    return dc_replace(contract, quality=new_q, parameter_provenance=new_pp)


def _perishability_w(env, inst_idx, contract, customer):
    c = int(customer)
    cls = int(env.temp_class[inst_idx, c])
    value = contract.quality.product_value[cls]
    rate = contract.quality.reference_rate_per_hour[cls]
    qty = float(env.demands[inst_idx, c])
    return value * qty * rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--rate-scale', type=float, default=1.0)
    ap.add_argument('--value-scale', type=float, default=1.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = apply_objective_profile(default_pilot_contract(), profile)
    contract = scale_quality_physics(contract, args.rate_scale, args.value_scale)
    objective = contract.objective

    probe = Probe(contract, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=contract)
        env.run(inst)
    states = probe.states
    print(f'collected {len(states)} states over {args.max_instances} instances', flush=True)

    rows = []
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']
        P0 = st['P0']

        # 距离贪心修复（P_dg）
        P = _plan_copy(partial)
        remaining = list(mask_set)
        while remaining:
            legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
            if not legal:
                P = None
                break
            legal.sort(key=lambda t: (t[2].incremental_distance, t[1].action_id()))
            c, a, _cand = legal[0]
            P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
            remaining.remove(c)
        P_dg = P
        r_dg = _eval_plan(vis, P_dg, contract, objective) if P_dg is not None else None

        # 易腐重排：同一 (customer, vehicle) 指派，只重排每车 suffix 顺序
        if P_dg is not None:
            suf = _plan_suffixes(P_dg)
            reordered = {}
            for vid, custs in suf.items():
                reordered[vid] = tuple(sorted(custs, key=lambda c: _perishability_w(env, inst_idx, contract, c)))
            P_perish = _plan_from_suffixes(partial, reordered)
            r_perish = _eval_plan(vis, P_perish, contract, objective)
        else:
            P_perish = None
            r_perish = None

        gap = None
        if r_dg is not None and r_perish is not None:
            gap = float(r_dg.J_vis) - float(r_perish.J_vis)  # >0 = 重排改善
        rows.append({
            'inst': inst_idx, 'event': st['event'],
            'J_dg': float(r_dg.J_vis) if r_dg is not None else None,
            'J_perish': float(r_perish.J_vis) if r_perish is not None else None,
            'gap': gap,
            'D_delta': (float(r_dg.D) - float(r_perish.D)) if (r_dg is not None and r_perish is not None) else None,
            'Q_delta': (float(r_dg.Q) - float(r_perish.Q)) if (r_dg is not None and r_perish is not None) else None,
            'E_delta': (float(r_dg.E) - float(r_perish.E)) if (r_dg is not None and r_perish is not None) else None,
        })
        print(f"  [inst {inst_idx}/evt {st['event']}] J_dg={rows[-1]['J_dg'] if rows[-1]['J_dg'] is None else round(rows[-1]['J_dg'],4)} "
              f"J_perish={rows[-1]['J_perish'] if rows[-1]['J_perish'] is None else round(rows[-1]['J_perish'],4)} "
              f"gap={rows[-1]['gap'] if rows[-1]['gap'] is None else round(rows[-1]['gap'],5)}", flush=True)

    gaps = [r['gap'] for r in rows if r['gap'] is not None]
    d_deltas = [r['D_delta'] for r in rows if r['D_delta'] is not None]
    q_deltas = [r['Q_delta'] for r in rows if r['Q_delta'] is not None]
    e_deltas = [r['E_delta'] for r in rows if r['E_delta'] is not None]

    def _inst_agg(vals_by_row, key):
        per = {}
        for r in rows:
            v = r[key]
            if v is None:
                continue
            per.setdefault(r['inst'], []).append(v)
        return [float(np.mean(v)) for v in per.values()]

    def _ci(vals, n_boot=2000, seed=0):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rng = np.random.default_rng(seed)
        b = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), size=len(vals))]))
             for _ in range(n_boot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}

    summary = {
        'split': args.split, 'n_states': len(rows), 'n_instances': len(set(r['inst'] for r in rows)),
        'gap': _ci(_inst_agg(rows, 'gap')),
        'component_delta': {'D': _ci(_inst_agg(rows, 'D_delta')),
                            'Q': _ci(_inst_agg(rows, 'Q_delta')),
                            'E': _ci(_inst_agg(rows, 'E_delta'))},
        'n_gap_positive': int(sum(1 for g in gaps if g > 1e-9)),
        'n_gap_negative': int(sum(1 for g in gaps if g < -1e-9)),
        'per_state_quantiles': {
            'p25': float(np.percentile(gaps, 25)) if gaps else None,
            'p50': float(np.percentile(gaps, 50)) if gaps else None,
            'p90': float(np.percentile(gaps, 90)) if gaps else None,
            'max': float(np.max(gaps)) if gaps else None,
        },
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n=== D7-a perishability-ordering headroom (myopic) ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
