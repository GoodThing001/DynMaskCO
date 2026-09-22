"""输入一致性检查：同一 TRAIN context，训练路径 vs 在线/诊断路径的 context 特征逐字段对比。

训练路径 = extract_context_features(JSON snapshot)；在线路径 = _ctx_summary(restore→baseline planner
→ live vehicles)。二者应只在「时间点」上有预期差异（idle 车 ready_time 由 prepare_decision_point
抬到 clock），其余字段应一致。

用法（服务器）：
    python scripts/evaluation/run_input_consistency.py \
        --teacher-dir results/m0dev/train --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
        --max-contexts 4 --out results/m0dev/input_consistency
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import extract_context_features
from coldchain_utility_head import feature_only_context
from dynmaskco_cc_context import compute_incumbent_plans, cc_state_dict
from jf1h_repair import make_continuation
from strict_online_env import StrictOnlineEnv
from coldchain_contract import default_pilot_contract


def _train_context(ds, data_npz, ctx):
    snap = ctx['snapshot']
    feats = extract_context_features(data_npz, int(ctx['inst_idx']), snap)
    import jax.numpy as jnp
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid']))), feats


def _online_context(env, ctx):
    inst_idx = int(ctx['inst_idx'])
    cont = make_continuation()
    plans, vehicles, replan_ids = compute_incumbent_plans(env, ctx['snapshot'], cont)
    vis = [int(c) for c in ctx['snapshot']['customer_universe']
           if bool(ctx['snapshot']['visible_mask'][int(c)])]
    snap = {
        'num_vehicles': env.num_vehicles,
        'visible_mask': np.asarray(ctx['snapshot']['visible_mask'], bool),
        'vehicle_node': np.array([v.current_node for v in vehicles], np.int32),
        'vehicle_ready': np.array([v.ready_time for v in vehicles], np.float64),
        'vehicle_load': np.array([v.current_load for v in vehicles], np.float64),
        'needs_replan': np.array([v.needs_replan for v in vehicles], bool),
        'committed_next': np.array([-1 if v.committed_next is None else v.committed_next
                                    for v in vehicles], np.int32),
        'committed_arrive': np.array([np.nan if v.committed_arrive is None else v.committed_arrive
                                      for v in vehicles], np.float64),
        'committed_finish': np.array([np.nan if v.committed_finish is None else v.committed_finish
                                      for v in vehicles], np.float64),
        'vehicle_coldchain_state': [cc_state_dict(v.coldchain_state) for v in vehicles],
    }
    feats = extract_context_features(_npz_view(env, inst_idx), 0, snap)
    import jax.numpy as jnp
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid']))), feats


def _npz_view(env, inst_idx):
    return {'coords': env.coords[inst_idx:inst_idx+1], 'demands': env.demands[inst_idx:inst_idx+1],
            'tw_start': env.tw_start[inst_idx:inst_idx+1], 'tw_end': env.tw_end[inst_idx:inst_idx+1],
            'service_time': env.service_time[inst_idx:inst_idx+1],
            'temp_class': env.temp_class[inst_idx:inst_idx+1],
            'initial_quality': env.initial_quality[inst_idx:inst_idx+1],
            'reveal_time': env.reveal_time[inst_idx:inst_idx+1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-contexts', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    env = StrictOnlineEnv(data_npz, 50.0, 1.0, 25, replanner=make_continuation(),
                          coldchain_contract=default_pilot_contract())

    rows = []
    for ctx in ds.contexts[:args.max_contexts]:
        t_ctx, t_feats = _train_context(ds, data_npz, ctx)
        o_ctx, o_feats = _online_context(env, ctx)
        diff = np.abs(t_ctx - o_ctx)
        rows.append({
            'context_id': ctx['context_id'][:12],
            'inst_idx': int(ctx['inst_idx']),
            'event_id': int(ctx['event_id']),
            'max_abs_diff': float(diff.max()),
            'mean_abs_diff': float(diff.mean()),
            'n_fields_diff_gt_1e-6': int((diff > 1e-6).sum()),
            'ready_diff': float(np.abs(np.asarray(t_feats['fleet_feats'])[:, 1]
                                       - np.asarray(o_feats['fleet_feats'])[:, 1]).max()),
        })
        print(f"  ctx {ctx['context_id'][:12]} inst{ctx['inst_idx']}/evt{ctx['event_id']} "
              f"max_diff={diff.max():.3e} n_diff={int((diff > 1e-6).sum())} "
              f"ready_diff={rows[-1]['ready_diff']:.3e}", flush=True)

    with open(os.path.join(args.out, 'input_consistency.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"saved: {args.out}/input_consistency.json")


if __name__ == '__main__':
    main()
