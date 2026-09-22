"""feature-local 70 维在线/离线逐字段对账（④ 检查，真实部署路径 + pre-prepare context）。

对每个 context（snapshot × customer）：
  - 离线 = train_feature_only_local.build_dataset 的构造（从原始 snapshot 提取 context/action/local）；
  - 在线 = restore_vehicles_from_json → PrePrepareContextCollector.hook（pre-prepare context）
    → prepare_decision_point → 基线 continuation 生成 incumbent → 真实 `FeatureLocalReplanner._process`
    （捕获 scorer 拦截 _build_input 的 70 维输入）。
  - 按同一个 customer 分组比较，不把 pool 合并；普通动作/KEEP/DEFER 用同一 `_action_key` 规范化。

严格 verdict：有效检查数>0、每个 context 恰好一个参考动作、无重复 key、无构造错误、无非有限值、
off_only=on_only=0、context/action/local 全满足容差，才标 PASS。

用法（服务器）：
    python scripts/evaluation/run_feature_local_reconcile.py \
        --teacher-dir results/m0_scale/dev_check \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --max-contexts 8 --deindex --out results/m0_scale/iface_reconcile
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
from coldchain_visible_features import (extract_action_features,
                                        extract_candidate_local_features,
                                        extract_context_features)
from coldchain_utility_head import feature_only_context
from dynmaskco_cc_context import restore_vehicles_from_json
from jf1h_repair import make_continuation
from strict_online_env import StrictOnlineEnv
from coldchain_contract import default_pilot_contract
from action_contract import build_vehicle_plans
from feature_local_replanner import (FeatureLocalReplanner, PrePrepareContextCollector,
                                     _action_dict_of, _npz_view)


def _action_key(a):
    """统一规范化动作身份：slot_kind 同时接受 'slot_kind'/'kind'（DEFER 用 kind='defer'）。"""
    return (a.get('customer'), a.get('slot_kind') or a.get('kind'),
            a.get('slot_anchor'), a.get('position'),
            a.get('predecessor'), a.get('successor'),
            bool(a.get('incumbent', False)))


def _ctx_vec(dataset, inst_idx, snap, deindex):
    import jax.numpy as jnp
    feats = extract_context_features(dataset, inst_idx, snap, deindex=deindex)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


class _CaptureScorer:
    def __init__(self):
        self.s = 1.0
        self.inputs = []

    def score(self, x):
        self.inputs.append(np.asarray(x))
        return np.zeros(x.shape[:2], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-contexts', type=int, default=8)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    env = StrictOnlineEnv(data_npz, args.capacity, 1.0, 25, replanner=make_continuation(),
                          coldchain_contract=default_pilot_contract())
    cont = make_continuation()

    rows = []
    n_pairs = 0
    n_err = 0
    for ctx in ds.contexts[:args.max_contexts]:
        inst_idx = int(ctx['inst_idx'])
        cands = ds.candidates_by_context[ctx['context_id']]
        customer = int(cands[0]['action']['customer'])
        snap = ctx['snapshot']
        clock = float(snap['clock'])
        event_id = int(snap['event_id'])

        # ---- 离线（build_dataset 同口径：feasible 或 KEEP/DEFER 伪动作） ----
        off_ctx = _ctx_vec(data_npz, inst_idx, snap, args.deindex)
        off_by_key = {}
        n_ref = 0
        for c in cands:
            a = c.get('action') or {}
            is_pseudo = bool(c.get('is_pseudo', False))
            feasible = bool(c.get('certificate', {}).get('feasible', False))
            if not is_pseudo and not feasible:
                continue
            if is_pseudo:
                n_ref += 1
            key = _action_key(a)
            if key in off_by_key:
                rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                             'customer': customer, 'error': 'offline 重复 action key'})
                n_err += 1
                break
            av, avd = extract_action_features(c, deindex=args.deindex)
            lv, lvd = extract_candidate_local_features(
                data_npz, inst_idx, a, c.get('certificate', {}).get('incremental_distance'),
                args.capacity)
            off_by_key[key] = {'action': (av, avd), 'local': (lv, lvd)}
        if n_ref != 1:
            rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                         'customer': customer, 'error': f'参考动作数={n_ref}（应恰为 1）'})
            n_err += 1
            continue

        # ---- 在线（真实部署路径：collector + prepare + 基线 + _process） ----
        try:
            collector = PrePrepareContextCollector(deindex=args.deindex)
            capture = _CaptureScorer()
            replanner = FeatureLocalReplanner(scorer=capture, deindex=args.deindex,
                                              capacity=args.capacity, context_source=collector)
            vehicles = restore_vehicles_from_json(snap)
            if hasattr(replanner, 'restore_state'):
                replanner.restore_state(snap.get('replanner_state'))
            served_mask = snap['served_mask']
            all_customers = snap['customer_universe']
            # pre-prepare 采集（bump 前）
            collector.hook(env, inst_idx, clock, event_id, snap['reveal_idx'], vehicles, None,
                           served_mask, all_customers)
            # bump + 基线 incumbent
            env.prepare_decision_point(clock, vehicles)
            visible_ids = [int(c) for c in all_customers
                           if bool(snap['visible_mask'][int(c)])]
            replan_ids = {v.vehicle_id for v in vehicles
                          if v.status in ('idle', 'ready') and v.needs_replan}
            cont.restore_state(snap.get('replanner_state'))
            cont.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
            P = build_vehicle_plans(env, inst_idx, vehicles)
            mutable_ids = {v.vehicle_id for v in vehicles
                           if v.status in ('idle', 'ready') and v.needs_replan}

            on_ctx = collector.get(inst_idx, event_id, clock)
            ctx_diff = float(np.abs(off_ctx - on_ctx).max())

            # 单 customer 的真实 _process（含 _build_input + 捕获 scorer）
            replanner._process(env, inst_idx, clock, vehicles, served_mask, P, on_ctx,
                               _npz_view(env, inst_idx), customer, mutable_ids)
            if len(capture.inputs) != 1:
                rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                             'customer': customer,
                             'error': f'_process 捕获 {len(capture.inputs)} 个输入（应 1）'})
                n_err += 1
                continue
            x = capture.inputs[0]                       # [1, M, 70]
            on_by_key = {}
            from action_contract import enumerate_actions_from_plans
            cands_on, _ = enumerate_actions_from_plans(env, inst_idx, P, customer,
                                                       allowed_vehicle_ids=mutable_ids)
            feas = [c for c in cands_on if c.feasible]
            keys = [_action_key(_action_dict_of(c.action)) for c in feas]
            if not any(c.action.incumbent for c in feas):
                keys.append(_action_key({'customer': customer, 'kind': 'defer'}))
            if x.shape[1] != len(keys):
                rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                             'customer': customer, 'error': f'row/keys 数不一致 {x.shape[1]} vs {len(keys)}'})
                n_err += 1
                continue
            for j, key in enumerate(keys):
                if key in on_by_key:
                    rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                                 'customer': customer, 'error': '在线重复 action key'})
                    n_err += 1
                    break
                on_by_key[key] = {
                    'action': (x[0, j, 25:33], x[0, j, 33:41].astype(bool)),
                    'local': (x[0, j, 41:66], x[0, j, 66:70].astype(bool)),
                }
            else:
                # ---- 对账（按 customer，双向集合 + 逐字段） ----
                off_keys = set(off_by_key)
                on_keys = set(on_by_key)
                common = off_keys & on_keys
                act_diff = 0.0
                loc_diff = 0.0
                for key in sorted(common):
                    oa = off_by_key[key]['action']
                    na = on_by_key[key]['action']
                    act_diff = max(act_diff, float(np.abs(oa[0] - na[0]).max()),
                                   float(np.abs(oa[1].astype(float) - na[1].astype(float)).max()))
                    ol = off_by_key[key]['local']
                    nl = on_by_key[key]['local']
                    loc_diff = max(loc_diff, float(np.abs(ol[0] - nl[0]).max()),
                                   float(np.abs(ol[1].astype(float) - nl[1].astype(float)).max()))
                if not np.isfinite(act_diff) or not np.isfinite(loc_diff) or not np.isfinite(ctx_diff):
                    rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                                 'customer': customer, 'error': '非有限差值'})
                    n_err += 1
                    continue
                n_pairs += len(common)
                rows.append({
                    'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                    'event_id': event_id, 'customer': customer,
                    'n_off': len(off_keys), 'n_on': len(on_keys), 'n_common': len(common),
                    'n_off_only': len(off_keys - on_keys), 'n_on_only': len(on_keys - off_keys),
                    'ctx_max_diff': ctx_diff, 'action_max_diff': act_diff,
                    'local_max_diff': loc_diff,
                })
                print(f"  ctx {ctx['context_id'][:12]} inst{inst_idx}/evt{event_id} "
                      f"cust{customer} common={len(common)} off_only={len(off_keys-on_keys)} "
                      f"on_only={len(on_keys-off_keys)} ctx={ctx_diff:.2e} act={act_diff:.2e} "
                      f"loc={loc_diff:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({'context_id': ctx['context_id'][:12], 'inst_idx': inst_idx,
                         'customer': customer, 'error': repr(e)})
            n_err += 1
            continue

    valid = [r for r in rows if 'error' not in r]
    action_max = max((r['action_max_diff'] for r in valid), default=0.0)
    local_max = max((r['local_max_diff'] for r in valid), default=0.0)
    ctx_max = max((r['ctx_max_diff'] for r in valid), default=0.0)
    n_off_only = sum(r['n_off_only'] for r in valid)
    n_on_only = sum(r['n_on_only'] for r in valid)
    passed = (len(valid) > 0 and n_err == 0 and n_off_only == 0 and n_on_only == 0
              and ctx_max < 1e-6 and action_max < 1e-6 and local_max < 1e-6)
    summary = {
        'n_contexts': len(rows), 'n_valid': len(valid), 'n_error': n_err,
        'n_checked_pairs': n_pairs, 'n_off_only': n_off_only, 'n_on_only': n_on_only,
        'ctx_max_diff': ctx_max, 'action_max_diff': action_max, 'local_max_diff': local_max,
        'verdict': 'PASS' if passed else 'FAIL',
        'rows': rows,
    }
    with open(os.path.join(args.out, 'reconcile.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  valid={len(valid)} err={n_err} pairs={n_pairs} off_only={n_off_only} "
          f"on_only={n_on_only}")
    print(f"  ctx_max={ctx_max:.2e} action_max={action_max:.2e} local_max={local_max:.2e}")
    print(f"  verdict={summary['verdict']}")
    print(f"saved: {args.out}/reconcile.json")


if __name__ == '__main__':
    main()
