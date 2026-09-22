"""缓存验收（真实状态）：缓存/未缓存 score_fn 在真实重构状态上分数 + argmax + 完整计划一致；
并确认同一 H0 下改变部分计划与动态特征后 C/decoder 确实重算（分数变化）。
"""
import argparse
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

from strict_online_env import StrictOnlineEnv
from dynmaskco_cc_graph import build_visible_index
from mpre_policy import stepwise_sample, enumerate_legal_actions
from mpre import load_cvrp_model
from mtrained_replanner import (load_trained_model, _make_extract, mpre_score_fn,
                                mtrained_score_fn, mpre_score_fn_cached,
                                mtrained_score_fn_cached)
from coldchain_contract import default_pilot_contract, apply_objective_profile
from run_fixed_state_quality import Probe, _load_profile
from action_contract import plan_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--max-instances', type=int, default=2)
    args = ap.parse_args()

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    eff = apply_objective_profile(default_pilot_contract(), profile)
    tw_max = float(dataset['tw_end'][:, 0].max())

    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)

    probe = Probe(eff, 50.0, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, 50.0, 1.0, 25, replanner=probe, coldchain_contract=eff)
        env.run(inst)
    states = probe.states
    print(f"collected {len(states)} states", flush=True)

    worst = 0.0
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in st['visible_ids'] if int(i) != 0])
        extract_fn = _make_extract(env, inst_idx, st['clock'], st['vehicles'], st['served_mask'],
                                   st['visible_ids'], 50.0, tw_max, node_to_local, local_to_node,
                                   Nv, st['P0'], st['mutable_ids'])
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']
        legal = enumerate_legal_actions(env, inst_idx, partial, mask_set, mutable_ids)
        state = extract_fn(partial, mask_set, legal)

        # 未缓存 vs 缓存（miss / hit）
        for name, unc, cac in (('Mpre', mpre_score_fn(backbone), mpre_score_fn_cached(backbone)),
                               ('Mtrained', mtrained_score_fn(model),
                                mtrained_score_fn_cached(model))):
            a = np.asarray(unc(state))
            b = np.asarray(cac(state))      # miss
            c = np.asarray(cac(state))      # hit
            d1 = float(np.max(np.abs(a - b)))
            d2 = float(np.max(np.abs(a - c)))
            arg_ok = (int(np.argmax(a)) == int(np.argmax(b)) == int(np.argmax(c)))
            worst = max(worst, d1, d2)
            print(f"  {name} inst{inst_idx}/evt{st['event']} Nv={Nv} M={len(legal)}: "
                  f"miss|Δ|={d1:.2e} hit|Δ|={d2:.2e} argmax={arg_ok}")

        # 完整计划一致：greedy 用未缓存/缓存各跑一遍，比较 plan_hash
        P1, _s, _r, f1 = stepwise_sample(mtrained_score_fn(model), extract_fn, env, inst_idx,
                                         partial, mask_set, mutable_ids, deterministic=True)
        P2, _s, _r, f2 = stepwise_sample(mtrained_score_fn_cached(model), extract_fn, env,
                                         inst_idx, partial, mask_set, mutable_ids,
                                         deterministic=True)
        same_plan = (f1 == f2) and (plan_hash(P1) == plan_hash(P2))
        print(f"  Mtrained full-greedy plan hash match: {same_plan} "
              f"(h1={plan_hash(P1) if P1 else None}, h2={plan_hash(P2) if P2 else None})")

        # 动态特征重算：同一 H0，改 mask 标记 → 分数应变化
        mtrained_cac = mtrained_score_fn_cached(model)
        _ = mtrained_cac(state)
        state2 = extract_fn(partial, [], legal)  # 空 mask（改 mask_set 动态特征）
        d_dyn = float(np.max(np.abs(np.asarray(mtrained_cac(state2)) - np.asarray(mtrained_cac(state)))))
        print(f"  Mtrained dynamic-feature change |Δ| = {d_dyn:.3e} (should be >0)")

    print(f"\nworst cache |Δ| = {worst:.3e}")
    print('PASS' if worst < 1e-6 else 'WARNING non-zero delta')


if __name__ == '__main__':
    main()
