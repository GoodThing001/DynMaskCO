"""诊断：固定容量 padding 是否无损（padded vs unpadded 的 Mpre/Mtrained 分数对比）。

对真实决策点建状态，用 Nv_max=None（unpadded）与 Nv_max=51/M_max=2048（padded）分别评分，
比较有效动作分数。若 padding 无损，两者应逐元素一致（仅浮点噪声）。

用法（服务器）：
    python scripts/evaluation/diag_padding_lossless.py \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
        --model-ckpt results/m0_scale/mpre_reinforce_s42/model.ckpt \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-events 3 --out results/m0_scale/_diag_padding
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from cc_lns_replanner import _select_one_mask
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from repair_state import extract_repair_state_v1
from mpre_policy import enumerate_legal_actions
from mpre_replanner import retained_edge_ratio
from mpre import load_cvrp_model
from mtrained_replanner import load_trained_model, mpre_score_fn, mtrained_score_fn
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)


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


class Probe(JF1HRepairReplanner):
    def __init__(self, contract, capacity, tw_max, max_events):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.capacity = capacity
        self.tw_max = tw_max
        self.max_events = int(max_events)
        self.states = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        if len(self.states) >= self.max_events:
            super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                         replan_ids=replan_ids)
            return
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = [int(c) for c in visible_ids if not served_mask[int(c)]]
        mask = _select_one_mask(env, inst_idx, P0, pool, 2, seed=len(self.states))
        if not mask:
            return
        mask_set = set(mask)
        partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                tuple(x for x in p.suffix if x not in mask_set))
                   for vid, p in P0.items()}
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in visible_ids if int(i) != 0])
        legal = enumerate_legal_actions(env, inst_idx, partial, list(mask_set), mutable_ids)
        if not legal:
            return
        st = extract_repair_state_v1(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                                     partial, mask_set, legal, self.capacity, self.tw_max,
                                     allowed_vehicle_ids=mutable_ids,
                                     node_to_local=node_to_local, local_to_node=local_to_node,
                                     Nv=Nv)
        A0 = build_plan_adjacency(P0, node_to_local, Nv)
        st['timestep'] = retained_edge_ratio(A0, st['adjmat'][0])
        self.states.append(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-events', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = load_coldchain_contract(None) if False else default_pilot_contract()
    eff = apply_objective_profile(contract, profile)
    tw_max = float(dataset['tw_end'][:, 0].max())
    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)

    rep = Probe(eff, args.capacity, tw_max, args.max_events)
    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                          replanner=rep, coldchain_contract=eff)
    env.run(0)
    states = rep.states
    print(f'captured {len(states)} states', flush=True)

    mpre_u = mpre_score_fn(backbone)          # unpadded
    mpre_p = mpre_score_fn(backbone, 51, 2048)  # padded
    mt_u = mtrained_score_fn(model)           # unpadded
    mt_p = mtrained_score_fn(model, 51, 2048)  # padded

    report = {'states': []}
    for i, st in enumerate(states):
        M = st['cust'].shape[1]
        mpre_u_s = mpre_u(st); mpre_p_s = mpre_p(st)
        mt_u_s = mt_u(st); mt_p_s = mt_p(st)
        r = {
            'i': i, 'Nv': st['raw_3d'].shape[1], 'M': M,
            'mpre_max_abs_diff': float(np.abs(mpre_u_s - mpre_p_s[:M]).max()),
            'mtrained_max_abs_diff': float(np.abs(mt_u_s - mt_p_s[:M]).max()),
        }
        report['states'].append(r)
        print(f"  [state {i}] Nv={r['Nv']} M={r['M']} "
              f"mpre_diff={r['mpre_max_abs_diff']:.3e} mtrained_diff={r['mtrained_max_abs_diff']:.3e}",
              flush=True)

    report['mpre_lossless'] = all(s['mpre_max_abs_diff'] < 1e-4 for s in report['states'])
    report['mtrained_lossless'] = all(s['mtrained_max_abs_diff'] < 1e-4 for s in report['states'])
    report['ALL_LOSSLESS'] = report['mpre_lossless'] and report['mtrained_lossless']
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"mpre_lossless={report['mpre_lossless']} mtrained_lossless={report['mtrained_lossless']}")
    print(f"ALL_LOSSLESS={report['ALL_LOSSLESS']}")


if __name__ == '__main__':
    main()
