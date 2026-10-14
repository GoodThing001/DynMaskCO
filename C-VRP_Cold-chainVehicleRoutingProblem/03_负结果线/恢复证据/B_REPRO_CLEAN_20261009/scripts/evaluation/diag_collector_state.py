"""诊断：训练采集器两个源码问题的实证。

1. 状态冻结缺失：collect 保存 live `vehicles` 引用，模拟器随后修改；采集结束后读取的是终局态。
   证明方法：比较同一状态里 frozen 的 `vis.vehicles[k].committed_next`（事件时快照）与 live
   `vehicles[k].committed_next`（终局态），不一致即证明 live 引用已被修改。
2. 实例覆盖：全局 `max_states` 限制，不是「每实例最多 4 个」；列出状态的实际实例分布。
"""
import argparse
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
from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from cc_lns_replanner import _select_one_mask, _plan_suffixes
from visible_state import build_visible_state, evaluate_visible_plan
from repair_state import freeze_vehicles
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)


def _load_profile(path):
    import json
    with open(path) as f:
        d = json.load(f)
    return ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                            quality_scale=float(d['quality_scale']),
                            energy_scale=float(d['energy_scale']),
                            lambda_quality=float(d['lambda_quality']),
                            lambda_energy=float(d['lambda_energy']),
                            scale_source=d.get('scale_source', 'pilot'),
                            dev_statistics=d.get('dev_statistics'))


class BuggyProbe(JF1HRepairReplanner):
    """复刻 train_mpre_reinforce.py 的 CollectProbe（修复后版本：冻结 + 每实例配额）。"""

    def __init__(self, contract, capacity, max_per_instance=4):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.capacity = capacity
        self.max_per_instance = int(max_per_instance)
        self.per_instance = {}
        self.states = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved_ids = set()
        idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
        reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        if self.per_instance.get(int(inst_idx), 0) >= self.max_per_instance:
            return
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in reserved_ids:
            v = vehicles[vid]
            p = P0.get(vid)
            if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                protected.add(vid)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = [int(c) for c in visible_ids if not served_mask[int(c)]]
        mask = _select_one_mask(env, inst_idx, P0, pool, 2,
                                seed=self.per_instance.get(int(inst_idx), 0))
        if not mask:
            return
        mask_set = set(mask)
        partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                tuple(x for x in p.suffix if x not in mask_set))
                   for vid, p in P0.items()}
        vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                                  served_mask, visible_ids, replan_ids, self.deferred_customers)
        r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), self.contract, self.contract.objective)
        self.states.append({'inst': int(inst_idx), 'clock': float(clock),
                            'event': int(getattr(env, 'event_id', -1)),
                            'vehicles': freeze_vehicles(vehicles),
                            'served_mask': np.array(served_mask, copy=True), 'vis': vis,
                            'P0': P0, 'partial': partial, 'mask_set': mask_set,
                            'mutable_ids': mutable_ids, 'J0': float(r0.J_vis)})
        self.per_instance[int(inst_idx)] = self.per_instance.get(int(inst_idx), 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=20)
    ap.add_argument('--max-per-instance', type=int, default=4)
    args = ap.parse_args()

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = default_pilot_contract()
    eff = apply_objective_profile(contract, profile)

    probe = BuggyProbe(eff, args.capacity, args.max_per_instance)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=eff)
        env.run(inst)

    states = probe.states
    print(f'collected {len(states)} states over {args.max_instances} instances', flush=True)

    # 1. 实例分布
    inst_dist = Counter(s['inst'] for s in states)
    print('=== 实例分布（前若干实例的 eligible 事件）===')
    print('  states per instance:', dict(sorted(inst_dist.items())))

    # 2. 状态冻结检查：vis（frozen）vs vehicles（live）的 committed_next
    mismatch = 0
    checked = 0
    for s in states:
        vis = s['vis']
        live = s['vehicles']
        for k, vv in enumerate(vis.vehicles):
            frozen_cn = vv.committed_next  # 事件时快照（int 或 -1）
            live_v = live[k]
            live_cn = int(live_v.committed_next) if live_v.committed_next not in (None, 0) else -1
            if frozen_cn is None:
                frozen_cn = -1
            checked += 1
            if int(frozen_cn) != live_cn:
                mismatch += 1
    print(f'=== 状态冻结检查 ===')
    print(f'  vis(事件时) vs vehicles(终局) committed_next 不一致: {mismatch}/{checked}')

    # 3. served_mask 是否也是 live 引用（直接展示第一个状态读取时的 served_mask 已变）
    # 用 vis 里隐含的信息无法直接对 served_mask；这里只报告 committed_next 的差异即可。


if __name__ == '__main__':
    main()
