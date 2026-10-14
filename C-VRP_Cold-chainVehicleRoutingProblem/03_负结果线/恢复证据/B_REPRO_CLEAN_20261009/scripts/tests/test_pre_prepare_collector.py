"""PrePrepareContextCollector 合成状态测试（batch A）。

验证 collector 在 snapshot_hook 时点捕获 pre-prepare 状态，缓存键与拒绝语义：
  1. 有 replan 需求才缓存（无 replan 不缓存）；
  2. 多事件缓存按 (inst_idx, event_id, clock) 正确更新；
  3. pre-bump 与 post-bump 捕获到不同 context（ready_time 被 bump 改变）；
  4. 缺失/归属不符缓存 get 显式报错。

正确性（pre-bump 值 == 训练 snapshot）由 batch B 对账（ctx=0）承担；本测试只覆盖缓存机制。
纯本地对象层，无需服务器数据。
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import VehicleState
from feature_local_replanner import PrePrepareContextCollector


class _MockEnv:
    def __init__(self, num_nodes=4):
        self.num_nodes = num_nodes
        self.num_vehicles = 3
        self.coords = np.random.default_rng(0).uniform(0, 1, (1, num_nodes, 2)).astype(np.float32)
        self.demands = np.zeros((1, num_nodes), np.float32)
        self.demands[0, 1:] = 1.0
        self.tw_start = np.zeros((1, num_nodes), np.float32)
        self.tw_end = np.full((1, num_nodes), 50.0, np.float32)
        self.service_time = np.zeros((1, num_nodes), np.float32)
        self.temp_class = np.zeros((1, num_nodes), np.int32)
        self.initial_quality = np.ones((1, num_nodes), np.float32)
        self.reveal_time = np.full((1, num_nodes), 0.0, np.float32)


def _vehicles():
    v0 = VehicleState(vehicle_id=0)   # idle
    v1 = VehicleState(vehicle_id=1)   # idle
    v2 = VehicleState(vehicle_id=2)   # ready，已完成一腿
    v2.status = 'ready'
    v2.current_node = 1
    v2.ready_time = 10.0
    for v in (v0, v1, v2):
        v.needs_replan = True
    return [v0, v1, v2]


def _prepare(vehicles, clock):
    for v in vehicles:
        if v.status == 'idle':
            v.ready_time = float(clock)
        elif v.status == 'ready':
            v.ready_time = max(v.ready_time, float(clock))


def main():
    env = _MockEnv()
    col = PrePrepareContextCollector(deindex=True)
    all_customers = [1, 2, 3]
    served_mask = np.zeros(4, bool)

    results = []
    def record(name, ok, detail=''):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # 1. 无 replan 需求不缓存
    veh = _vehicles()
    for v in veh:
        v.needs_replan = False
    col.hook(env, 0, 2.0, 1, 0, veh, None, served_mask, all_customers)
    record('no-replan does not cache', len(col.cache) == 0)

    # 2. pre-bump vs post-bump 捕获不同（ready_time 被 bump）
    veh = _vehicles()
    col.hook(env, 0, 2.0, 1, 0, veh, None, served_mask, all_customers)   # pre-bump (idle=0)
    c_pre = col.get(0, 1, 2.0)
    _prepare(veh, 2.0)                                                   # bump idle→2
    col.hook(env, 0, 5.0, 2, 0, veh, None, served_mask, all_customers)   # pre-bump of evt2 (idle=2)
    c_evt2 = col.get(0, 2, 5.0)
    record('pre-bump vs post-bump differ', float(np.abs(c_pre - c_evt2).max()) > 0.0,
           f"maxdiff={np.abs(c_pre - c_evt2).max():.3e}")

    # 3. 多事件缓存键
    _prepare(veh, 5.0)
    col.hook(env, 0, 8.0, 3, 0, veh, None, served_mask, all_customers)
    keys = sorted(col.cache.keys())
    record('multi-event cache keys', keys == [(0, 1, 2.0), (0, 2, 5.0), (0, 3, 8.0)], f"{keys}")

    # 4. 缺失缓存报错
    try:
        col.get(0, 99, 1.0)
        record('missing cache raises', False, '未报错')
    except RuntimeError:
        record('missing cache raises', True)

    # 5. 归属不符报错（inst 不同）
    try:
        col.get(7, 1, 2.0)
        record('wrong-inst cache raises', False, '未报错')
    except RuntimeError:
        record('wrong-inst cache raises', True)

    print(f"\n  ALL: {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
