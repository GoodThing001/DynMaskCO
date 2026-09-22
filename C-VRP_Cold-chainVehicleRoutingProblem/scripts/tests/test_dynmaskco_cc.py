"""DynMaskCO-CC online replanner 最小测试（no-op 骨架验收）。

验证：
  1. no-op（model=None）与 JF1-H-F baseline 在完整动态轨迹上精确一致（终局 cost + served）；
  2. 多条 rollout 分支（不同 replanner 实例）状态隔离，不互相串扰。

用法：python scripts/tests/test_dynmaskco_cc.py
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from dynmaskco_cc_replanner import make_dynmaskco_replanner
from counterfactual_teacher import _eval

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _tiny(n=5):
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.02, 0.98, (1, n + 1, 2)).astype(np.float32)
    coords[0, 0] = [0.5, 0.5]
    demands = np.zeros((1, n + 1), np.float32)
    demands[0, 1:] = 1.0
    return {
        'coords': coords,
        'demands': demands,
        'tw_start': np.zeros((1, n + 1), np.float32),
        'tw_end': np.full((1, n + 1), 100.0, np.float32),
        'service_time': np.zeros((1, n + 1), np.float32),
        'reveal_time': np.zeros((1, n + 1), np.float32),
        'temp_class': np.zeros((1, n + 1), np.int32),
        'initial_quality': np.ones((1, n + 1), np.float32),
    }


def test_noop_parity():
    dataset = _tiny()
    objective = 'distance'
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 3, replanner=make_continuation())
    traces_b, served_b = env.run(0)
    base = _eval(env, 0, traces_b, objective, served_mask=served_b)

    menv = StrictOnlineEnv(dataset, 50.0, 1.0, 3,
                           replanner=make_dynmaskco_replanner(model=None, K=1))
    traces_m, served_m = menv.run(0)
    m1 = _eval(menv, 0, traces_m, objective, served_mask=served_m)

    ok = np.array_equal(served_b, served_m)
    ok = ok and abs(float(base['distance_cost']) - float(m1['distance_cost'])) <= 1e-9
    ok = ok and bool(base['complete']) == bool(m1['complete'])
    record('noop_parity', ok,
           f"cost base={base['distance_cost']:.4f} m1={m1['distance_cost']:.4f}")
    return ok


def test_branch_isolation():
    dataset = _tiny(6)
    objective = 'distance'
    # 两个 no-op replanner 实例独立跑，结果都等于 baseline（决策状态不跨实例串扰）
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 3, replanner=make_continuation())
    traces_b, served_b = env.run(0)
    base_cost = float(_eval(env, 0, traces_b, objective, served_mask=served_b)['distance_cost'])
    for _ in range(2):
        menv = StrictOnlineEnv(dataset, 50.0, 1.0, 3,
                               replanner=make_dynmaskco_replanner(model=None, K=1))
        traces_m, served_m = menv.run(0)
        mc = float(_eval(menv, 0, traces_m, objective, served_mask=served_m)['distance_cost'])
        if abs(mc - base_cost) > 1e-9:
            record('branch_isolation', False, f"cost {mc} != base {base_cost}")
            return False
    record('branch_isolation', True)
    return True


def main():
    ok = [test_noop_parity(), test_branch_isolation()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}  ({sum(ok)}/{len(ok)})")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
