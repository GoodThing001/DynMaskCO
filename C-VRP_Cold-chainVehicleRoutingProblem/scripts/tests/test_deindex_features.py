"""去编号（deindex）特征与重编号不变性最小测试（纯 NumPy + 合成小实例）。

验证：
  1. deindex 只屏蔽编号通道（动作 customer/slot_anchor/predecessor/successor；
     车队 vehicle_node/committed_next），有效位与非编号通道不变；
  2. 重编号 perm 是合法双射（depot 固定）；
  3. 重编号只改编号通道（非编号特征严格不变）；
  4. deindex 后重编号 → 特征完全一致（去编号版本应通过的对应不变性检查）。

用法：python scripts/tests/test_deindex_features.py
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'expert', 'data',
          'training', 'models'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_visible_features import (extract_action_features, extract_fleet_features,
                                        ACTION_ID_CHANNELS, FLEET_ID_CHANNELS,
                                        ACTION_FEAT_DIM, FLEET_FEAT_DIM)
from run_id_sensitivity import (renumber_context, build_context_tensors,
                                _verify_renumber_invariant)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _dataset(N=5):
    rng = np.random.default_rng(0)
    return {
        'coords': rng.uniform(0, 10, (1, N, 2)).astype(np.float32),
        'demands': rng.uniform(1, 5, (1, N)).astype(np.float32),
        'tw_start': np.zeros((1, N), np.float32),
        'tw_end': np.full((1, N), 100.0, np.float32),
        'service_time': np.zeros((1, N), np.float32),
        'temp_class': np.array([[0, 0, 1, 2, 0]], np.int32),
        'initial_quality': np.ones((1, N), np.float32),
        'reveal_time': np.zeros((1, N), np.float32),
    }


def _snapshot(N=5):
    return {
        'num_vehicles': 2,
        'visible_mask': np.ones(N, bool).tolist(),
        'vehicle_node': [0, 3],
        'vehicle_ready': [0.0, 0.0],
        'vehicle_load': [0.0, 0.0],
        'needs_replan': [True, True],
        'committed_next': [-1, 2],
        'committed_arrive': [float('nan'), float('nan')],
        'committed_finish': [float('nan'), float('nan')],
        'vehicle_coldchain_state': [None, None],
    }


def _cands():
    return [
        {'action': {'customer': 3, 'slot_kind': 'anchored', 'slot_anchor': 0,
                    'position': 0, 'predecessor': 0, 'successor': 1, 'incumbent': False},
         'is_pseudo': False, 'certificate': {'incremental_distance': 2.5}},
        {'action': {'customer': 1, 'slot_kind': 'new_route', 'slot_anchor': 0,
                    'position': 0, 'predecessor': 0, 'successor': 0, 'incumbent': False},
         'is_pseudo': False, 'certificate': {'incremental_distance': 1.0}},
        {'action': {'customer': 2, 'kind': 'defer'}, 'is_pseudo': True, 'pseudo': 'DEFER',
         'certificate': {}},
    ]


def test_deindex_masks_action_channels():
    cand = {'action': {'customer': 3, 'slot_kind': 'anchored', 'slot_anchor': 2,
                       'position': 1, 'predecessor': 1, 'successor': 2,
                       'incumbent': False}, 'is_pseudo': False}
    v0, ok0 = extract_action_features(cand, deindex=False)
    v1, ok1 = extract_action_features(cand, deindex=True)
    keep = [i for i in range(ACTION_FEAT_DIM) if i not in ACTION_ID_CHANNELS]
    ok = np.array_equal(ok0, ok1)
    ok = ok and np.allclose(v0[keep], v1[keep])
    ok = ok and all(v1[c] == 0.0 for c in ACTION_ID_CHANNELS)
    ok = ok and all(v0[c] != 0.0 for c in ACTION_ID_CHANNELS)
    record('deindex_masks_action_channels', ok,
           f"id_channels={ACTION_ID_CHANNELS} v0={v0.tolist()} v1={v1.tolist()}")
    return ok


def test_deindex_masks_fleet_channels():
    # 用非零编号值的 snapshot 以便断言「屏蔽前编号非零、屏蔽后为零」。
    snap = {
        'num_vehicles': 2,
        'vehicle_node': [3, 2],
        'vehicle_ready': [0.0, 0.0],
        'vehicle_load': [1.0, 2.0],
        'needs_replan': [True, True],
        'committed_next': [2, 1],
        'committed_arrive': [1.0, 2.0],
        'committed_finish': [3.0, 4.0],
        'vehicle_coldchain_state': [None, None],
    }
    f0, vv0 = extract_fleet_features(snap, deindex=False)
    f1, vv1 = extract_fleet_features(snap, deindex=True)
    keep = [i for i in range(FLEET_FEAT_DIM) if i not in FLEET_ID_CHANNELS]
    ok = np.array_equal(vv0, vv1)
    ok = ok and np.allclose(f0[:, keep], f1[:, keep])
    ok = ok and all(np.all(f1[:, c] == 0.0) for c in FLEET_ID_CHANNELS)
    # 非零编号值被屏蔽
    ok = ok and all(np.all(f0[:, c] != 0.0) for c in FLEET_ID_CHANNELS)
    record('deindex_masks_fleet_channels', ok,
           f"id_channels={FLEET_ID_CHANNELS}")
    return ok


def test_renumber_is_bijection():
    ds = _dataset(5)
    snap = _snapshot(5)
    cands = _cands()
    rng = np.random.default_rng(1)
    _, _, _, perm = renumber_context(ds, snap, cands, rng)
    N = 5
    ok = perm[0] == 0
    ok = ok and sorted(perm.tolist()) == list(range(N))
    record('renumber_is_bijection', ok, f"perm={perm.tolist()}")
    return ok


def test_renumber_only_touches_id_channels():
    ds = _dataset(5)
    snap = _snapshot(5)
    cands = _cands()
    rng = np.random.default_rng(2)
    t_orig = build_context_tensors(ds, 0, snap, cands, 50.0, deindex=False)
    rd, rs, rc, perm = renumber_context(ds, snap, cands, rng)
    t_ren = build_context_tensors(rd, 0, rs, rc, 50.0, deindex=False)
    ok = _verify_renumber_invariant(t_orig, t_ren)
    # 编号通道确实变了（非恒等置换且编号非全 0）
    av_o, av_r = t_orig[1], t_ren[1]
    id_changed = any(np.any(av_o[:, c] != av_r[:, c]) for c in ACTION_ID_CHANNELS)
    record('renumber_only_touches_id_channels', ok and id_changed,
           f"invariant={ok} id_changed={id_changed}")
    return ok and id_changed


def test_deindex_is_renumber_invariant():
    ds = _dataset(5)
    snap = _snapshot(5)
    cands = _cands()
    rng = np.random.default_rng(3)
    t_orig = build_context_tensors(ds, 0, snap, cands, 50.0, deindex=True)
    rd, rs, rc, perm = renumber_context(ds, snap, cands, rng)
    t_ren = build_context_tensors(rd, 0, rs, rc, 50.0, deindex=True)
    ok = all(np.allclose(a, b) for a, b in zip(t_orig, t_ren))
    record('deindex_is_renumber_invariant', ok, f"perm={perm.tolist()}")
    return ok


def main():
    ok = [test_deindex_masks_action_channels(), test_deindex_masks_fleet_channels(),
          test_renumber_is_bijection(), test_renumber_only_touches_id_channels(),
          test_deindex_is_renumber_invariant()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
