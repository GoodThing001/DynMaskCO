"""M0 读取器与可见特征提取器的防泄漏最小测试（合成小实例，纯 NumPy）。

验证：
  1. 读取器：加载 COMPLETE 数据集、按 context 分组、legal/supervision 两个 mask 分离；
  2. 读取器拒绝 FAILED / 无 COMPLETE / NPZ hash 不符；
  3. 特征：扰动未揭示订单 → order_feats 不变；
  4. 特征：修改终局标签 → 特征不变（终局不进输入）；
  5. 动作：DEFER 缺失字段 → 有效位区分，不当作 0 值。

用法：python scripts/tests/test_coldchain_reader_features.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'expert', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_contract import ObjectiveProfile
from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        extract_order_features)

EXPORTER = os.path.join(_CVRPTW, 'scripts', 'expert', 'export_coldchain_teacher.py')
RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_dataset(coords, demands, temp_class, reveal=None):
    N = coords.shape[0]
    return {
        'coords': coords[None].astype(np.float32),
        'demands': demands[None].astype(np.float32),
        'tw_start': np.zeros((1, N), np.float32),
        'tw_end': np.full((1, N), 100.0, np.float32),
        'service_time': np.zeros((1, N), np.float32),
        'temp_class': np.asarray(temp_class, np.int32)[None],
        'initial_quality': np.ones((1, N), np.float32),
        'reveal_time': (np.zeros((1, N), np.float32) if reveal is None
                        else np.asarray(reveal, np.float32)[None]),
    }


def _write_profile(tmp):
    p = ObjectiveProfile('test-profile', 18.0, 2.9, 1193.0, 1.0, 1.0, 'pilot', None)
    path = os.path.join(tmp, 'objective_profile.json')
    json.dump(p.to_manifest(), open(path, 'w'))
    return path


def _export(tmp, npz, profile, out='teacher'):
    out_dir = os.path.join(tmp, out)
    r = subprocess.run([sys.executable, EXPORTER, '--data', npz, '--dataset-role',
                        'train_teacher', '--objective', 'coldchain',
                        '--objective-profile', profile, '--num-vehicles', '2',
                        '--max-instances', '1', '--max-contexts-per-instance', '3',
                        '--out', out_dir], capture_output=True, text=True)
    return r, out_dir


def _tiny(tmp):
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    ds = _make_dataset(coords, demands, [0, 0, 0, 0])
    npz = os.path.join(tmp, 'tiny.npz')
    np.savez_compressed(npz, **ds)
    return npz


def test_reader_load_and_masks():
    tmp = tempfile.mkdtemp(prefix='reader_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _export(tmp, npz, profile)
        if r.returncode != 0:
            record('reader_load_and_masks', False, f'export exit={r.returncode}')
            return False
        ds = load_teacher_dataset(out, data_path=npz)
        ok = len(ds.contexts) >= 1 and len(ds.candidates) >= 1
        # legal 与 supervision 是两个不同 mask，且 legal ⊇ supervision（除不可行负例外）。
        for ctx in ds.contexts:
            cands = ds.candidates_by_context[ctx['context_id']]
            legal = ds.legal_mask(cands)
            sup = ds.supervision_mask(cands)
            ok = ok and len(legal) == len(cands) and len(sup) == len(cands)
            # 每个 supervision 候选必是 legal。
            ok = ok and all((not s) or l for s, l in zip(sup, legal))
        record('reader_load_and_masks', ok,
               f"contexts={len(ds.contexts)} candidates={len(ds.candidates)} "
               f"unsupervised={len(ds.unsupervised)}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reader_rejects_failed():
    tmp = tempfile.mkdtemp(prefix='reader_fail_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _export(tmp, npz, profile)
        # 人为标记 FAILED 并删 COMPLETE。
        os.remove(os.path.join(out, 'COMPLETE'))
        open(os.path.join(out, 'FAILED'), 'w').write('{"status":"FAILED"}')
        try:
            load_teacher_dataset(out, data_path=npz)
            ok = False
        except ValueError:
            ok = True
        record('reader_rejects_failed', ok)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reader_rejects_hash_mismatch():
    tmp = tempfile.mkdtemp(prefix='reader_hash_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _export(tmp, npz, profile)
        # 换一个不同内容的 NPZ 路径。
        other = os.path.join(tmp, 'other.npz')
        coords = np.array([[0., 0.], [9., 9.], [2., 0.]], np.float32)
        ds2 = _make_dataset(coords, np.array([0., 1., 1.], np.float32), [0, 0, 0])
        np.savez_compressed(other, **ds2)
        try:
            load_teacher_dataset(out, data_path=other)
            ok = False
        except ValueError:
            ok = True
        record('reader_rejects_hash_mismatch', ok)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unrevealed_order_perturbation_invariant():
    # 客户 3 未揭示（reveal_time=10 > clock=0）。
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [30., 30.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    reveal = np.array([0., 0., 0., 10.], np.float32)
    ds = _make_dataset(coords, demands, [0, 0, 0, 0], reveal=reveal)
    visible = np.array([True, True, True, False])
    order1, vis1 = extract_order_features(ds, 0, visible)
    # 扰动未揭示客户 3 的坐标 → order_feats 不变（已掩码）。
    ds2 = dict(ds)
    c = ds2['coords'].copy()
    c[0, 3] = [999., 999.]
    ds2['coords'] = c
    order2, vis2 = extract_order_features(ds2, 0, visible)
    ok = np.array_equal(order1, order2)
    ok = ok and not vis1[3]  # 客户 3 未揭示
    record('unrevealed_order_perturbation_invariant', ok,
           f"visible_nodes={int(vis1.sum())}/{len(vis1)}")
    return ok


def test_terminal_label_not_in_features():
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32)
    ds = _make_dataset(coords, np.array([0., 1., 1., 1.], np.float32), [0, 0, 0, 0])
    snapshot = {
        'num_vehicles': 2,
        'visible_mask': np.array([True, True, True, True]),
        'vehicle_node': np.array([0, 0], np.int32),
        'vehicle_ready': np.array([0.0, 0.0]),
        'vehicle_load': np.array([0.0, 0.0]),
        'needs_replan': np.array([True, True]),
        'committed_next': np.array([-1, -1], np.int32),
        'committed_arrive': np.array([np.nan, np.nan]),
        'committed_finish': np.array([np.nan, np.nan]),
        'vehicle_coldchain_state': [None, None],
    }
    feats1 = extract_context_features(ds, 0, snapshot)
    # 修改一个候选的终局标签（不进入 extract_context_features）不影响特征。
    cand = {'action': {'customer': 1, 'slot_kind': 'anchored', 'slot_anchor': 0,
                       'position': 0, 'predecessor': 0, 'successor': 0,
                       'incumbent': False},
            'outcome': {'coldchain_cost': 999.0}}
    a1, v1 = extract_action_features(cand)
    cand['outcome']['coldchain_cost'] = 0.0  # 改标签
    a2, v2 = extract_action_features(cand)
    ok = np.array_equal(a1, a2) and np.array_equal(v1, v2)
    record('terminal_label_not_in_features', ok)
    return ok


def test_defer_missing_fields_valid_bits():
    cand = {'action': {'customer': 2, 'kind': 'defer'}, 'is_pseudo': True, 'pseudo': 'DEFER'}
    vals, valid = extract_action_features(cand)
    # DEFER 的 slot_anchor/position/predecessor/successor 缺失 → valid=False，值=0。
    # customer 存在 → valid=True。
    ok = bool(valid[0]) and not bool(valid[2]) and not bool(valid[3]) \
         and not bool(valid[4]) and not bool(valid[5])
    record('defer_missing_fields_valid_bits', ok,
           f"valid={valid.tolist()} vals={vals.tolist()}")
    return ok


def main():
    ok = [test_reader_load_and_masks(), test_reader_rejects_failed(),
          test_reader_rejects_hash_mismatch(), test_unrevealed_order_perturbation_invariant(),
          test_terminal_label_not_in_features(), test_defer_missing_fields_valid_bits()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
