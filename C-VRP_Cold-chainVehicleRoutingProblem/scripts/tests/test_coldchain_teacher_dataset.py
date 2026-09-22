"""M0 candidate teacher 导出器的 schema/parity/leakage 最小测试（合成小实例，纯 NumPy）。

验证：
  1. 导出器在合成冷链接上跑通，输出 manifest / contexts / candidates / COMPLETE；
  2. manifest 绑定 schema、allow-list、raw/effective contract 与 profile hash；
  3. gate_summary 报告 keep_parity=PASS、contract_consistent=True、fail_closed=True；
  4. 候选行包含结构化 action payload + certificate + 终局 outcome + 派生标签；
  5. KEEP/DEFER 基准伪动作显式存在（trainer 能学习“保持当前计划”）；
  6. --max-contexts-per-instance 精确限制 (snapshot × customer) context 数；
  7. manifest cell 按 NPZ hash 匹配并登记 instance_seed / scene_instance_id；
  8. --max-instances 0 与 --max-contexts-per-instance 0 被拒绝。

用法：python scripts/tests/test_coldchain_teacher_dataset.py
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_contract import ObjectiveProfile
import export_coldchain_teacher as teacher_mod

EXPORTER = os.path.join(_CVRPTW, 'scripts', 'expert', 'export_coldchain_teacher.py')
RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _make_cc_dataset(coords, demands, temp_class):
    N = coords.shape[0]
    return {
        'coords': coords[None].astype(np.float32),
        'demands': demands[None].astype(np.float32),
        'tw_start': np.zeros((1, N), np.float32),
        'tw_end': np.full((1, N), 100.0, np.float32),
        'service_time': np.zeros((1, N), np.float32),
        'temp_class': np.asarray(temp_class, np.int32)[None],
        'initial_quality': np.ones((1, N), np.float32),
        'reveal_time': np.zeros((1, N), np.float32),
    }


def _write_profile(tmp):
    p = ObjectiveProfile('test-profile', 18.0, 2.9, 1193.0, 1.0, 1.0, 'pilot', None)
    path = os.path.join(tmp, 'objective_profile.json')
    json.dump(p.to_manifest(), open(path, 'w'))
    return path


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def _tiny(tmp, extra_coords=True):
    coords = (np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32)
              if extra_coords else np.array([[0., 0.], [1., 0.], [2., 0.]], np.float32))
    demands = np.array([0., 1., 1., 1.], np.float32)[:coords.shape[0]]
    dataset = _make_cc_dataset(coords, demands, [0] * coords.shape[0])
    npz = os.path.join(tmp, 'tiny.npz')
    np.savez_compressed(npz, **dataset)
    return npz


def _run(tmp, npz, profile, max_instances=1, max_contexts=2, manifest=None, out='teacher'):
    out_dir = os.path.join(tmp, out)
    cmd = [sys.executable, EXPORTER, '--data', npz, '--dataset-role', 'train_teacher',
           '--objective', 'coldchain', '--objective-profile', profile,
           '--num-vehicles', '2', '--max-instances', str(max_instances),
           '--max-contexts-per-instance', str(max_contexts), '--out', out_dir]
    if manifest:
        cmd += ['--manifest', manifest]
    return subprocess.run(cmd, capture_output=True, text=True), out_dir


def test_export_smoke_and_schema():
    tmp = tempfile.mkdtemp(prefix='teacher_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _run(tmp, npz, profile, max_contexts=3)
        if r.returncode != 0:
            record('teacher_export_smoke', False, f'exit={r.returncode} {r.stderr[-400:]}')
            return False

        manifest = json.load(open(os.path.join(out, 'manifest.json')))
        ok = manifest.get('schema') == 'o0cc-teacher-dataset-v1'
        ok = ok and manifest.get('feature_schema_version') == 'v1'
        ok = ok and 'allow_list_summary' in manifest
        ok = ok and manifest.get('raw_contract_hash') and manifest.get('effective_contract_hash')
        ok = ok and manifest.get('profile_hash')

        contexts = [json.loads(l) for l in open(os.path.join(out, 'contexts.jsonl'))]
        candidates = [json.loads(l) for l in open(os.path.join(out, 'candidates.jsonl'))]
        ok = ok and len(contexts) >= 1 and len(candidates) >= 1
        ok = ok and os.path.exists(os.path.join(out, 'COMPLETE'))
        ok = ok and not os.path.exists(os.path.join(out, 'FAILED'))

        ok = ok and all('action' in c and 'certificate' in c and 'context_id' in c
                        for c in candidates)
        ok = ok and all('snapshot' in c and 'baseline_outcome' in c and 'state_hash' in c
                        for c in contexts)

        gate = json.load(open(os.path.join(out, 'qc', 'gate_summary.json')))
        ok = ok and gate.get('keep_parity') == 'PASS'
        ok = ok and gate.get('contract_consistent') is True
        ok = ok and gate.get('fail_closed') is True

        record('teacher_export_smoke', ok,
               f"contexts={len(contexts)} candidates={len(candidates)} "
               f"service_ok={gate.get('n_service_ok')} proto_err={gate.get('n_protocol_error')}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_keep_defer_present():
    tmp = tempfile.mkdtemp(prefix='teacher_keep_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _run(tmp, npz, profile, max_contexts=4)
        if r.returncode != 0:
            record('teacher_keep_defer_present', False, f'exit={r.returncode} {r.stderr[-300:]}')
            return False
        candidates = [json.loads(l) for l in open(os.path.join(out, 'candidates.jsonl'))]
        pseudo = [c for c in candidates if c.get('is_pseudo')]
        ok = len(pseudo) >= 1
        # 每个 pseudo 行都有 delta_vs_keep=0 且是 KEEP 或 DEFER。
        ok = ok and all(c['delta_vs_keep'] == 0.0 and c['pseudo'] in ('KEEP', 'DEFER')
                        for c in pseudo)
        # 至少一个 context 内 teacher 选择了某个候选或伪动作。
        ok = ok and any(c.get('selected_by_teacher') for c in candidates)
        record('teacher_keep_defer_present', ok,
               f"pseudo={len(pseudo)} (KEEP/DEFER) selected={sum(c.get('selected_by_teacher') for c in candidates)}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_context_limit():
    tmp = tempfile.mkdtemp(prefix='teacher_ctx_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _run(tmp, npz, profile, max_contexts=1)
        contexts = [json.loads(l) for l in open(os.path.join(out, 'contexts.jsonl'))]
        ok = r.returncode == 0 and len(contexts) == 1
        record('teacher_context_limit', ok, f"contexts={len(contexts)} (expect 1)")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_matching():
    tmp = tempfile.mkdtemp(prefix='teacher_manifest_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        # 构造 manifest：cell sha256 匹配 NPZ，含 instance_seeds/scene_ids。
        manifest = {'split_role': 'train_teacher',
                    'cells': [{'type': 'R1', 'edod': 0.5, 'file': 'tiny.npz',
                               'sha256': _sha256_file(npz),
                               'instance_seeds': [111, 222, 333],
                               'scene_instance_ids': ['s0', 's1', 's2']}]}
        mp = os.path.join(tmp, 'MANIFEST.json')
        json.dump(manifest, open(mp, 'w'))
        r, out = _run(tmp, npz, profile, max_instances=1, manifest=mp)
        if r.returncode != 0:
            record('teacher_manifest_matching', False, f'exit={r.returncode} {r.stderr[-300:]}')
            return False
        m = json.load(open(os.path.join(out, 'manifest.json')))
        ctx = json.loads(open(os.path.join(out, 'contexts.jsonl')).readline())
        ok = m.get('split_registered') is True
        ok = ok and m.get('instance_seeds') == [111]
        ok = ok and ctx.get('instance_seed') == 111 and ctx.get('scene_instance_id') == 's0'
        record('teacher_manifest_matching', ok,
               f"seed={ctx.get('instance_seed')} scene={ctx.get('scene_instance_id')}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_args_rejected():
    tmp = tempfile.mkdtemp(prefix='teacher_args_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r0, _ = _run(tmp, npz, profile, max_instances=0)
        r1, _ = _run(tmp, npz, profile, max_contexts=0)
        ok = r0.returncode != 0 and r1.returncode != 0
        record('teacher_bad_args_rejected', ok, f"max-instances=0:{r0.returncode} "
              f"max-contexts=0:{r1.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cell(sha256, seeds, scenes, file='tiny.npz'):
    return {'type': 'R1', 'edod': 0.5, 'file': file, 'sha256': sha256,
            'instance_seeds': seeds, 'scene_instance_ids': scenes}


def test_manifest_same_name_wrong_hash_rejected():
    tmp = tempfile.mkdtemp(prefix='teacher_manhash_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        manifest = {'split_role': 'train_teacher',
                    'cells': [_cell('0' * 64, [111, 222], ['s0', 's1'])]}  # 同名错 hash
        mp = os.path.join(tmp, 'M.json'); json.dump(manifest, open(mp, 'w'))
        r, _ = _run(tmp, npz, profile, manifest=mp)
        ok = r.returncode != 0
        record('teacher_manifest_same_name_wrong_hash', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_duplicate_hash_rejected():
    tmp = tempfile.mkdtemp(prefix='teacher_manhash2_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        h = _sha256_file(npz)
        manifest = {'split_role': 'train_teacher',
                    'cells': [_cell(h, [111, 222], ['s0', 's1'], 'a.npz'),
                              _cell(h, [333, 444], ['t0', 't1'], 'b.npz')]}  # 重复 hash
        mp = os.path.join(tmp, 'M.json'); json.dump(manifest, open(mp, 'w'))
        r, _ = _run(tmp, npz, profile, manifest=mp)
        ok = r.returncode != 0
        record('teacher_manifest_duplicate_hash', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_missing_scene_rejected():
    tmp = tempfile.mkdtemp(prefix='teacher_manscene_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        manifest = {'split_role': 'train_teacher',
                    'cells': [_cell(_sha256_file(npz), [111, 222], None)]}  # 缺 scene
        mp = os.path.join(tmp, 'M.json'); json.dump(manifest, open(mp, 'w'))
        r, _ = _run(tmp, npz, profile, manifest=mp)
        ok = r.returncode != 0
        record('teacher_manifest_missing_scene', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_action_hash_consistent():
    """每个候选的 action_hash 必须等于其保存的 action payload 的规范 hash。"""
    tmp = tempfile.mkdtemp(prefix='teacher_ahash_')
    try:
        npz = _tiny(tmp)
        profile = _write_profile(tmp)
        r, out = _run(tmp, npz, profile, max_contexts=4)
        if r.returncode != 0:
            record('teacher_action_hash_consistent', False, f'exit={r.returncode}')
            return False
        candidates = [json.loads(l) for l in open(os.path.join(out, 'candidates.jsonl'))]
        ok = all(c['action_hash'] == teacher_mod._action_hash(c['action']) for c in candidates)
        record('teacher_action_hash_consistent', ok, f"candidates={len(candidates)}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ok = [test_export_smoke_and_schema(), test_keep_defer_present(), test_context_limit(),
          test_manifest_matching(), test_bad_args_rejected(),
          test_manifest_same_name_wrong_hash_rejected(), test_manifest_duplicate_hash_rejected(),
          test_manifest_missing_scene_rejected(), test_action_hash_consistent()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
