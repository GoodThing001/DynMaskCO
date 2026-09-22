"""生成 PyVRP-RH-D identity rev4 行为 parity 证据（OR6.2 收口）。

在 rev4 代码下重跑 r1_02 inst_0/1（max_iterations=300, seed=0），与冻结行为
证据 devproto_9x2_iter300 的旧实例对比 decision_hash，证明 identity rev3→rev4
（common baseline_contract.artifact_hash 排除顶层 runtime_s）不改变行为。

输出 identity_rev4_behavior_parity.json，绑定：比较器源码 hash、新旧实例文件
sha256、新旧 decision_hash / code_hash。

用法：
    python tools/gen_rev4_parity.py
"""
import hashlib
import json
import os
import sys
import time

_DCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.normpath(os.path.join(_DCC, '..', '..', 'common'))
for p in (_DCC, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

import numpy as np
from strict_online_runner import run_instance, load_objective_profile
from pyvrp_adapter import PyVRPRHDAdapter

_PROJECT = _bootstrap.PROJECT_EXTENSION_ROOT
DATA_DIR = os.path.join(_PROJECT, 'data', 'baseline', '50_node', 'dev_proto')
PROFILE = os.path.join(_PROJECT, 'results', 'o0cc', 'scale_v2',
                       'objective_profile.json')
ADAPTER = os.path.join(_DCC, 'pyvrp_adapter.py')
OLD_DIR = os.path.join(_DCC, 'results', 'devproto_9x2_iter300')
NEW_DIR = os.path.join(_DCC, 'results', 'rev4_parity', 'r1_02')
OUT = os.path.join(_DCC, 'identity_rev4_behavior_parity.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    comparator = sha256_file(os.path.abspath(__file__))
    dev_manifest = json.load(open(os.path.join(DATA_DIR, 'DEV_MANIFEST.json'),
                                  encoding='utf-8'))
    profile = load_objective_profile(PROFILE)
    c = next(x for x in dev_manifest['cells']
             if x['type'] == 'R1' and str(x['edod']).replace('.', '') == '02')
    ds = dict(np.load(os.path.join(DATA_DIR, c['file'])))

    os.makedirs(os.path.join(NEW_DIR, 'instances'), exist_ok=True)
    entries = []
    all_match = True
    for inst in (0, 1):
        rec = run_instance(
            ds, 50, 25,
            adapter_factory=lambda: PyVRPRHDAdapter(max_iterations=300, seed=0),
            inst_idx=inst, objective='coldchain', profile=profile,
            seed=0, data_sha256=c['sha256'],
            adapter_module_path=ADAPTER,
            instance_seed=c['instance_seeds'][inst],
            scene_instance_id=c['scene_instance_ids'][inst])
        new_path = os.path.join(NEW_DIR, 'instances', f'inst_{inst}.json')
        with open(new_path, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False)
        old_path = os.path.join(OLD_DIR, 'r1_02', 'instances',
                                f'inst_{inst}.json')
        old_rec = json.load(open(old_path, encoding='utf-8'))
        match = rec['decision_hash'] == old_rec['decision_hash']
        all_match &= match
        entries.append({
            'instance_id': inst,
            'decision_hash_match': match,
            'decision_hash_rev4': rec['decision_hash'],
            'decision_hash_old': old_rec['decision_hash'],
            'artifact_hash_rev4': rec['artifact_hash'],
            'artifact_hash_old': old_rec['artifact_hash'],
            'new_instance_file_sha256': sha256_file(new_path),
            'old_instance_file_sha256': sha256_file(old_path),
            'code_hash_rev4': rec['code_hash'],
            'code_hash_old': old_rec['code_hash'],
        })

    doc = {
        'schema_version': 'cc-compare-identity-parity-v1',
        'method': 'PyVRP-RH-D',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparator_source_sha256': comparator,
        'old_run': os.path.relpath(OLD_DIR, _DCC),
        'new_run': os.path.relpath(NEW_DIR, _DCC),
        'cells': ['r1_02'],
        'all_match': all_match,
        'verdict': 'BEHAVIOR_UNCHANGED' if all_match else 'BEHAVIOR_CHANGED',
        'instances': entries,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f'identity_rev4_behavior_parity.json written: {OUT}')
    print(f'  comparator_source_sha256 = {comparator}')
    print(f'  verdict = {doc["verdict"]}')
    print(f'  code_hash_rev4 = {entries[0]["code_hash_rev4"]}')


if __name__ == '__main__':
    main()
