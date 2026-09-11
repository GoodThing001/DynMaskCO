"""生成 v2→v3 行为不变报告（OR6.2 收口项 3）。

对比 devproto_9x2_v2（旧控制链，code_hash 1a1ca353）与 devproto_9x2_v3（新控制链，
code_hash 29c10e92）的 decision_hash，证明控制层变化（run_id 唯一 / pre-run 复验 /
落盘完整 common 合同 / checkpoint 完整 64 位 / 三层身份）没有改变求解行为。

只读现有产物，不重新运行实例。输出 results/decision_parity_v2_to_v3.json，
绑定：比较器源码 hash、两边实例文件 sha256、两边 compute hash、每实例 decision hash。

用法：
    python tools/gen_v2_v3_parity.py
"""
import hashlib
import json
import os
import sys
import time

_DCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DCC)

CELLS = [f'{t}_{ed}' for t in ('r1', 'c1', 'rc1') for ed in ('02', '05', '08')]
RUN_A = os.path.join(_DCC, 'results', 'devproto_9x2_v2')
RUN_B = os.path.join(_DCC, 'results', 'devproto_9x2_v3')
OUT = os.path.join(_DCC, 'results', 'decision_parity_v2_to_v3.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    comparator = sha256_file(os.path.abspath(__file__))
    pre_a = json.load(open(os.path.join(RUN_A, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    pre_b = json.load(open(os.path.join(RUN_B, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    entries = []
    all_match = True
    for cell in CELLS:
        for inst in (0, 1):
            pa = os.path.join(RUN_A, cell, 'instances', f'inst_{inst}.json')
            pb = os.path.join(RUN_B, cell, 'instances', f'inst_{inst}.json')
            ra = json.load(open(pa, encoding='utf-8'))
            rb = json.load(open(pb, encoding='utf-8'))
            match = ra['decision_hash'] == rb['decision_hash']
            all_match &= match
            entries.append({
                'cell': cell, 'instance_id': inst,
                'decision_hash_match': match,
                'decision_hash_v2': ra['decision_hash'],
                'decision_hash_v3': rb['decision_hash'],
                'file_sha256_v2': sha256_file(pa),
                'file_sha256_v3': sha256_file(pb),
            })
    doc = {
        'schema_version': 'cc-compare-decision-parity-v1',
        'method': 'OR-Tools-RH-D',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparator_source_sha256': comparator,
        'run_v2': os.path.relpath(RUN_A, _DCC),
        'run_v3': os.path.relpath(RUN_B, _DCC),
        'compute_hash_v2': pre_a.get('compute_hash'),
        'compute_hash_v3': pre_b.get('compute_hash'),
        'n_instances': len(entries),
        'all_match': all_match,
        'verdict': ('BEHAVIOR_UNCHANGED_18_OF_18' if all_match
                    else 'BEHAVIOR_CHANGED'),
        'instances': entries,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f'decision_parity_v2_to_v3.json written: {OUT}')
    print(f'  comparator_source_sha256 = {comparator}')
    print(f'  verdict = {doc["verdict"]}')
    print(f'  compute_hash v2 = {pre_a.get("compute_hash")[:16]} '
          f'v3 = {pre_b.get("compute_hash")[:16]}')


if __name__ == '__main__':
    main()
