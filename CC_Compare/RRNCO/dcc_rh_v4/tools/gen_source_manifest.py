"""RRNCO-Ordering-RH-D R1 冻结：源码清单生成器（tools/gen_source_manifest.py）。

对 adapter 目录全部源码按 logical path（相对 dcc_rh_v4/）计算 sha256，分组
control / analysis / identity，输出 SOURCE_MANIFEST.json。可重跑复验（确定性）。

用法（仓库根）：
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/RRNCO/dcc_rh_v4/tools/gen_source_manifest.py
    （本地任意 python3 亦可，纯 stdlib）
"""
import hashlib
import json
import os
import sys

_CONTROL_FILES = (
    'subproblem.py',
    'preference.py',
    'coordinator.py',
    'pickup_certificate.py',
    'rrnco_backend.py',
    'rrnco_guided_adapter.py',
)
_ANALYSIS_FILES = (
    'run_r0_5.py',
    'b2_injection_gate.py',
    'verify_r0_5.py',
    'server_preflight.py',
    'r0_5_snapshots.py',
    'testutil.py',
)
_IDENTITY_FILES = (
    'identity.py',
    'tools/gen_source_manifest.py',
    'verify_freeze.py',
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    dcc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    groups = {}
    for gname, files in (('control', _CONTROL_FILES),
                         ('analysis', _ANALYSIS_FILES),
                         ('identity', _IDENTITY_FILES)):
        g = {}
        for f in files:
            p = os.path.join(dcc, f)
            if not os.path.exists(p):
                print('MISSING:', f, file=sys.stderr)
                sys.exit(1)
            g[f] = sha256_file(p)
        groups[gname] = g
    manifest = {
        'schema_version': 'rrnco-dcc-rh-v4-source-manifest-v1',
        'generated_by': 'tools/gen_source_manifest.py',
        'files': {f: h for g in groups.values() for f, h in g.items()},
        'groups': groups,
        'note': 'RRNCO-Ordering-RH-D（旧 strict-online DCC-VRP 协议）R1 冻结源码清单。'
                '上游 ../rrnco/、../../common/ 与 ../../../MASKCO_code/ 不在此清单'
                '（冻结边界外，只读）。',
    }
    out = os.path.join(dcc, 'SOURCE_MANIFEST.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    print('SOURCE_MANIFEST.json written:', out)
    print('  control :', len(groups['control']), 'files')
    print('  analysis:', len(groups['analysis']), 'files')
    print('  identity:', len(groups['identity']), 'files')


if __name__ == '__main__':
    main()
