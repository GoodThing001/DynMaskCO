"""生成 OR7 证据包（不可变压缩归档），记录 SHA-256 / 大小 / 恢复说明。

包内容 = OR7-A 六批原始实例 + OR7-B 三批原始实例 + 三份 parity 报告 + 预算选择结果。
包本身因全局 `*.zip` 忽略规则不进入 git；其 SHA-256 记入 EVIDENCE_PACKAGE.json，
并绑定进 FROZEN_CONFIG（seal revision）。

用法：
    python tools/gen_evidence_package.py
"""
import hashlib
import json
import os
import sys
import time
import zipfile

_DCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_DCC, 'results')
OUT = os.path.join(RESULTS, 'or7_evidence_package.zip')
MANIFEST = os.path.join(_DCC, 'EVIDENCE_PACKAGE.json')

CONTENTS = [
    'or7a_9x2_l10_runA', 'or7a_9x2_l10_runB',
    'or7a_9x2_l30_runA', 'or7a_9x2_l30_runB',
    'or7a_9x2_l100_runA', 'or7a_9x2_l100_runB',
    'or7b_9x32_l10', 'or7b_9x32_l30', 'or7b_9x32_l100',
    'or7a_parity_l10.json', 'or7a_parity_l30.json', 'or7a_parity_l100.json',
    'or7_budget_selection.json',
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in CONTENTS:
            p = os.path.join(RESULTS, name)
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in sorted(files):
                        fp = os.path.join(root, fn)
                        z.write(fp, os.path.relpath(fp, RESULTS))
            else:
                z.write(p, os.path.relpath(p, RESULTS))
    size = os.path.getsize(OUT)
    sha = sha256_file(OUT)
    manifest = {
        'schema_version': 'cc-compare-evidence-package-v1',
        'method': 'OR-Tools-RH-D',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'package': 'results/or7_evidence_package.zip',
        'sha256': sha,
        'size_bytes': size,
        'contents': CONTENTS,
        'restore': 'unzip results/or7_evidence_package.zip -d results/',
    }
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'evidence package written: {OUT}')
    print(f'  sha256 = {sha}')
    print(f'  size_bytes = {size}')


if __name__ == '__main__':
    main()
