#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""受控归档（文档与结果分离，2026-10-14 用户裁决）。

**为什么要分离**：`CC_Compare/results/**` 被 `.gitignore` 忽略（有意——实验证据不入库），
所以交付表与 README 在 GitHub 上**永远看不到**。用户裁决：
  * **Markdown 文档入 Git** → `CC_Compare/docs/a1_development_comparison/`
  * **Excel 与完整 `dataset.json` 不入 Git**，作为独立「development」证据包（Release 附件）
  * **不解除** `results/**` 的忽略规则

本工具负责**复制 + 生成清单**（可重复执行，不是手工拷贝）：
  * 复制三份 Markdown 快照；
  * 为**入库快照**与**Release 附件**分别计算 SHA-256；
  * 写 `ARTIFACT_MANIFEST.json`（含生成命令、来源路径、大小、哈希）。

用法：
    python CC_Compare/tools/make_artifact_manifest.py            # 归档 + 生成清单
    python CC_Compare/tools/make_artifact_manifest.py --verify    # 只校验快照哈希是否与清单一致
退出码：0 = OK；1 = 校验失败；2 = 输入缺失。
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, 'CC_Compare', 'results', 'a1_compare_table')
DST = os.path.join(ROOT, 'CC_Compare', 'docs', 'a1_development_comparison')

# 入库的 Markdown 快照（源文件名 → 目标相对名）
IN_REPO = [
    ('README.md', 'README.md'),
    ('A-v1对比方法总表.md', 'A-v1对比方法总表.md'),
    ('读表注意与已知限定.md', '读表注意与已知限定.md'),
]
# 只作 Release 附件、**不入 Git** 的产物（含大文件）
RELEASE_ONLY = [
    'A-v1对比方法总表.xlsx',
    'dataset.json',
    'A-v1对比方法效用图.png',
    'TIMEOUT_SCAN.md',
    'V2批次结果.md',
    'README_SECTION2_paste.md',
]
GEN = ('python CC_Compare/tools/collect_compare_table.py --json '
       'CC_Compare/results/a1_compare_table/dataset.json  &&  '
       'python CC_Compare/tools/render_compare_table.py --data '
       'CC_Compare/results/a1_compare_table/dataset.json --outdir '
       'CC_Compare/results/a1_compare_table')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def build():
    if not os.path.isdir(SRC):
        print('FAIL: 源目录不存在: %s' % SRC)
        return 2
    os.makedirs(DST, exist_ok=True)
    man = {
        'kind': 'development-evidence-package',
        'note': ('**development 证据包，不是最终测试结果，也不声称 SOTA。** '
                 'Markdown 文档入 Git；Excel 与完整 dataset.json 仅作 Release 附件，'
                 '`CC_Compare/results/**` 的忽略规则保持不变。'),
        'source_dir': os.path.relpath(SRC, ROOT).replace('\\', '/'),
        'generated_by': GEN,
        'in_repo': [],
        'release_only': [],
    }
    for s, d in IN_REPO:
        sp, dp = os.path.join(SRC, s), os.path.join(DST, d)
        if not os.path.isfile(sp):
            print('FAIL: 缺源文件 %s' % sp)
            return 2
        shutil.copy2(sp, dp)
        man['in_repo'].append({
            'path': 'CC_Compare/docs/a1_development_comparison/%s' % d,
            'source': '%s/%s' % (man['source_dir'], s),
            'size': os.path.getsize(dp), 'sha256': sha256(dp)})
    for s in RELEASE_ONLY:
        sp = os.path.join(SRC, s)
        if not os.path.isfile(sp):
            continue
        man['release_only'].append({
            'name': s, 'in_repo': False,
            'size': os.path.getsize(sp), 'sha256': sha256(sp),
            'note': 'Release 附件；不入 Git'})
    with open(os.path.join(DST, 'ARTIFACT_MANIFEST.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print('归档 %d 份 Markdown → %s' % (len(man['in_repo']), os.path.relpath(DST, ROOT)))
    for e in man['in_repo']:
        print('   %-46s %8d B  %s' % (e['path'].split('/')[-1], e['size'], e['sha256'][:16]))
    print('Release 附件（不入 Git）%d 份：' % len(man['release_only']))
    for e in man['release_only']:
        print('   %-34s %10d B  %s' % (e['name'], e['size'], e['sha256'][:16]))
    return 0


def verify():
    mp = os.path.join(DST, 'ARTIFACT_MANIFEST.json')
    if not os.path.isfile(mp):
        print('FAIL: 缺 ARTIFACT_MANIFEST.json（先跑一次不带 --verify）')
        return 2
    man = json.load(open(mp, encoding='utf-8'))
    bad = []
    for e in man.get('in_repo') or []:
        p = os.path.join(ROOT, *e['path'].split('/'))
        if not os.path.isfile(p):
            bad.append('缺文件 %s' % e['path'])
        elif sha256(p) != e['sha256']:
            bad.append('哈希不符 %s（快照被改动过？）' % e['path'])
    if bad:
        for b in bad:
            print('   ✗ %s' % b)
        print('ARTIFACT_MANIFEST_VERIFY_FAIL: %d' % len(bad))
        return 1
    print('ARTIFACT_MANIFEST_VERIFY_PASS（%d 份入库快照哈希一致）' % len(man['in_repo']))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    sys.exit(verify() if a.verify else build())
