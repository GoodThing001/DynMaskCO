#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档同步门：README §2/§8/§10 散文里的**个数与极值**是否仍与数据一致。

背景（为什么必须有它）：
  2026-10-13 `print_readme_section2.py` 首版把 OR-Tools 混进「学习型外部」best
  （+277.3 而非 +311.7）、又把 Ours−C1 用了错口径（−125.8 而非 gate 内 −116.7）。
  那次是**人工发现**的。v2 批次落地后 `coenrich`/`mapt` 会真正进表
  ⇒ 「13 个外部方法」必须变成「15 个」、若干极值也可能换人 ——
  这类「个数/极值」散文一旦漏改，读起来完全正常但**是错的**。

  所以本门唯一职责：**从权威生成器取数，再回头核对散文**。
  取数一律走 `print_readme_section2.py`（它已含 `formal` 过滤 + 批次 RANK 的
  正确选臂口径），**本文件不重新实现任何选臂逻辑** —— 这正是上一版
  `_doc_sync_after_v2.py` 的教训：自己再写一套口径，就会自己算错。

用法（仓库根）：
    python CC_Compare/tools/_doc_prose_check.py \
        --data CC_Compare/results/a1_compare_table/dataset.json \
        --readme CC_Compare/results/a1_compare_table/README.md

退出码：0 = 全部一致（允许 warning）；2 = 有不一致（必须改散文后再交付）。
"""
import argparse
import os
import re
import subprocess
import sys

# 自保（2026-10-14 实测）：中文 Windows 控制台默认 GBK，而本脚本打印 `−`(U+2212)/`⇒`
# ⇒ 直接跑会 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2212'`，
# 把**通过**的门禁报成 rc=1。必须在任何 print 之前重绑 stdout。
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                                errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))

sys.path.insert(0, HERE)
from collect_compare_table import METHOD_META  # noqa: E402


def section2_numbers(data_rel):
    """跑权威生成器，解析出需要的数。"""
    # 子进程必须显式继承/覆盖 PYTHONIOENCODING=utf-8：中文 Windows 上父进程未设时，
    # 子进程 stdout 会是 GBK，中文/符号会被写成 GBK 字节，而这里按 utf-8 解码
    # ⇒ `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbb` 把**通过**的门禁报成失败。
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    r = subprocess.run([sys.executable, os.path.join(HERE, 'print_readme_section2.py'),
                        '--data', data_rel],
                       cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env)
    if r.returncode != 0:
        raise SystemExit('print_readme_section2.py 失败 rc=%s\n%s' % (r.returncode, r.stderr))
    out = r.stdout
    num = {}

    def grab(pattern, key, cast=float):
        m = re.search(pattern, out)
        if m:
            num[key] = cast(m.group(1))

    grab(r'「N 个外部方法」= (\d+)', 'n_external', int)
    grab(r'学习型 (\d+) ；经典求解器：(\d+)', 'n_learned', int)
    m = re.search(r'最好\*\*学习型\*\*外部（p_c=0）：(\S+) = ([\d.\-]+)', out)
    if m:
        num['best_learned_method'] = m.group(1)
        num['best_learned'] = float(m.group(2))
    m = re.search(r'Ours（MaskCO A3 pretrained，p_c=0）= ([\d.\-]+)', out)
    if m:
        num['ours0'] = float(m.group(1))
    grab(r'Ours − 最好\*\*学习型\*\*外部 = ([\+\-\d.]+)', 'd_learned')
    grab(r'Ours − 最好外部（含 classic）= ([\+\-\d.]+)', 'd_classic')
    grab(r'Ours − 最好规则 = ([\+\-\d.]+)', 'd_rule')
    grab(r'Ours − \*\*该 gate 内\*\* cond_hist = ([\+\-\d.]+)', 'd_c1_gate')
    grab(r'规则臂：(\d+)', 'n_rule', int)
    return num, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='CC_Compare/results/a1_compare_table/dataset.json')
    ap.add_argument('--readme', default='CC_Compare/results/a1_compare_table/README.md')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    data_abs = args.data if os.path.isabs(args.data) else os.path.join(ROOT, *args.data.split('/'))
    readme_abs = args.readme if os.path.isabs(args.readme) else os.path.join(ROOT, *args.readme.split('/'))
    rel_for_script = os.path.relpath(data_abs, ROOT).replace('\\', '/')

    num, out = section2_numbers(rel_for_script)
    if not args.quiet:
        print('== 权威生成器给出的数（print_readme_section2.py）==')
        for k in ('n_learned', 'n_external', 'n_rule', 'best_learned_method', 'best_learned',
                  'ours0', 'd_learned', 'd_classic', 'd_rule', 'd_c1_gate'):
            if k in num:
                print('  %-20s %s' % (k, num[k]))
        print()

    with open(readme_abs, encoding='utf-8') as f:
        text = f.read()

    fails, warns = [], []

    n_ext = num.get('n_external')
    if n_ext is None:
        fails.append('生成器未给出 n_external，无法核对')
    else:
        found = sorted({int(x) for x in re.findall(r'(\d+) 个外部方法', text)})
        if found == [n_ext]:
            print('OK   「N 个外部方法」：README 中的全部取值 %s == 生成器 %d' % (found, n_ext))
        elif not found:
            warns.append('README 里没有出现「N 个外部方法」句式（可能措辞变了）')
        else:
            fails.append('「N 个外部方法」不一致：README 写了 %s，生成器说 %d'
                         % (found, n_ext))
        # 另一种措辞：把规则臂并进来数（历史上写成「13 个外部与规则对照」，口径含糊）
        m2 = re.findall(r'(\d+) 个外部与规则对照', text)
        if m2:
            tot = n_ext + (num.get('n_rule') or 0)
            if set(m2) == {str(tot)}:
                print('OK   「N 个外部与规则对照」= %s == %d+%d=%d' % (m2, n_ext, num.get('n_rule') or 0, tot))
            else:
                warns.append('「N 个外部与规则对照」= %s，按生成器应为 %d（外部 %d + 规则 %d）'
                             '—— 口径含糊，建议改写为「全部 %d 个外部方法与 %d 条规则对照」'
                             % (m2, tot, n_ext, num.get('n_rule') or 0, n_ext, num.get('n_rule') or 0))

    bl_m, bl_v = num.get('best_learned_method'), num.get('best_learned')
    if bl_m and bl_v is not None:
        label = (METHOD_META.get(bl_m) or {}).get('label') or bl_m
        pat = re.compile(re.escape(label) + r'[^0-9\n]{0,12}' + re.escape('%.1f' % bl_v))
        if pat.search(text):
            print('OK   「最好学习型外部」= %s %.1f 在 README 中出现' % (label, bl_v))
        else:
            fails.append('README 里找不到「最好学习型外部 = %s %.1f」'
                         '（v2 后极值可能换人，必须同步改散文）' % (label, bl_v))

    d = num.get('d_learned')
    if d is not None:
        s = '%+.1f' % d
        if s in text:
            print('OK   「Ours − 最好学习型外部」= %s 在 README 中出现' % s)
        else:
            fails.append('README 里找不到 Ours − 最好学习型外部 = %s' % s)

    for key, human in (('d_c1_gate', 'Ours − 该 gate 内 cond_hist'),
                       ('d_classic', 'Ours − 最好外部（含 classic）')):
        v = num.get(key)
        if v is None:
            continue
        s = '%.1f' % abs(v)
        if s in text:
            print('OK   「%s」= %+.1f 的数字 %s 在 README 中出现' % (human, v, s))
        else:
            warns.append('README 里没找到 %s = %+.1f（若该口径未被引用可忽略）' % (human, v))

    print()
    if warns:
        for w in warns:
            print('WARN %s' % w)
    if fails:
        for x in fails:
            print('FAIL %s' % x)
        print('DOC_PROSE_CHECK_FAIL: %d 项不一致（改完散文重跑本门）' % len(fails))
        return 2
    print('DOC_PROSE_CHECK_PASS%s' % ('（有 warning）' if warns else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
