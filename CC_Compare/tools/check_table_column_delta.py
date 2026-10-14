#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表列迁移核对器：新增一列时，证明**除该列外逐格未变**。

为什么需要：给宽表插一列会让**每一行文本都位移**，于是 `git diff` 显示"22 行全变"，
这**不能**说明数字没被动过 —— 恰恰是最危险的情形（顺手改了个效用值也看不出来）。
本工具按**列名**对齐两版 Markdown 表，丢掉新列后**逐格字符串比对**，
把"只加了列"与"改了数"区分开。

用法：
    python CC_Compare/tools/check_table_column_delta.py \
        --old <旧表.md> --new <新表.md> --new-col "期刊/会议等级 (CCF)" \
        --allow "OR-Tools-RH accept|期刊/会议" --allow "RouteFinder accept|期刊/会议"

`--allow "<行标签>|<列名>"` 用来声明**本次有意改动**的单元格（必须逐条列出，
不许用通配）；未列出的任何差异都判失败（rc=1）。原有脚注同样不得丢失。
"""
import argparse
import io
import sys


def parse(path):
    head, rows, notes = None, {}, []
    for ln in io.open(path, encoding='utf-8'):
        ln = ln.rstrip('\n')
        if ln.startswith('| # |'):
            head = [c.strip() for c in ln.split('|')[1:-1]]
        elif ln.startswith('| ') and ln[2:3].isdigit():
            cells = [c.strip() for c in ln.split('|')[1:-1]]
            rows[cells[1]] = dict(zip(head, cells))
        elif ln.startswith('- ※'):
            notes.append(ln)
    return head, rows, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--new-col', required=True, help='本次新增的列名（必须恰好这一列）')
    ap.add_argument('--allow', action='append', default=[],
                    help='"<行标签>|<列名>"：本次**有意**改动的单元格，可重复')
    a = ap.parse_args()
    allowed = set()
    for s in a.allow:
        if '|' not in s:
            print('FAIL: --allow 需形如 "<行>|<列>"，实得 %r' % s)
            return 1
        allowed.add(tuple(s.split('|', 1)))

    ho, ro, no = parse(a.old)
    hn, rn, nn = parse(a.new)
    bad = []

    added = [c for c in hn if c not in ho]
    removed = [c for c in ho if c not in hn]
    print('列：旧 %d → 新 %d；新增 %s；删除 %s' % (len(ho), len(hn), added, removed or '无'))
    if added != [a.new_col]:
        bad.append('新增列 %s ≠ 期望 [%s]' % (added, a.new_col))
    if removed:
        bad.append('有列被删除：%s' % removed)
    if set(ro) != set(rn):
        bad.append('行集合不同：仅旧 %s / 仅新 %s'
                   % (sorted(set(ro) - set(rn)), sorted(set(rn) - set(ro))))

    n_cells, diffs, unlisted = 0, [], []
    for k in ro:
        if k not in rn:
            continue
        for c in ho:
            n_cells += 1
            x, y = ro[k].get(c), rn[k].get(c)
            if x != y:
                diffs.append((k, c, x, y))
                if (k, c) not in allowed:
                    unlisted.append((k, c, x, y))
    print('共同列逐格比对：%d 行 × %d 列 = %d 格；不同 %d 格（其中已声明 %d 格）'
          % (len(ro), len(ho), n_cells, len(diffs), len(diffs) - len(unlisted)))
    for k, c, x, y in diffs:
        tag = 'OK(已声明)' if (k, c) in allowed else '✗ 未声明'
        print('   %s %-40s | %-18s | 旧=%s | 新=%s' % (tag, k[:40], c, x, y))
    for k, c, x, y in unlisted:
        bad.append('未声明的单元格改动：%s | %s' % (k, c))

    lost = [n for n in no if n not in nn]
    print('脚注：旧 %d → 新 %d；原有脚注丢失 %d' % (len(no), len(nn), len(lost)))
    for n in lost[:5]:
        print('   ✗ 丢失：%s' % n[:110])
    if lost:
        bad.append('原有脚注丢失 %d 条' % len(lost))
    stale = [t for t in allowed if not any((k, c) == t for k, c, _x, _y in diffs)]
    if stale:
        print('   ⚠️ --allow 里 %d 条并未实际发生（可能已过时）：%s' % (len(stale), sorted(stale)))

    if bad:
        for b in bad:
            print('   ✗ %s' % b)
        print('TABLE_COLUMN_DELTA_FAIL: %d' % len(bad))
        return 1
    print('TABLE_COLUMN_DELTA_PASS：除新增列 `%s` 与 %d 个已声明单元格、'
          '以及新增脚注外，**逐格未变**' % (a.new_col, len(diffs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
