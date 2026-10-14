#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标准交付：把完成的工件 commit 并 push 到 GitHub（常设约定，2026-10-14 用户要求）。

用法（从仓库根目录执行）：
    python tools/git_deliver.py --message-file <msg.txt>                 # 提交全部改动
    python tools/git_deliver.py --message-file <msg.txt> --paths a b c   # 只提交指定路径
    python tools/git_deliver.py --message-file <msg.txt> --check-only    # 只报告状态

它把"每次完成工作就 push"做成**机械步骤**而不是口头承诺，并强制下列既有规则：

  1. **推送前一致性检查**：`git status` 干净（或只剩声明的排除项）才推；
  2. **排除幽灵子模块** `CC_Compare/MAPT_code`（无 `.gitmodules` 映射的 gitlink，
     内层工作树脏是既有状态，不得顺手 bump）；
  3. **行尾冻结**：对已登记 raw sha256 的字节容器根执行 `git add --renormalize`，
     保证 `blob == 工作树字节`（仓库 `core.autocrlf=true`）；
  4. **只做快进推送**：`origin/main` 必须是 `HEAD` 的祖先，否则**拒绝**推送并报错
     （绝不 force、绝不改写已发布历史）；
  5. **推送后校验**：`origin/<branch>` 与 `origin/main` 都必须等于本地 `HEAD`；
  6. **并发写入者检测**：若最近 N 秒内有其它进程在写工作树（maskco 自动化等），
     默认**拒绝提交**（`--allow-concurrent` 可覆盖），避免把半写状态入库；
  7. 失败一律非零退出并把原因打印清楚；**不静默跳过**推送。

退出码：0 = 已推送并校验通过；2 = 无改动可提交；3 = 前置检查拒绝；4 = 推送/校验失败。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXCLUDE = ['CC_Compare/MAPT_code']
FROZEN_ROOTS = [
    'CC_Compare/tools',
    'research/review_3080_20261010/pickup/a1_d1_resource_identifiability_root_v1',
    'research/review_3080_20261010/delivery/a1_d2_same_mask_attribution_root_v1',
    'research/review_3080_20261010/delivery/cpu_only_8plus2_diagnostic_root_v1',
]
BRANCHES = ['codex/o0cc-cal-phys', 'main']


def run(args, check=True, capture=True):
    p = subprocess.run(['git'] + args, capture_output=capture, text=True,
                       encoding='utf-8', errors='replace')
    if check and p.returncode != 0:
        raise SystemExit('git %s failed (%d):\n%s\n%s'
                         % (' '.join(args), p.returncode, p.stdout, p.stderr))
    return p


def recent_writers(root: Path, seconds: int, ignore=()):
    """Files written within the window, excluding build noise and caller-declared paths.

    `__pycache__` / `*.pyc` are build noise (running any test regenerates them) and are not
    evidence of a concurrent author.  `ignore` lets the caller declare files it authored in
    this same session.
    """
    now = time.time()
    ignore = set(ignore or ())
    hits = []
    for base in ('CC_Compare', 'research', 'C-VRP_Cold-chainVehicleRoutingProblem', 'tools'):
        d = root / base
        if not d.is_dir():
            continue
        for p in d.rglob('*'):
            try:
                if not p.is_file() or now - p.stat().st_mtime >= seconds:
                    continue
            except (OSError, ValueError):
                continue
            rel = str(p.relative_to(root))
            if '__pycache__' in rel or p.suffix == '.pyc' or rel in ignore:
                continue
            hits.append(rel)
            if len(hits) >= 25:
                return hits
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--message-file')
    ap.add_argument('--paths', nargs='*')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--allow-concurrent', action='store_true')
    ap.add_argument('--concurrent-window', type=int, default=90)
    ap.add_argument('--ignore-recent', nargs='*', default=[],
                    help='paths authored by the caller in this session (excluded from the '
                         'concurrent-writer scan)')
    a = ap.parse_args()

    root = Path(run(['rev-parse', '--show-toplevel']).stdout.strip())
    head = run(['rev-parse', '--short', 'HEAD']).stdout.strip()
    print('repo   :', root)
    print('HEAD   :', head, run(['log', '-1', '--format=%s']).stdout.strip()[:90])

    if (root / '.git' / 'index.lock').exists():
        print('REFUSED: .git/index.lock exists (another git process?)')
        return 3
    writers = recent_writers(root, a.concurrent_window, a.ignore_recent)
    if writers and not a.allow_concurrent:
        print('REFUSED: %d file(s) written in the last %ds -> a concurrent writer is active:'
              % (len(writers), a.concurrent_window))
        for w in writers[:10]:
            print('   ', w)
        print('   (this is the maskco automation hazard; re-run with --allow-concurrent only '
              'after verifying the writes are complete and additive)')
        return 3

    status = run(['-c', 'core.quotepath=false', 'status', '--porcelain=v1', '-uall']).stdout
    entries = [l for l in status.splitlines() if l.strip()]
    print('status :', len(entries), 'entr(ies)')
    if not entries:
        print('nothing to commit')
    if a.check_only:
        for l in entries[:30]:
            print('   ', l)
        return 0

    if not entries:
        return 2
    if not a.message_file:
        print('REFUSED: --message-file is required when there is something to commit')
        return 3

    if a.paths:
        run(['add', '--'] + a.paths)
        # only renormalize what was explicitly staged: a global renormalize would sweep in
        # another writer's half-finished files under the byte-container roots
        run(['add', '--renormalize', '--'] + a.paths)
    else:
        run(['add', '-A', '--', '.'] + [':(exclude)%s' % e for e in EXCLUDE])
        for r in FROZEN_ROOTS:
            if (root / r).is_dir():
                run(['add', '--renormalize', '--', r])

    staged = run(['diff', '--cached', '--name-only']).stdout.strip().splitlines()
    for e in EXCLUDE:
        if any(s == e or s.startswith(e) for s in staged):
            print('REFUSED: %s is staged; it must stay excluded' % e)
            return 3
    print('staged :', len(staged), 'file(s)')
    if not staged:
        return 2

    run(['commit', '-F', a.message_file], capture=False)
    new_head = run(['rev-parse', '--short', 'HEAD']).stdout.strip()
    if new_head == head:
        print('REFUSED: commit produced no new HEAD')
        return 3

    run(['fetch', 'origin', '--quiet'])
    for b in BRANCHES:
        anc = subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/%s' % b, 'HEAD'],
                             capture_output=True).returncode
        if anc != 0:
            print('REFUSED: origin/%s is not an ancestor of HEAD -> would need a non-fast-forward '
                  'push; refusing to rewrite published history' % b)
            return 4
    for b in BRANCHES:
        p = run(['push', 'origin', 'HEAD:%s' % b], check=False)
        if p.returncode != 0:
            print('PUSH FAILED for %s:\n%s\n%s' % (b, p.stdout, p.stderr))
            return 4
        print('pushed : HEAD -> %s' % b)

    run(['fetch', 'origin', '--quiet'])
    ok = True
    for b in BRANCHES:
        remote = run(['rev-parse', 'origin/%s' % b]).stdout.strip()
        same = remote == run(['rev-parse', 'HEAD']).stdout.strip()
        print('verify : origin/%-20s %s  %s' % (b, remote[:9], 'MATCH' if same else 'MISMATCH'))
        ok = ok and same
    left = run(['rev-list', '--left-right', '--count', 'origin/main...HEAD']).stdout.strip()
    print('ahead/behind vs main:', left)
    return 0 if ok else 4


if __name__ == '__main__':
    raise SystemExit(main())
