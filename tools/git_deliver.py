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
import json
import os
import stat
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


def _is_reparse_point(p):
    """True for an NTFS junction / symlink directory (`st_file_attributes` reparse bit).

    `os.path.islink()` is NOT enough on Windows: it returns False for a JUNCTION, while pathlib's
    `rglob` happily descends into one.  `_bverify_pinned/shadow/{data,MASKCO_code,results}` are
    junctions back to the extension root, and because the shadow lives UNDER `results/`, one file
    was reachable through 5 nested echo paths (`.../shadow/results/m0_scale/_bverify_pinned/
    shadow/results/...`).  The guard then reported a phantom concurrent writer -- the same 6 real
    files counted 14 times -- and `--ignore-recent` could not clear it, because the echo paths are
    not the paths the caller wrote.
    """
    try:
        st = os.lstat(p)
    except OSError:
        return True                       # unreadable -> do not descend
    attrs = getattr(st, 'st_file_attributes', 0)
    return bool(attrs & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)) or os.path.islink(p)


def _walk_files_no_links(top):
    """Every file under `top`, never descending through a symlink or junction.

    A guard that walks links does not see the repository; it sees the repository N times, and every
    duplicate is a false positive that makes the guard unusable exactly when it is needed.
    """
    for dirpath, dirnames, filenames in os.walk(str(top), followlinks=False):
        dirnames[:] = [d for d in dirnames if not _is_reparse_point(os.path.join(dirpath, d))]
        for f in filenames:
            yield Path(dirpath) / f


def recent_writers(root: Path, seconds: int, ignore=()):
    """Files written within the window, excluding build noise and caller-declared paths.

    `__pycache__` / `*.pyc` are build noise (running any test regenerates them) and are not
    evidence of a concurrent author.  `ignore` lets the caller declare files it authored in
    this same session.  Link and junction directories are NOT descended into (see
    `_is_reparse_point`): an echo path is the same file, not a second writer.
    """
    now = time.time()
    ignore = set(ignore or ())
    hits = []
    for base in ('CC_Compare', 'research', 'C-VRP_Cold-chainVehicleRoutingProblem', 'tools'):
        d = root / base
        if not d.is_dir():
            continue
        for p in _walk_files_no_links(d):
            try:
                if now - p.stat().st_mtime >= seconds:
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
    ap.add_argument('--push-only', action='store_true',
                    help='skip the commit phase entirely and only push HEAD to both branches; '
                         'use when the work is already committed and the remaining worktree '
                         'entries belong to another (concurrent) writer')
    ap.add_argument('--allow-concurrent', action='store_true')
    ap.add_argument('--skip-secret-scan', action='store_true',
                    help='SKIP the staged-content secret scan for a DELIBERATE, REVIEWED exception. '
                         'Never use it to get past a finding you have not resolved.')
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
    raw_entries = [l for l in status.splitlines() if l.strip()]
    # the intentionally excluded phantom submodule must not count as "something to commit",
    # otherwise every invocation demands a --message-file for a path we never stage
    entries = [l for l in raw_entries
               if not any(l.endswith(e) or ('/' + e) in l or e in l for e in EXCLUDE)]
    excluded_seen = len(raw_entries) - len(entries)
    print('status :', len(entries), 'entr(ies)'
          + (' (+%d excluded: %s)' % (excluded_seen, ', '.join(EXCLUDE)) if excluded_seen else ''))
    if not entries:
        print('nothing to commit')
    if a.check_only:
        for l in entries[:30]:
            print('   ', l)
        run(['fetch', 'origin', '--quiet'])
        for b in BRANCHES:
            local = run(['rev-parse', 'HEAD']).stdout.strip()
            remote = run(['rev-parse', 'origin/%s' % b], check=False).stdout.strip()
            print('   origin/%-20s %s  %s' % (b, remote[:9],
                                              'MATCH' if remote == local else 'BEHIND -> push needed'))
        return 0

    # NOTE: "nothing to commit" must NOT short-circuit the push.  Committing and pushing are
    # routinely separated (an earlier call commits, this one pushes), and returning early here
    # would silently leave the work unpushed -- the exact failure the convention forbids.
    committed = False
    if a.push_only:
        print('push-only: skipping the commit phase')
    elif not entries:
        print('nothing to commit; proceeding to the push check')
    else:
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

        staged = run(['-c', 'core.quotepath=false', 'diff', '--cached', '--name-only']).stdout.splitlines()
        for e in EXCLUDE:
            if any(s == e or s.startswith(e) for s in staged):
                print('REFUSED: %s is staged; it must stay excluded' % e)
                return 3

        # SECRET-SHAPE SCAN -- the check whose ABSENCE let two 256-bit execution secrets reach the
        # public repository inside raw run directories.  It runs on the STAGED content (the index is
        # what becomes public, not the worktree) and it REFUSES rather than warns: a scan that only
        # warns is the scan that was missing.  `--skip-secret-scan` exists for a case where the change
        # is a deliberate, reviewed exception; it is recorded loudly when used.
        if a.skip_secret_scan:
            print('WARNING: --skip-secret-scan given; the staged secret scan is being SKIPPED.  '
                  'This must be a deliberate, reviewed exception.')
        else:
            _scanner = root / 'tools' / 'scan_staged_secrets.py'
            if not _scanner.is_file():
                print('REFUSED: the staged secret scanner is missing (%s); refusing to commit '
                      'without it' % _scanner)
                return 4
            _scan = subprocess.run([sys.executable, str(_scanner), '--repo', str(root)],
                                   capture_output=True, text=True)
            if _scan.returncode != 0:
                print('REFUSED: the staged secret scan found credential-shaped content; the finding '
                      'below names the PATH, the LINE and the RULE only -- never the value:')
                print(_scan.stdout.strip())
                print('A refusal is deliberate: this is the control that was missing when two '
                      'execution secrets were published.  Remove or redact the offending content, '
                      'or pass --skip-secret-scan for a reviewed exception.')
                return 4
            try:
                _summary = json.loads(_scan.stdout)
                # COVERAGE IS PRINTED, not assumed.  The scanner used to C-quote non-ASCII staged
                # paths, fail to read those blobs, skip them silently and still be reported here as
                # "CLEAN (0 staged file(s) scanned)" -- a green line over an unread file set.
                print('secret scan: CLEAN (%d/%d staged entr(ies) content-scanned, %d exempted, '
                      '%d gitlink, rules R1-R5)'
                      % (_summary.get('n_staged_scanned', 0), _summary.get('n_staged_entries', 0),
                         _summary.get('n_exempted', 0), _summary.get('n_gitlink_skipped', 0)))
                if not _summary.get('coverage_complete', True):
                    print('REFUSED: the secret scan did not cover every staged entry '
                          '(n_unreadable=%s)' % _summary.get('n_unreadable'))
                    return 4
            except Exception:                                              # noqa: BLE001
                print('secret scan: CLEAN')

        print('staged :', len(staged), 'file(s)')
        if not staged:
            print('nothing staged after add; proceeding to the push check')
        else:
            run(['commit', '-F', a.message_file], capture=False)
            committed = True

    run(['fetch', 'origin', '--quiet'])
    if committed:
        new_head = run(['rev-parse', '--short', 'HEAD']).stdout.strip()
        if new_head == head:
            print('REFUSED: commit produced no new HEAD')
            return 3

    head_sha = run(['rev-parse', 'HEAD']).stdout.strip()
    to_push = []
    for b in BRANCHES:
        remote = run(['rev-parse', 'origin/%s' % b], check=False).stdout.strip()
        if remote != head_sha:
            to_push.append(b)
    if not to_push:
        print('push    : both branches already at HEAD; nothing to push')
        return 0 if committed else 2

    for b in BRANCHES:
        anc = subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/%s' % b, 'HEAD'],
                             capture_output=True).returncode
        if anc != 0:
            print('REFUSED: origin/%s is not an ancestor of HEAD -> would need a non-fast-forward '
                  'push; refusing to rewrite published history' % b)
            return 4
    for b in to_push:
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
