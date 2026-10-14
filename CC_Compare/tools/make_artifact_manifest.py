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

**commit 语义（2026-10-14 第二版；复核指出旧 `target_commit` 语义混淆）**：
旧实现只写一个 `target_commit = git rev-parse HEAD`，把**两件不同的事**挤在一个字段里：

  * `source_commit` —— **快照来源**：这份归档是从哪个提交的树里取出来的。
    它写进入库清单，**必然早于**「包含本清单的那次提交」。
  * `release_commit` —— **发布靶点**：Release 从哪个提交/标签切出去。只有在清单**提交之后**
    才能确定，因此**不能**在 build 时写进入库清单 —— 否则产生**循环自指**
    （清单声称自己是某个 commit，而那个 commit 正是为了记录它才产生的）。

⇒ 现在的做法：
  * 入库清单写 `source_commit`（+ `source_commit_dirty_*` + `self_reference_note`），
    **不再**写 `target_commit`；
  * `release_commit` / `release_tag` 写进**独立的 Release 元数据**（`--release-metadata`，
    落在 `results/` 下、**不入 Git**），并以 `manifest_sha256` **指回**入库清单 ⇒ 单向、无环。
  * `--verify` 在 `source_commit != HEAD` 时**告警**（这是清单自身提交带来的**预期**差异），
    但若 `source_commit` **不是** HEAD 的祖先 ⇒ **判失败**（那才是真漂移）。

用法：
    python CC_Compare/tools/make_artifact_manifest.py                  # 归档 + 生成清单
    python CC_Compare/tools/make_artifact_manifest.py --verify          # 校验清单与文件
    python CC_Compare/tools/make_artifact_manifest.py --release-metadata [--release-tag vX]
退出码：0 = OK；1 = 校验失败；2 = 输入缺失。
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, 'CC_Compare', 'results', 'a1_compare_table')
DST = os.path.join(ROOT, 'CC_Compare', 'docs', 'a1_development_comparison')
MANIFEST_NAME = 'ARTIFACT_MANIFEST.json'
# Release 元数据落在 **results/**（gitignored）⇒ 只随 Release 附件走，不入 Git。
RELEASE_META = os.path.join(SRC, 'RELEASE_METADATA.json')

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
# 判定「快照来源是否干净」的路径范围。
#
# ⚠️ 必须只看**入库快照的目的地**（`docs/a1_development_comparison/`），理由：
#   * 六个 Release 附件与三份快照的**源**都在 `CC_Compare/results/**`，而该目录被
#     `.gitignore` 忽略 ⇒ 对它们做 `git status` 恒为空 ⇒ 会**假报"干净"**，毫无意义；
#   * 入库快照的**目的地**是 git 跟踪的 ⇒ `git status` 在那里**确实**回答了我们真正要问的问题：
#     「本次归档出的快照是否已经提交」。
#   * 但清单文件本身也在这个目录里，且归档时刚被写入 ⇒ 必须**排除**它，否则该字段恒 >0，
#     永远到不了"干净"这个不动点。
# ⇒ 语义：`source_commit_dirty_n > 0` 表示「快照内容尚未提交，快照内容 ≠ `source_commit` 的树」；
#    `= 0` 表示「快照内容恰等于 `source_commit` 的树」（清单自身除外）。
DIRTY_SCOPE = 'CC_Compare/docs/a1_development_comparison'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _git(*args):
    """跑一条 git 命令；不可用/失败一律返回 None（**不编造**）。"""
    try:
        r = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                        # noqa: BLE001
        return None


def _source_commit():
    """`source_commit` = 快照来源提交 = 归档时刻的 HEAD。"""
    return _git('rev-parse', 'HEAD')


def _dirty_files():
    """入库快照目录下**已跟踪但未提交**的改动（快照内容是否已提交）。

    排除清单文件自身（归档时刚写入，否则该字段恒 >0，永远到不了干净的不动点）。
    返回 `(n, 前若干条)`；git 不可用 ⇒ `(None, None)`（如实缺失，不写 0 假装干净）。
    """
    out = _git('status', '--porcelain', '--', DIRTY_SCOPE)
    if out is None:
        return None, None
    lines = [l for l in out.splitlines()
             if l.strip() and not l.rstrip().endswith(MANIFEST_NAME)]
    return len(lines), lines[:5]


def _is_ancestor(a, b):
    """a 是否为 b 的祖先（`merge-base --is-ancestor`；rc=0 即真）。"""
    try:
        r = subprocess.run(['git', 'merge-base', '--is-ancestor', a, b], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:                                        # noqa: BLE001
        return None


def build():
    if not os.path.isdir(SRC):
        print('FAIL: 源目录不存在: %s' % SRC)
        return 2
    os.makedirs(DST, exist_ok=True)
    # ⚠️ 2026-10-14 用户核实：旧实现遇到缺失附件**直接跳过** ⇒ 清单可以"少件"却仍报成功；
    # `--verify` 也只验入库 Markdown。现在**六份附件逐个要求存在**，缺一即失败。
    missing = [s for s in RELEASE_ONLY if not os.path.isfile(os.path.join(SRC, s))]
    if missing:
        print('FAIL: Release 附件缺失 %d 份（发布包不完整）: %s' % (len(missing), missing))
        return 2
    man = {
        'kind': 'development-evidence-package',
        'note': ('**development 证据包，不是最终测试结果，也不声称 SOTA。** '
                 'Markdown 文档入 Git；Excel 与完整 dataset.json 仅作 Release 附件，'
                 '`CC_Compare/results/**` 的忽略规则保持不变。'),
        # —— 快照来源（入库清单里唯一的 commit 字段）——
        'source_commit': _source_commit(),
        # ⚠️ 这两个字段必须**在复制之后**填（见下方赋值），否则会把"复制之前"的状态当成
        # 快照内容的状态：本次归档刚写出的新快照尚未提交，却会被报告成 dirty=0、
        # 进而错误地声称"快照内容恰等于 source_commit 的树"。
        'source_commit_dirty_n': None,
        'source_commit_dirty_files': None,
        'source_commit_dirty_scope': DIRTY_SCOPE,
        'source_commit_dirty_meaning': (
            '入库快照目录下**未提交**的已跟踪文件数（**已排除清单自身**）。'
            '`=0` ⇒ 快照内容恰等于 `source_commit` 的树；`>0` ⇒ 快照内容 = `source_commit` '
            '**加上**这些改动，此时**不得**把 `source_commit` 当作快照内容的定位。'
            '⚠️ 该判据**不覆盖** `results/**`（被 .gitignore 忽略 ⇒ 对其做 git status 恒为空，'
            '会假报干净）；Release 附件的完整性由 `release_only` 的 size+sha256 保证。'),
        'self_reference_note': (
            '`source_commit` 是**快照来源**（归档时刻 HEAD）。包含本清单的那次提交是它的'
            '**子**提交 ⇒ `--verify` 时 `source_commit != HEAD` 是**预期**的，不是漂移；'
            '真正要判的是 `source_commit` 是否为 HEAD 的祖先。'
            '快照**内容**的定位：看 `source_commit_dirty_n` —— 为 0 时内容 = `source_commit` 的树，'
            '否则再加上所列改动；无论哪种，最终判据都是本清单记录的 `sha256`。'),
        # —— 发布靶点**不在**入库清单里（避免循环自指）——
        'release_commit': None,
        'release_commit_note': ('发布靶点（tag/commit）在**清单提交之后**才能确定，'
                                '记录于独立 Release 元数据 %s（不入 Git，以 '
                                '`manifest_sha256` 指回本清单 ⇒ 单向无环）。'
                                '用 `--release-metadata` 生成。'
                                % os.path.relpath(RELEASE_META, ROOT).replace('\\', '/')),
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
        man['release_only'].append({
            'name': s, 'in_repo': False,
            'size': os.path.getsize(sp), 'sha256': sha256(sp),
            'note': 'Release 附件；不入 Git；**发布前必须逐个校验 size+sha256**'})
    # —— 快照已写出，**现在**才能判"快照内容是否已提交" ——
    dn, dlines = _dirty_files()
    man['source_commit_dirty_n'], man['source_commit_dirty_files'] = dn, dlines
    with open(os.path.join(DST, MANIFEST_NAME), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print('source_commit = %s（dirty=%s）' % (str(man['source_commit'])[:12], dn))
    print('归档 %d 份 Markdown → %s' % (len(man['in_repo']), os.path.relpath(DST, ROOT)))
    for e in man['in_repo']:
        print('   %-46s %8d B  %s' % (e['path'].split('/')[-1], e['size'], e['sha256'][:16]))
    print('Release 附件（不入 Git）%d 份：' % len(man['release_only']))
    for e in man['release_only']:
        print('   %-34s %10d B  %s' % (e['name'], e['size'], e['sha256'][:16]))
    print('⚠️ release_commit **未**记录（入库清单里恒为 None，避免循环自指）。')
    print('   发布时再跑：make_artifact_manifest.py --release-metadata [--release-tag <tag>]')
    return 0


def release_metadata(release_tag=None):
    """生成**独立** Release 元数据：把发布靶点写给入库清单，并指回清单哈希（无环）。"""
    mp = os.path.join(DST, MANIFEST_NAME)
    if not os.path.isfile(mp):
        print('FAIL: 缺 %s（先跑一次归档）' % MANIFEST_NAME)
        return 2
    man = json.load(open(mp, encoding='utf-8'))
    head = _git('rev-parse', 'HEAD')
    src = man.get('source_commit')
    if not head:
        print('FAIL: 取不到 HEAD（git 不可用）⇒ 不编造 release_commit')
        return 2
    anc = _is_ancestor(src, head) if src else None
    if src and anc is False:
        print('FAIL: source_commit=%s **不是** HEAD=%s 的祖先 ⇒ 快照来源不在发布线上'
              % (str(src)[:12], str(head)[:12]))
        return 1
    meta = {
        'kind': 'development-release-metadata',
        'release_commit': head,
        'release_tag': release_tag,
        'manifest_path': os.path.relpath(mp, ROOT).replace('\\', '/'),
        'manifest_sha256': sha256(mp),
        'source_commit': src,
        'source_commit_is_ancestor_of_release': anc,
        'source_commit_dirty_n': man.get('source_commit_dirty_n'),
        'note': ('**发布靶点**在此（不在入库清单里，避免循环自指）。'
                 '`manifest_sha256` 指向入库清单 ⇒ Release 包可单向校验：'
                 'release_commit ▸ manifest ▸ 三份 Markdown 快照 + 六份附件。'),
        'release_only': man.get('release_only') or [],
    }
    with open(RELEASE_META, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('release_commit = %s（tag=%s）' % (head[:12], release_tag))
    print('source_commit  = %s（是其祖先=%s）' % (str(src)[:12], anc))
    print('manifest_sha256 = %s' % meta['manifest_sha256'][:16])
    print('wrote %s（不入 Git，随 Release 附件发布）'
          % os.path.relpath(RELEASE_META, ROOT))
    return 0


def verify():
    """校验入库快照 + 六份 Release 附件，并核对 commit 语义（**区分告警与失败**）。"""
    mp = os.path.join(DST, MANIFEST_NAME)
    if not os.path.isfile(mp):
        print('FAIL: 缺 %s（先跑一次不带 --verify）' % MANIFEST_NAME)
        return 2
    man = json.load(open(mp, encoding='utf-8'))
    bad, warn = [], []
    for e in man.get('in_repo') or []:
        p = os.path.join(ROOT, *e['path'].split('/'))
        if not os.path.isfile(p):
            bad.append('缺文件 %s' % e['path'])
        elif os.path.getsize(p) != e['size'] or sha256(p) != e['sha256']:
            bad.append('size/哈希不符 %s（快照被改动过？）' % e['path'])
    src = os.path.join(ROOT, *str(man.get('source_dir') or '').split('/'))
    rel = man.get('release_only') or []
    if len(rel) != len(RELEASE_ONLY):
        bad.append('Release 附件条目数 %d != 预定 %d' % (len(rel), len(RELEASE_ONLY)))
    for e in rel:
        p = os.path.join(src, e['name'])
        if not os.path.isfile(p):
            bad.append('Release 附件缺失 %s' % e['name'])
        elif os.path.getsize(p) != e['size'] or sha256(p) != e['sha256']:
            bad.append('Release 附件 size/哈希不符 %s' % e['name'])

    # ---- commit 语义 ----
    sc = man.get('source_commit')
    if not sc:
        bad.append('清单未记录 source_commit')
    if 'target_commit' in man:
        bad.append('清单仍含已废弃的 `target_commit`（语义混淆，应只留 '
                   '`source_commit` + 独立 Release 元数据）')
    head = _git('rev-parse', 'HEAD')
    if sc and head:
        if sc == head:
            warn.append('source_commit == HEAD：本清单**尚未提交**（清单自身所在提交还不存在），'
                        '提交后会出现预期的 `source_commit != HEAD`')
        elif _is_ancestor(sc, head) is False:
            bad.append('source_commit=%s **不是** HEAD=%s 的祖先 ⇒ 真漂移（快照来源不在当前线上）'
                       % (sc[:12], head[:12]))
        else:
            warn.append('source_commit=%s ≠ HEAD=%s（**预期**：清单自身所在提交是其子提交，'
                        '已核实为祖先关系，非漂移）' % (sc[:12], head[:12]))
    elif not head:
        warn.append('git 不可用 ⇒ 无法核对 source_commit 与 HEAD 的关系')
    dn = man.get('source_commit_dirty_n')
    if dn:
        warn.append('source_commit 时刻 %s 下有 %d 个已跟踪文件未提交（快照来源不干净）：%s'
                    % (DIRTY_SCOPE, dn, man.get('source_commit_dirty_files')))
        warn.append('⇒ 快照内容 = `source_commit` **加上**清单所在提交携带的改动，'
                    '**不等于** `source_commit` 的树；定位内容请看**清单所在提交**'
                    '（发布时即 `release_commit`），最终以清单 `sha256` 为准')

    # ---- 独立 Release 元数据（未发布不算失败）----
    rc_meta = None
    if os.path.isfile(RELEASE_META):
        rc_meta = json.load(open(RELEASE_META, encoding='utf-8'))
        if rc_meta.get('manifest_sha256') != sha256(mp):
            bad.append('RELEASE_METADATA.manifest_sha256 与当前清单**不符** ⇒ 清单在打标签后'
                       '被改动过（Release 记录已过期）')
        rcm = rc_meta.get('release_commit')
        if not rcm:
            bad.append('RELEASE_METADATA 缺 release_commit')
        elif _git('cat-file', '-t', rcm) != 'commit':
            bad.append('release_commit=%s 在仓库中不存在或不是 commit' % str(rcm)[:12])
        if rc_meta.get('source_commit') != sc:
            bad.append('RELEASE_METADATA.source_commit (=%s) 与清单 (=%s) 不一致'
                       % (str(rc_meta.get('source_commit'))[:12], str(sc)[:12]))
    else:
        warn.append('未生成独立 Release 元数据（尚未发布）⇒ release_commit **未记录**；'
                    '发布时跑 `--release-metadata`')

    for w in warn:
        print('   ⚠️ %s' % w)
    if bad:
        for b in bad:
            print('   ✗ %s' % b)
        print('ARTIFACT_MANIFEST_VERIFY_FAIL: %d' % len(bad))
        return 1
    print('ARTIFACT_MANIFEST_VERIFY_PASS（入库快照 %d 份 + Release 附件 %d 份；'
          'source_commit=%s；release_commit=%s；告警 %d 条）'
          % (len(man['in_repo']), len(rel), str(sc)[:12],
             (str(rc_meta.get('release_commit'))[:12] if rc_meta else '未记录'), len(warn)))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--release-metadata', action='store_true', dest='release_metadata')
    ap.add_argument('--release-tag', default=None)
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify())
    if a.release_metadata:
        sys.exit(release_metadata(a.release_tag))
    sys.exit(build())
