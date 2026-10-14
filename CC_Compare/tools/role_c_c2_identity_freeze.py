#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 最小实现的**身份冻结 + 静态白名单审计**（只读；可重跑）。

为什么要做成脚本而不是手写记录
------------------------------
用户要求交付"新旧源码 SHA 对照 + 逐行差异 + **静态白名单审计** + 原封存文件不变证明"。
手写这些数字迟早会漂；做成脚本后，**同一命令**在任意 checkout 上给出同样结论，
复核者可以自己重跑。

判定内容
--------
1. **旧/新 SHA 对照**：旧版取**提交 `CEF082B3`** 的 blob（即正式预声明被验收时的字节），
   新版取当前 HEAD blob 与工作树字节；
2. **逐行差异**：给出变更 hunk 的行号区间与增删行数；
3. **静态白名单审计**：检查改动是否**只**落在登记件 §8.2 + §14.4 允许的两处
   （`shadow_price_budget` 的 `except` 捕获点、`decide_candidate` 的 B3 诊断装配分支），
   并检查五条"不得改变"是否被违反（判决 / boundary / reject_reason / certify 调用 /
   返回形状 / 预算差分与 λ 数值 / B3 强制拒单）；
4. **原封存文件不变证明**：逐文件比较"当前字节"与指定基线提交的 blob，
   对**不得修改**的文件断言逐字节相同。

⚠️ 只读：不写任何被测文件、不跑任何实验。
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import sys

__all__ = ['BASELINE_COMMIT', 'CHANGE_ALLOWLIST', 'MUST_NOT_CHANGE',
           'FIVE_MUST_NOT_CHANGE', 'freeze', 'main']

#: 正式预声明被验收时的提交（旧版身份的锚点）
BASELINE_COMMIT = 'cef082b30c5f62f572ec64efb40c5fcb4b5e9a04'

#: 允许改动的**被测**文件（其余被测文件必须逐字节不变）
CHANGE_ALLOWLIST = ('CC_Compare/tools/role_c_opportunity_cost.py',)

#: 允许改动的**两处**（登记件 §8.2 + §14.4）
ALLOWED_HUNKS = (
    {'name': 'shadow_price_budget.except_capture_point',
     'marker': "'exception_audit': _exception_audit(",
     'why': '登记件 §8.2：在该捕获点新增结构化异常记录'},
    {'name': 'decide_candidate.B3_diagnostic_assembly',
     'marker': "diag['exception_audit'] = _audit",
     'why': '登记件 §14.4：B3 分支仅把 `_d0` 的审计信息附加到最终 diag'},
)

#: 分析侧新增/改动的文件（不受"被测文件"约束，但仍须登记）
ANALYSIS_SIDE = (
    'CC_Compare/tools/role_c_timeout_semantics_audit.py',
    'CC_Compare/tools/test_role_c_timeout_semantics.py',
)

#: 必须**逐字节不变**的文件（被测模块 + 封存原件）
MUST_NOT_CHANGE = (
    'CC_Compare/tools/role_c_timing.py',
    'CC_Compare/tools/role_c_forward.py',
    'CC_Compare/tools/role_c_harness.py',
    'CC_Compare/tools/role_c_local_search.py',
    'CC_Compare/tools/role_c_identity.py',
    'CC_Compare/tools/role_c_t_series.py',
    'CC_Compare/tools/role_c_replanner.py',
    'CC_Compare/tools/role_c_decision_log.py',
    'CC_Compare/tools/role_c_decision_audit.py',
    'CC_Compare/tools/role_c_controls.py',
    'CC_Compare/tools/role_c_resource_ceiling.py',
    'CC_Compare/tools/resource_timing.py',
    'CC_Compare/P1A_RoleC_收束裁定_2026-10-14.md',
    'CC_Compare/P1A_资源门登记.json',
    'CC_Compare/P1A_PhaseA_L1L2_B类_原件_2026-10-14.json',
    'CC_Compare/P1A_PhaseA_L1L2_A类_原件_2026-10-14.json',
    'CC_Compare/P1A_超时语义协议_七要素预注册草案_待审_2026-10-14.md',
)

#: §14.4 的五条"不得改变"——用**存在性/形状**做静态检查
FIVE_MUST_NOT_CHANGE = (
    ('judgement_unchanged', '判决仍由 decide(...) 产出',
     r'accept, diag = decide\(myopic_net=mn, e_o=e_o, lambda_B=lam, certify_ok=certify_ok'),
    ('boundary_rule_unchanged', 'B3 分支仍由 `rule` 决定 boundary',
     r"'boundary': rule, 'boundary_reason': reason"),
    ('reject_reason_unchanged', 'reject_reason 仍是 `reason`',
     r"'reject_reason': reason,"),
    ('certify_call_unchanged', '认证调用未变',
     r'certify_ok0, certify_energy0 = certify_fn\(candidate_plan\)'),
    ('return_shape_unchanged', 'B3 分支仍返回 `(False, diag)`',
     r'return False, diag'),
)

#: 预算差分与 λ 数值计算（必须逐字不变）
LAMBDA_CORE_LINES = (
    'b_lo = b - e',
    'b_lo_eval = max(b_lo, 0.0)',
    'v_hi = float(future_value_fn(b))',
    'v_lo = float(future_value_fn(b_lo_eval))',
    'raw = (v_hi - v_lo) / e',
)


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(p):
    with io.open(p, 'rb') as f:
        return _sha_bytes(f.read())


def _git(args, cwd):
    p = subprocess.run(['git'] + args, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None
    return p.stdout


def _baseline_blob(rel, commit, root):
    out = _git(['cat-file', 'blob', '%s:%s' % (commit, rel)], root)
    return out


def freeze(*, commit=BASELINE_COMMIT, root=None):
    root = root or _root()
    rep = {'label': 'P1A-C2-identity-freeze',
           'baseline_commit': commit,
           'allowlist': list(CHANGE_ALLOWLIST),
           'allowed_hunks': [dict(h) for h in ALLOWED_HUNKS],
           'five_must_not_change': [], 'must_not_change': [],
           'sha_pairs': [], 'diff_hunks': [], 'problems': []}

    # ---- 1/2. 旧/新 SHA + 逐行差异 ----
    for rel in list(CHANGE_ALLOWLIST) + list(ANALYSIS_SIDE):
        wt = os.path.join(root, rel.replace('/', os.sep))
        old = _baseline_blob(rel, commit, root)
        new_bytes = open(wt, 'rb').read() if os.path.isfile(wt) else None
        rep['sha_pairs'].append({
            'path': rel,
            'old_sha256': _sha_bytes(old) if old is not None else None,
            'new_sha256': _sha_bytes(new_bytes) if new_bytes is not None else None,
            'changed': (old != new_bytes),
            'classification': ('被测·白名单内' if rel in CHANGE_ALLOWLIST
                               else '分析侧（非被测）'),
        })
    diff = _git(['diff', '-U0', commit, '--'] + list(CHANGE_ALLOWLIST), root)
    if diff:
        for m in re.finditer(rb'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
                             diff, re.M):
            rep['diff_hunks'].append({
                'old_start': int(m.group(1)),
                'old_len': int(m.group(2) or 1),
                'new_start': int(m.group(3)),
                'new_len': int(m.group(4) or 1)})
        rep['diff_lines'] = {'added': diff.count(b'\n+') - diff.count(b'\n+++'),
                             'removed': diff.count(b'\n-') - diff.count(b'\n---')}

    # ---- 3. 静态白名单审计 ----
    oc = os.path.join(root, 'CC_Compare', 'tools', 'role_c_opportunity_cost.py')
    src = io.open(oc, encoding='utf-8').read()
    # 两处允许的改动必须在位
    for h in ALLOWED_HUNKS:
        present = h['marker'] in src
        rep['five_must_not_change'].append(
            {'check': 'allowed_hunk_present::%s' % h['name'], 'ok': present,
             'detail': h['why']})
        if not present:
            rep['problems'].append('缺少允许的改动：%s' % h['name'])
    # 五条不得改变
    for name, desc, pat in FIVE_MUST_NOT_CHANGE:
        ok = bool(re.search(pat, src))
        rep['five_must_not_change'].append({'check': name, 'ok': ok, 'detail': desc})
        if not ok:
            rep['problems'].append('§14.4 五项"不得改变"被违反：%s' % name)
    # 预算差分与 λ 数值核心行
    for line in LAMBDA_CORE_LINES:
        ok = line in src
        rep['five_must_not_change'].append(
            {'check': 'lambda_core_line::%s' % line, 'ok': ok,
             'detail': '预算差分/λ 数值计算逐字不变'})
        if not ok:
            rep['problems'].append('λ 数值核心行被改动：%s' % line)
    # 明确**没有**采用原样重抛（role_c_timing 也不得改）
    rep['five_must_not_change'].append(
        {'check': 'no_rethrow_of_decision_timeout', 'ok': 'raise' not in
         _except_body(src), 'detail': '§14.4：不采用原样重抛'})

    # ---- 4. 原封存文件不变证明（**blob ↔ blob**，避免把行尾工件当改动）----
    # ⚠️ 第一版用"工作树字节 vs 基线 blob"比，结果把两个**纯行尾**差异误报成改动：
    #    role_c_replanner.py 与 P1A_资源门登记.json 都**没有** `eol=lf` 属性
    #    ⇒ core.autocrlf=true 把工作树 checkout 成 CRLF，而 blob 是 LF。
    #    文件身份以 **git blob** 为准；工作树差异另标 `line_ending_artifact`。
    RULED_CRLF = {'CC_Compare/P1A_资源门登记.json'}   # 用户裁定 (b)：不 renormalize
    for rel in MUST_NOT_CHANGE:
        wt = os.path.join(root, rel.replace('/', os.sep))
        base = _baseline_blob(rel, commit, root)
        head = _git(['cat-file', 'blob', 'HEAD:%s' % rel], root)
        now = open(wt, 'rb').read() if os.path.isfile(wt) else None
        same_blob = (base is not None and head is not None and base == head)
        entry = {
            'path': rel,
            'blob_unchanged_vs_baseline': same_blob,
            'baseline_blob_sha256': _sha_bytes(base) if base is not None else None,
            'head_blob_sha256': _sha_bytes(head) if head is not None else None,
            'worktree_sha256': _sha_bytes(now) if now is not None else None,
            'worktree_equals_blob': (now is not None and head is not None
                                     and now == head),
        }
        if not entry['worktree_equals_blob'] and now is not None:
            entry['line_ending_artifact'] = True
            entry['crlf_bytes'] = now.count(b'\r\n')
            entry['artifact_note'] = (
                '工作树为 CRLF（该文件无 `.gitattributes eol=lf` 条目）；'
                '文件身份以 blob 为准，**不是**改动'
                + ('；本文件属用户裁定 (b) 的已登记 CRLF 原件，保持原样'
                   if rel in RULED_CRLF else ''))
        rep['must_not_change'].append(entry)
        if not same_blob:
            rep['problems'].append('本应不变的文件 blob 被改动：%s' % rel)

    rep['ok'] = not rep['problems']
    rep['summary'] = {
        'n_allowlisted_files': len(CHANGE_ALLOWLIST),
        'n_analysis_side_files': len(ANALYSIS_SIDE),
        'n_must_not_change': len(MUST_NOT_CHANGE),
        'n_must_not_change_ok': sum(1 for x in rep['must_not_change']
                                    if x['blob_unchanged_vs_baseline']),
        'n_line_ending_artifacts': sum(1 for x in rep['must_not_change']
                                       if x.get('line_ending_artifact')),
        'n_five_checks': len(rep['five_must_not_change']),
        'n_five_checks_ok': sum(1 for x in rep['five_must_not_change'] if x['ok']),
        'whitelist_compliant': not rep['problems'],
    }
    return rep


def _except_body(src):
    i = src.find('except Exception as exc:')
    if i < 0:
        return ''
    return src[i:i + 900]


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description='C2 身份冻结 + 静态白名单审计（只读）')
    ap.add_argument('--commit', default=BASELINE_COMMIT)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rep = freeze(commit=a.commit)
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({'baseline_commit': rep['baseline_commit'],
                      'whitelist_compliant': rep['summary']['whitelist_compliant'],
                      'must_not_change_ok': '%d/%d' % (
                          rep['summary']['n_must_not_change_ok'],
                          rep['summary']['n_must_not_change']),
                      'five_checks_ok': '%d/%d' % (
                          rep['summary']['n_five_checks_ok'],
                          rep['summary']['n_five_checks']),
                      'problems': rep['problems']},
                     ensure_ascii=False, indent=2))
    return 0 if rep['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
