#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**只读**复算对比方法资格矩阵（`CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_*.json`）。

为什么要有这个工具
------------------
资格矩阵的**统计口径**会被反复引用（"多少臂具备正式比较资格"）。手抄数字会漂移，
所以把复算做成**可重跑**的只读脚本：同一条命令给出同样的计数，引用时可直接复核。

⚠️ 只读：不改矩阵、不改任何 gate、不改封存结果；**不重算历史效用**、不增跑种子。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys

__all__ = ['MATRIX_GLOB', 'load_matrix', 'recompute', 'main']

MATRIX_GLOB = 'CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_*.json'

#: 逐臂 `Q` 子字典里的判据键（按用户引用的那一组 + 相关项）
Q_KEYS = ('Q_days', 'Q_finite', 'Q_hard', 'Q_no_crash', 'Q_nondegen',
          'Q_identity', 'Q_method_health', 'Q_method_participation',
          'Q_formal_core', 'Q_timeout_clean', 'Q_formal_strict')


def _latest(pattern=MATRIX_GLOB):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hits = sorted(glob.glob(os.path.join(root, pattern.replace('/', os.sep))))
    return hits[-1] if hits else None


def load_matrix(path=None):
    p = path or _latest()
    if not p:
        raise SystemExit('找不到资格矩阵文件（%s）' % MATRIX_GLOB)
    with io.open(p, encoding='utf-8') as f:
        return p, json.load(f)


def recompute(path=None):
    """返回可复算的计数报告（**不做任何解释性结论**）。"""
    p, d = load_matrix(path)
    arms = d.get('arms') or []

    def q(a, k):
        return (a.get('Q') or {}).get(k)

    def n_true(k):
        return sum(1 for a in arms if q(a, k) is True)

    def n_false(k):
        return sum(1 for a in arms if q(a, k) is False)

    def n_none(k):
        return sum(1 for a in arms if q(a, k) is None)

    core_false = [a for a in arms if q(a, 'Q_formal_core') is False]
    core_true = [a for a in arms if q(a, 'Q_formal_core') is True]
    strict_false_within_core = [a for a in core_true if q(a, 'Q_formal_strict') is False]

    return {
        'label': 'CC_Compare-qualification-matrix-recomputation',
        'matrix_path': os.path.relpath(p, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/'),
        'matrix_sha256': _sha(p),
        'criteria_revision': (d.get('criteria') or {}).get('revision'),
        'n_arms': len(arms),
        'Q_counts': {k: {'true': n_true(k), 'false': n_false(k), 'none': n_none(k)}
                     for k in Q_KEYS},
        'veto_counts': dict(collections.Counter(
            v for a in arms for v in (a.get('Q_vetoes') or []))),
        'not_applicable_counts': dict(collections.Counter(
            v for a in arms for v in (a.get('Q_reasons_not_applicable') or []))),
        'composition': {
            'by_role': dict(collections.Counter(a.get('role') for a in arms)),
            'by_group': dict(collections.Counter(a.get('group') for a in arms)),
            'by_p_c': {str(k): v for k, v in
                       collections.Counter(repr(a.get('p_c')) for a in arms).items()},
            'source_stable': {str(k): v for k, v in
                              collections.Counter(repr(a.get('source_stable'))
                                                  for a in arms).items()},
        },
        'core_false_breakdown': {
            'n_core_false': len(core_false),
            'by_veto_combination': {
                str(list(k)): v for k, v in collections.Counter(
                    tuple(sorted(a.get('Q_vetoes') or []))
                    for a in core_false).items()},
        },
        'core_true_but_strict_false': {
            'n': len(strict_false_within_core),
            'by_veto_combination': {
                str(list(k)): v for k, v in collections.Counter(
                    tuple(sorted(a.get('Q_vetoes') or []))
                    for a in strict_false_within_core).items()},
        },
        'source_stable_false_arms': [
            {'role': a.get('role'), 'method': a.get('method'),
             'p_c': a.get('p_c'), 'Q_formal_core': q(a, 'Q_formal_core'),
             'Q_formal_strict': q(a, 'Q_formal_strict'),
             'vetoes': a.get('Q_vetoes') or []}
            for a in arms if a.get('source_stable') is False],
        'timeout_clean_false_by_role': {
            str(k): v for k, v in collections.Counter(
                a.get('role') for a in arms
                if q(a, 'Q_timeout_clean') is False).items()},
        'caveats': [
            ('`n_arms` 是**方法臂记录数**，不是独立算法数，也不是正式合格对照数：'
             '其中含主模型交付记录、不同档位与重复适配配置'),
            ('`Q_*` 取值除 true/false 外还有 **none**（不适用）—— 统计时必须单独列出，'
             '不得把 none 归并进 true 或 false'),
            ('本报告只复算**计数**，**不重算**任何历史效用，也不对资格做新的实质判断'),
        ],
    }


def _sha(p):
    import hashlib
    with io.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='只读复算对比方法资格矩阵')
    ap.add_argument('--matrix', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rep = recompute(a.matrix)
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({'n_arms': rep['n_arms'],
                      'Q_formal_core_true': rep['Q_counts']['Q_formal_core']['true'],
                      'Q_formal_strict_true': rep['Q_counts']['Q_formal_strict']['true'],
                      'veto_Q_identity': rep['veto_counts'].get('Q_identity'),
                      'veto_Q_method_health': rep['veto_counts'].get('Q_method_health'),
                      'veto_Q_nondegen': rep['veto_counts'].get('Q_nondegen'),
                      'Q_timeout_clean_false':
                          rep['Q_counts']['Q_timeout_clean']['false']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
