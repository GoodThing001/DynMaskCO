#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`audit_qualification_matrix` 的反例测试（P0 资格矩阵的**门自身**缺陷）。

2026-10-14 复核在源码里指出两处**确定性**缺陷，本文件把它们的反例补齐并**锁定**：

  ① `day_id` 被 `int()` **静默强制转换**：
         `ids = [int(d.get('i')) for d in pd if d.get('i') is not None]`
     ⇒ `0.5 → 0`、`'5' → 5`、`True → 1`。于是 40 条 `i=0.5..39.5` 会被截断成 `0..39`
     并被判"日号完整"。原始日号必须是**真整数**（`bool` 也不算）。
  ② 有限性用 `isinstance(u, (int, float))` 判定：
         `fin = [u for u in utils if isinstance(u, (int, float))]`
     ⇒ `float('nan')` / `float('inf')` **都是** `float`，被算作"有限"，判据形同虚设。
     必须用 `math.isfinite`，并显式排除 `None` / 字符串 / `bool`。

测试里的 `naive_*` 断言**故意**复刻旧实现，证明"旧实现确实会放过这些输入" ——
否则这个测试只是在验证新代码，证明不了缺陷曾经存在。

运行：python CC_Compare/tools/test_audit_qualification_matrix.py   （退出码 0 = 全过）
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 注：本模块不自绑 sys.stdout（`audit_qualification_matrix` 会自绑），避免双重
# TextIOWrapper 关闭底层 buffer 的问题。
import audit_qualification_matrix as aqm                        # noqa: E402

N = 40


def good_pd(n=N):
    return [dict(i=k, utility=100.0 + k, hard_feasible=True, failures=[],
                 n_served=60, n_rejected=140) for k in range(n)]


# --------------------------------------------------------------------- Q_days
def _naive_days_ok(pd, expected=N):
    """**旧实现**的判据（含 int() 强转）—— 仅用于证明缺陷存在。"""
    ids = [int(d.get('i')) for d in pd if d.get('i') is not None]
    return len(ids) == expected and sorted(set(ids)) == list(range(expected))


def days_cases():
    cs = []

    def add(name, pd, want, naive=None):
        cs.append((name, pd, want, naive))

    add('① 正常 0..39', good_pd(), True)
    # —— 缺陷 ① 的精确反例 ——
    add('② day_id 整体平移 1..40', [dict(d, i=d['i'] + 1) for d in good_pd()], False)
    add('③ day_id 全为**非整数** 0.5..39.5（旧实现 int() 截断成 0..39 ⇒ 误通过）',
        [dict(d, i=d['i'] + 0.5) for d in good_pd()], False, naive=True)
    add('④ day_id 为**数字字符串** \'0\'..\'39\'（旧实现 int() 强转 ⇒ 误通过）',
        [dict(d, i=str(d['i'])) for d in good_pd()], False, naive=True)
    add('⑤ day_id 为 bool（True 是 int 子类，不得当整数）',
        [dict(d, i=bool(d['i'] % 2)) for d in good_pd()], False)
    add('⑥ day_id 缺失（39 条有键 + 1 条 None）',
        [dict(d, i=(None if d['i'] == 7 else d['i'])) for d in good_pd()], False)
    add('⑦ day_id 越界（39 缺失、40 出现）',
        [dict(d, i=(40 if d['i'] == 39 else d['i'])) for d in good_pd()], False)
    add('⑧ day_id 重复（40 条只有 39 个唯一）',
        [dict(d, i=(38 if d['i'] == 39 else d['i'])) for d in good_pd()], False)
    add('⑨ 只有 39 条记录', good_pd(39), False)
    add('⑩ 空记录', [], False)
    return cs


# ------------------------------------------------------------------- Q_finite
def _naive_finite_ok(pd):
    """**旧实现**的判据（只查 isinstance）—— 仅用于证明缺陷存在。"""
    utils = [d.get('utility') for d in pd]
    fin = [u for u in utils if isinstance(u, (int, float))]
    return bool(utils) and len(fin) == len(utils)


def finite_cases():
    cs = []

    def add(name, pd, want, naive=None):
        cs.append((name, pd, want, naive))

    add('① 全部有限', good_pd(), True)
    # —— 缺陷 ② 的精确反例 ——
    add('② 某日效用 NaN（旧实现 isinstance(nan,float)=True ⇒ 误通过）',
        [dict(d, utility=float('nan')) if d['i'] == 3 else d for d in good_pd()],
        False, naive=True)
    add('③ 某日效用 +inf（旧实现误通过）',
        [dict(d, utility=float('inf')) if d['i'] == 3 else d for d in good_pd()],
        False, naive=True)
    add('④ 某日效用 -inf（旧实现误通过）',
        [dict(d, utility=float('-inf')) if d['i'] == 3 else d for d in good_pd()],
        False, naive=True)
    add('⑤ 某日效用 None',
        [dict(d, utility=None) if d['i'] == 5 else d for d in good_pd()], False)
    add('⑥ 某日效用字符串 \'12.5\'',
        [dict(d, utility='12.5') if d['i'] == 5 else d for d in good_pd()], False)
    add('⑦ 某日效用 bool True（True 是 int 子类，不是效用值）',
        [dict(d, utility=True) if d['i'] == 5 else d for d in good_pd()], False)
    add('⑧ 空记录', [], False)
    return cs


# ------------------------------------------------- audit_block / adjudicate 端到端
def end_to_end():
    """缺陷必须**传导到裁决**：非整数日号 / NaN 效用 ⇒ `Q_formal_core=False`。"""
    out = []
    blk = {'per_day': [dict(d, n_served=60) for d in good_pd()],
           'arms': {'m': {'n_days': 40}},
           'solver_meta': {'n_solves': 100, 'n_fail': 0}}

    r0 = aqm.audit_block({}, blk, 'm')
    out.append(('健康块 ⇒ Q_days=True ∧ Q_finite=True ∧ Q_formal_core=True',
                r0['Q']['Q_days'] is True and r0['Q']['Q_finite'] is True
                and aqm.adjudicate(dict(r0), True, None, 'm')['Q']['Q_formal_core'] is True))

    blk_f = {'per_day': [dict(d, utility=float('nan'), n_served=60) for d in good_pd()],
             'arms': {'m': {}}, 'solver_meta': {'n_solves': 100, 'n_fail': 0}}
    r1 = aqm.audit_block({}, blk_f, 'm')
    out.append(('NaN 效用 ⇒ Q_finite=False ∧ Q_formal_core=False ∧ 有可读原因',
                r1['Q']['Q_finite'] is False
                and aqm.adjudicate(dict(r1), True, None, 'm')['Q']['Q_formal_core'] is False
                and 'nan' in (r1['independent']['Q_finite_detail'] or '').lower()))

    blk_d = {'per_day': [dict(d, i=d['i'] + 0.5, n_served=60) for d in good_pd()],
             'arms': {'m': {}}, 'solver_meta': {'n_solves': 100, 'n_fail': 0}}
    r2 = aqm.audit_block({}, blk_d, 'm')
    out.append(('非整数 day_id ⇒ Q_days=False ∧ Q_formal_core=False ∧ 原因点明"非整数"',
                r2['Q']['Q_days'] is False
                and aqm.adjudicate(dict(r2), True, None, 'm')['Q']['Q_formal_core'] is False
                and '非整数' in (r2['independent']['Q_days_detail'] or '')))

    blk_i = {'per_day': [dict(d, utility=float('inf'), n_served=60) for d in good_pd()],
             'arms': {'m': {}}, 'solver_meta': {'n_solves': 100, 'n_fail': 0}}
    r3 = aqm.audit_block({}, blk_i, 'm')
    out.append(('+inf 效用 ⇒ Q_finite=False',
                r3['Q']['Q_finite'] is False
                and aqm.adjudicate(dict(r3), True, None, 'm')['Q']['Q_formal_core'] is False))
    return out


def run():
    passed = failed = 0

    print('Q_days 独立复算（原始日号必须是真整数）：')
    for name, pd, want, naive in days_cases():
        ok, detail = aqm.recheck_days(pd)
        good = (ok == want)
        if naive is not None and naive:
            # 证明**旧实现**确实会放过这个输入（缺陷真实存在，不是假想）
            good = good and (_naive_days_ok(pd) is True)
        if good:
            passed += 1
            print('  PASS %-58s → %s' % (name, ok))
        else:
            failed += 1
            print('  FAIL %-58s 得 %s，期望 %s（detail=%s）' % (name, ok, want, detail))

    print('\nQ_finite 独立复算（math.isfinite，排除 NaN/±Inf/None/字符串/bool）：')
    for name, pd, want, naive in finite_cases():
        ok, detail = aqm.recheck_finite(pd)
        good = (ok == want)
        if naive is not None and naive:
            good = good and (_naive_finite_ok(pd) is True)
        if good:
            passed += 1
            print('  PASS %-58s → %s' % (name, ok))
        else:
            failed += 1
            print('  FAIL %-58s 得 %s，期望 %s（detail=%s）' % (name, ok, want, detail))

    print('\n端到端（缺陷必须传导到 Q_formal_core）：')
    for name, ok in end_to_end():
        if ok:
            passed += 1
            print('  PASS %s' % name)
        else:
            failed += 1
            print('  FAIL %s' % name)

    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run())
