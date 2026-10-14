#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`check_pairing_identity` 的反例测试：**必须拒绝**跨批次配对。

用户的原始要求：「对于当前 I3，检查器应明确拒绝把 631.4 与 −144.680 配在同一行」。
本测试用**合成的被篡改数据集**验证这一点 —— 真实数据集是干净的，用真实数据测不出"能不能挡住"。

运行：python CC_Compare/tools/test_pairing_identity.py   （退出码 0 = 全过）
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
# ⚠️ 不要在这里重绑 `sys.stdout`：`check_pairing_identity` 被 import 时**自己会重绑**。
# 两个 TextIOWrapper 套同一个 buffer，先建的那个被 GC 时会**关掉底层 buffer**
# ⇒ 后续 print 报 `ValueError: I/O operation on closed file`（2026-10-14 实测踩到）。
import check_pairing_identity as cpi                        # noqa: E402

DATA = os.path.join(ROOT, 'CC_Compare', 'results', 'a1_compare_table', 'dataset.json')
I3_GATE = 'results/i3_online_recovery_1009_v1/online_20260926/gate.json'
I3_TIER = 'p_c=(5,10,15)'


def load():
    with open(DATA, encoding='utf-8') as f:
        return json.load(f)


def find(d, gate, arm, pc):
    for r in d['main_model']:
        if r.get('gate') == gate and r.get('arm') == arm and r.get('p_c') == pc:
            return r
    return None


def tamper_cross_batch(d):
    """把 I3 的配对差改成"用 631.410 当对照"算出来的值（= 跨协议混用）。"""
    rec = find(d, I3_GATE, 'maskco', I3_TIER)
    cond = find(d, I3_GATE, 'cond_hist', I3_TIER)
    a = rec['reported_utility']['mean']
    foreign = 631.410                    # 另一套协议（外部臂基线 a1_c1fresh_reveal_40）的值
    rec['main_comparison']['maskco_minus_cond_hist']['mean'] = a - foreign
    assert cond is not None
    return a - foreign


CASES = []


def case(name, expect_fail_part, mutate):
    CASES.append((name, expect_fail_part, mutate))


def _noop(d):
    return None


case('① 未篡改（真实数据）', None, _noop)
case('② I3 配对差改成用 631.410（跨协议）算出来的值', 'A', tamper_cross_batch)


def _tamper_b(d):
    """把同 gate 的 cond_hist 改成另一协议的值 ⇒ 隐含对照与它不符。"""
    cond = find(d, I3_GATE, 'cond_hist', I3_TIER)
    cond['reported_utility']['mean'] = 631.410
    return None


case('③ 把 I3 同批 cond_hist 改成 631.410（B 段应交待不符）', 'B', _tamper_b)


def _tamper_days(d):
    rec = find(d, I3_GATE, 'maskco', I3_TIER)
    rec['reported']['n_days'] = 39
    return None


case('④ I3 本臂只有 39 天（天数不一致）', 'A', _tamper_days)


def run():
    base = load()
    passed = failed = 0
    for name, expect, mutate in CASES:
        d = copy.deepcopy(base)
        mutate(d)
        _, fa = cpi.part_a(d)
        _, _, fb = cpi.part_b(d)
        fired = ({'A'} if fa else set()) | ({'B'} if fb else set())
        # 期望的判据**必须**被触发；同时允许其他判据一并触发（例如篡改对照值会同时破坏 A 与 B）
        ok = (expect in fired) if expect else (not fired)
        if ok:
            passed += 1
            print('  PASS %-46s 触发=%s（期望含 %s）'
                  % (name, ','.join(sorted(fired)) or '无', expect or '无'))
        else:
            failed += 1
            print('  FAIL %-46s 触发=%s（期望含 %s） A=%d B=%d'
                  % (name, ','.join(sorted(fired)) or '无', expect or '无', len(fa), len(fb)))
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run())
