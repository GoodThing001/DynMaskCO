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


# ---------------------------------------------------------------- gate 种子解析
# 2026-10-14：part_d ④ 之前把 0930/20261001 两个 gate 的 18 个块判成"CI 未复现"，
# 根因是种子取错（回落 `generated_from.seed`=20260926）。以下锁定解析规则本身：
# **必须**按确定性优先级取，**不得**事后试种子凑复现。

def _gate_json(tmp, name, payload):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'gate.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return p


def run_gate_seed():
    import tempfile
    tmp = tempfile.mkdtemp(prefix='cpiseed_')
    cases = []

    def check(name, got, want):
        cases.append((name, got == want, got, want))

    # ① 规范字段：run_a1_step3_compare / run_a1_external_accept 只写 ran
    p = _gate_json(tmp, 'a1_a3_x_20260930_y',
                   {'seeds': {'train': 20260925, 'ran': [20260930]}})
    check('① ran 单一 int ⇒ 用它', cpi._gate_seed({'seeds': {'ran': [20260930]}}, p),
          (20260930, 'seeds.ran'))
    # ② step2 schema：顶层 seeds.gate
    p2 = _gate_json(tmp, 'a1_step2_x_20260926_y', {'seeds': {'gate': 20260926}})
    check('② 无 ran、有 seeds.gate ⇒ 用它',
          cpi._gate_seed({'seeds': {'gate': 20260926}}, p2), (20260926, 'seeds.gate'))
    # ③ ran 非单一 int ⇒ fail-closed（不猜）
    g3, s3 = cpi._gate_seed({'seeds': {'ran': [20260926, 20260930]}}, p)
    check('③ ran=[两个] ⇒ None（歧义，不猜）', (g3, s3.startswith('ambiguous')), (None, True))
    # ④ ran 与 gate 冲突 ⇒ fail-closed（不择优）
    g4, s4 = cpi._gate_seed({'seeds': {'ran': [20260930], 'gate': 20260926}}, p)
    check('④ ran≠gate ⇒ None（冲突，不择优）', (g4, s4.startswith('conflict')), (None, True))
    # ⑤ 无任何种子字段 ⇒ 用目录日期兜底
    p5 = _gate_json(tmp, 'a1_a3_online_density_B_20261001_40_1003_040454', {'seeds': {}})
    check('⑤ 无 ran/gate ⇒ 目录日期兜底',
          cpi._gate_seed({'seeds': {}}, p5), (20261001, 'dir_date'))
    # ⑥ 连日期都没有 ⇒ 最后兜底 seeds.dev
    p6 = _gate_json(tmp, 'nodate', {'seeds': {'dev': 20260926}})
    check('⑥ 目录无日期 ⇒ seeds.dev 兜底',
          cpi._gate_seed({'seeds': {'dev': 20260926}}, p6), (20260926, 'seeds.dev'))
    # ⑦ 什么都没有 ⇒ None（part_d 会记 SKIP，**不**回落默认种子）
    p7 = _gate_json(tmp, 'nodate2', {'seeds': {}})
    check('⑦ 全无 ⇒ None（SKIP，不回落默认）',
          cpi._gate_seed({'seeds': {}}, p7), (None, 'absent'))

    # ⑧ 嵌套 schema：results[<seed>][pc]['per_day'] ⇒ 外层键即该块种子（证据最强）
    p8 = _gate_json(tmp, 'nested_20260926', {'seeds': {'ran': [20260926]}, 'results': {
        '20260930': {'p_c=(5,10,15)': {'per_day': {'maskco': [], 'cond_hist': []}}}}})
    pd8, s8, src8 = cpi._per_day_of(p8, 'p_c=(5,10,15)')
    check('⑧ 嵌套 results[seed] ⇒ 用外层键',
          (s8, src8, pd8), (20260930, 'results_key', {'maskco': [], 'cond_hist': []}))

    # ⑨ **必须按 pc 选块**：p_c=0 在前，请求的是主档 —— 取错档会一次跑出大量假失败
    p9 = _gate_json(tmp, 'flat_20260926', {'seeds': {'ran': [20260926]}, 'results': {
        '20260926': {
            'p_c=0': {'per_day': {'maskco': [1], 'cond_hist': [1]}},
            'p_c=(5,10,15)': {'per_day': {'maskco': [2], 'cond_hist': [2]}}}}})
    pd9, s9, _src9 = cpi._per_day_of(p9, 'p_c=(5,10,15)')
    check('⑨ 按 pc 选块（不得取第一个含 per_day 的块）',
          (s9, pd9['maskco']), (20260926, [2]))

    # ⑩ gate 文件缺失 ⇒ (None,None,'gate_missing') ⇒ part_d 记 SKIP
    check('⑩ gate 缺失 ⇒ gate_missing',
          cpi._per_day_of(os.path.join(tmp, 'nope', 'gate.json'), 'p_c=0'),
          (None, None, 'gate_missing'))

    passed = sum(1 for _n, ok, _g, _w in cases if ok)
    for n, ok, got, want in cases:
        if ok:
            print('  PASS [种子] %-40s → %r' % (n, got))
        else:
            print('  FAIL [种子] %-40s 得 %r，期望 %r' % (n, got, want))
    failed = len(cases) - passed
    print('\n种子解析：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


def main():
    return run() | run_gate_seed()


if __name__ == '__main__':
    sys.exit(main())
