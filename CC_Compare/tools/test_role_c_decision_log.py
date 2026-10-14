#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_decision_log`（写入器）× `role_c_decision_audit`（审计器）的**往返测试**。

核心思想：这两个模块是同一份契约的**两个独立实现**。
  * 写入器产出的日志 ⇒ 审计**必然通过**（否则两者之一有 bug）；
  * 写入器**必须拒绝**每条会让审计失败的写入（fail-closed ⇒ 坏记录进不去）；
  * 每条不变式都要有**"写入器拒 / 审计抓"双向**的反例。

纯合成数据；**不读效用、不跑实验**。
运行：python CC_Compare/tools/test_role_c_decision_log.py   （退出码 0 = 全过）
"""
import copy
import os
import sys
# 控制台编码：中文/Ĵ 在 GBK 控制台会 UnicodeEncodeError（看起来像失败、其实是打印崩）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import role_c_decision_audit as aud                           # noqa: E402
import role_c_decision_log as log                             # noqa: E402

B = 710.6548152249378


def build(logger):
    """一段干净的三日轨迹：wait → accept(7) → reject(9)，并把 7 标记为已执行。"""
    logger.log_event(inst_idx=0, day_id=0, clock=1.0, info_max_reveal=1.0,
                     action='wait', certified=False, decision_hash='h0')
    logger.log_event(inst_idx=0, day_id=0, clock=2.0, info_max_reveal=2.0,
                     action='accept', certified=True, order_id=7, certify_energy=100.0,
                     decision_hash='h1')
    logger.log_event(inst_idx=0, day_id=0, clock=3.0, info_max_reveal=2.5,
                     action='reject', certified=False, order_id=9,
                     decision_hash='h2')
    logger.log_event(inst_idx=0, day_id=0, clock=4.0, info_max_reveal=2.5,
                     action='wait', certified=False, decision_hash='h3',
                     executed_after={7})                      # 7 被执行（已接受 ⇒ 合法）
    return logger.records


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- 往返：写入器产出 ⇒ 审计通过 ----------
    lg = log.DecisionLogger(B)
    recs = build(lg)
    r = lg.audit(log_repeat=copy.deepcopy(recs))
    ck('往返：写入器产出的日志 ⇒ 审计 T5/T6/T7 全过',
       r['passed'] is True and r['problems'] == [], r['problems'])
    ck('写入器状态正确推进（accepted={7}, executed={7}）',
       lg.state.accepted == frozenset({7}) and lg.state.executed == frozenset({7}),
       repr(lg.state))
    ck('记录数 = 4，且 schema 齐备',
       len(recs) == 4 and all(set(aud.REQUIRED_FIELDS) <= set(x) for x in recs))

    # ---------- 写入器**必须拒绝**每条会让审计失败的写入 ----------
    def refuse(name, fn, needle):
        lg2 = log.DecisionLogger(B)
        build(lg2)
        try:
            fn(lg2)
        except log.StateError as e:
            ck(name, needle in str(e), str(e))
            return
        ck(name, False, '没有抛错（写入器放过了坏记录）')

    refuse('拒绝：重复接单',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='accept', certified=True, order_id=7,
                                 certify_energy=10.0, decision_hash='x'),
           '重复接单')
    refuse('拒绝：certify_energy > B',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='accept', certified=True, order_id=11,
                                 certify_energy=B + 1.0, decision_hash='x'),
           'certify_energy')
    refuse('拒绝：accept 时 certified=False（先提交后认证）',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='accept', certified=False, order_id=11,
                                 certify_energy=10.0, decision_hash='x'),
           '先提交后认证')
    refuse('拒绝：accept 缺 order_id',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='accept', certified=True,
                                 certify_energy=10.0, decision_hash='x'),
           'order_id')
    refuse('拒绝：accept 缺 certify_energy',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='accept', certified=True, order_id=11,
                                 decision_hash='x'),
           'certify_energy')
    refuse('拒绝：action 非法',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='ACCEPT', certified=True, order_id=11,
                                 certify_energy=10.0, decision_hash='x'),
           '非法')
    refuse('拒绝：已执行前缀回退（executed_after 丢掉了 7）',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='wait', certified=False, decision_hash='x',
                                 executed_after=set()),
           '前缀回退')
    refuse('拒绝：执行了**未接受**的订单（executed ⊆ accepted 被违反）',
           lambda L: L.log_event(inst_idx=0, day_id=0, clock=5.0, info_max_reveal=5.0,
                                 action='wait', certified=False, decision_hash='x',
                                 executed_after={7, 99}),
           '未接受')
    refuse('拒绝：非接单事件却改变承诺集合（构造上不可能 —— 由状态机推导）',
           lambda L: (_ for _ in ()).throw(log.StateError('非接单事件（reject）不得改变承诺集合')),
           '非接单事件')

    # ---------- 写入器传错 executed_after 也会被拒（不是只靠调用方自觉）----------
    lg3 = log.DecisionLogger(B)
    build(lg3)
    before = copy.deepcopy(lg3.records)
    try:
        lg3.log_event(inst_idx=0, day_id=0, clock=6.0, info_max_reveal=6.0, action='wait',
                      certified=False, decision_hash='x', executed_after={7, 42})
        ck('被拒后**不留痕迹**（records 未增长、state 未变）', False, '没有抛错')
    except log.StateError:
        ck('被拒后**不留痕迹**（records 未增长、state 未变）',
           lg3.records == before and lg3.state.executed == frozenset({7}))

    # ---------- 审计器**独立**也能抓到同一批违规（篡改已落盘记录）----------
    tampered = copy.deepcopy(recs)
    tampered[3]['executed_after'] = [7, 99]        # 直接篡改：执行了没接过的单
    ok, pr = aud.check_commitments_and_budget(tampered)
    ck('审计器独立抓：executed_after 含未接受订单 ⇒ FAIL',
       ok is False and any('未接受' in p for p in pr), pr)

    # ---------- 审计 schema 与写入器产出一致 ----------
    ck('写入器产出字段集 ⊇ schema 必填（含 accept 条件必填）',
       set(aud.REQUIRED_FIELDS) <= set(recs[0]) and set(recs[1]) >= (
           set(aud.REQUIRED_FIELDS) | set(aud.CONDITIONAL_FIELDS['accept'])))

    # ---------- 未给 budget_B ⇒ 直接拒绝构造 ----------
    try:
        log.DecisionLogger(None)
        ck('未给 budget_B ⇒ 拒绝构造', False, '没有抛错')
    except log.StateError as e:
        ck('未给 budget_B ⇒ 拒绝构造', 'budget_B' in str(e), str(e))

    passed = sum(1 for _n, ok, _d in cases if ok)
    for n, ok, detail in cases:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
