#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 决策记录的**写入器**：按 `role_c_decision_audit` 的 schema 落盘，**写入即校验**。

为什么要有这个模块（而不是让 runner 自己拼 dict）：

  * 审计器与写入器是同一份契约的**两个独立实现** —— 写入器若正确，审计必然通过；
    于是"审计失败"就只剩两种可能：记录被篡改，或写入器有 bug，**不会有第三种含糊情形**；
  * **写入即拒绝**（fail-closed）：一条会违反 schema 或违反承诺单调性的记录
    **根本进不了日志**，而不是先写进去、事后靠审计发现；
  * `before/after` 集合由**状态机**推导（不由调用方手填）⇒ 从结构上排除
    "手填 after 导致集合回退"这类错误。

契约（与 `role_c_decision_audit` 一致，且**额外强制**一条该模块尚未查的不变式）：
    accepted_before ⊆ accepted_after        （承诺不可撤销）
    executed_before ⊆ executed_after        （已执行前缀冻结）
    executed_after  ⊆ accepted_after        （**没接过的单不可能被执行**）
    action == 'accept' ⇒ 必须有 order_id 且 certify_energy ≤ budget_B

只做记录，**不读效用、不碰 env、不跑实验**。
"""
from __future__ import annotations

import role_c_decision_audit as aud


class StateError(ValueError):
    """写入被拒（记录会违反 schema 或承诺/预算不变式）。"""


class CommittedState:
    """已接受承诺 + 已执行前缀（**只增不减**）。"""

    __slots__ = ('accepted', 'executed')

    def __init__(self, accepted=(), executed=()):
        self.accepted = frozenset(accepted)
        self.executed = frozenset(executed)

    def copy(self):
        return CommittedState(self.accepted, self.executed)

    def __repr__(self):
        return 'CommittedState(accepted=%s, executed=%s)' % (
            sorted(self.accepted), sorted(self.executed))


class DecisionLogger:
    """把决策事件写成 `role_c_decision_audit` 可审计的记录序列。

    `budget_B` 必须给出（用于校验 `certify_energy ≤ B`；也是 `P1A_CONTROL_FROZEN.json`
    的共享身份锚点之一）。
    """

    def __init__(self, budget_B, state=None, tol=1e-9):
        if budget_B is None:
            raise StateError('必须提供 budget_B（否则无法校验 certify_energy ≤ B）')
        self.budget_B = float(budget_B)
        self.tol = float(tol)
        self.state = state or CommittedState()
        self.records = []

    # ---- 内部校验：任何一条不过就抛错，**不写入** ----
    def _validate(self, rec, acc_after, exe_after):
        if not (self.state.accepted <= acc_after):
            raise StateError('承诺被撤销：%s 不在 after 里'
                             % sorted(self.state.accepted - acc_after))
        if not (self.state.executed <= exe_after):
            raise StateError('已执行前缀回退：%s 不在 after 里'
                             % sorted(self.state.executed - exe_after))
        if not (exe_after <= acc_after):
            raise StateError('执行了**未接受**的订单：%s' % sorted(exe_after - acc_after))
        if rec['action'] != 'accept' and acc_after != self.state.accepted:
            raise StateError('非接单事件（%s）不得改变承诺集合' % rec['action'])
        miss = [f for f in aud.REQUIRED_FIELDS if f not in rec]
        if miss:
            raise StateError('缺字段 %s' % miss)

    def _precheck(self, action, certified, order_id, certify_energy):
        """**改状态之前**的前置校验。

        ⚠️ 顺序很重要（2026-10-14 实测踩到）：初版先 `acc_after.add(order_id)` 再校验，
        于是 `order_id=None` 会先进集合，随后 `sorted()` 抛 **TypeError**
        （`'<' not supported between int and NoneType`）—— 一个**误导性的内部错误**
        取代了本该给出的明确拒绝。⇒ 一切与"值是否合法"有关的判断必须在改状态**之前**。
        """
        if action not in aud.ACTIONS:
            raise StateError('action=%r 非法（应为 %s）' % (action, aud.ACTIONS))
        if not isinstance(certified, bool):
            raise StateError('certified 必须是 bool，实得 %r' % (certified,))
        if action == 'accept':
            if order_id is None:
                raise StateError('accept 必须带 order_id（None 不得进入承诺集合）')
            if certify_energy is None:
                raise StateError('accept 必须带 certify_energy（否则无法证明 ≤ B）')
            if float(certify_energy) > self.budget_B + self.tol:
                raise StateError('certify_energy=%r > budget_B=%r'
                                 % (certify_energy, self.budget_B))
            if certified is not True:
                raise StateError('accept 时 certified 必须为 True（不得"先提交后认证"）')
            if order_id in self.state.accepted:
                raise StateError('重复接单：order_id=%r 已在已接受集合里' % (order_id,))

    def log_event(self, *, inst_idx, day_id, clock, info_max_reveal, action,
                  certified, decision_hash, order_id=None, certify_energy=None,
                  executed_after=None):
        """记录一次决策。返回写入的 record（已更新内部状态）。

        `executed_after` = 本事件后**已执行**的订单集合（只增不减）。不传则保持不变。
        **前置校验 → 计算 after 集合 → 全量校验 → 才改状态并落盘**；
        任何一步抛错都**不留痕迹**（records 与 state 都不变）。
        """
        self._precheck(action, certified, order_id, certify_energy)
        acc_after = set(self.state.accepted)
        if action == 'accept':
            acc_after.add(order_id)
        exe_after = (set(self.state.executed) if executed_after is None
                     else set(executed_after))
        rec = dict(inst_idx=inst_idx, day_id=day_id, clock=float(clock),
                   info_max_reveal=float(info_max_reveal), order_id=order_id,
                   action=action, certified=bool(certified),
                   certify_energy=(None if certify_energy is None else float(certify_energy)),
                   budget_B=self.budget_B,
                   accepted_before=sorted(self.state.accepted),
                   accepted_after=sorted(acc_after),
                   executed_before=sorted(self.state.executed),
                   executed_after=sorted(exe_after),
                   decision_hash=decision_hash)
        self._validate(rec, frozenset(acc_after), frozenset(exe_after))
        # 通过 ⇒ 提交状态并落盘（顺序：先校验，后改状态，任何异常都不留痕迹）
        self.state = CommittedState(acc_after, exe_after)
        self.records.append(rec)
        return rec

    def audit(self, **kw):
        """对自己产出的日志跑审计（写入器正确 ⇒ 必然通过）。"""
        return aud.audit(self.records, **kw)
