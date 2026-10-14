#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_replanner` 骨架的回归（用**桩 env/车辆**，不碰真实环境、不跑实验）。

覆盖：
  ① `booking_horizon=None` ⇒ **拒绝构造**（因果性 fail-closed —— 源码里 None 会读真实未来）
  ② 揭示事件：只处理"未服务 ∧ 未被非 replan 车预留 ∧ 未决策过"的新订单
  ③ 处理顺序 = 揭示时刻升序（确定性）
  ④ 事务：`decide_fn` 抛错 ⇒ 该单**不**被标记、计划**回滚**（不留半完成状态）
  ⑤ 采纳规则：只在"接单 ∧ 通过认证"时采纳候选计划；未过认证 ⇒ 回滚 + 记拒单
  ⑥ `max_total_accepts` 上限 ⇒ 转拒单
  ⑦ `plan()` 回放：route 非空 ⇒ mutable_suffix=route（**不含尾部 0**）；
     空 + 在途 + 有未来 ⇒ `[]`（WAIT）；空 + 无未来 ⇒ `[0]`（返回）
  ⑧ `export_state` / `restore_state` 往返（含车辆键 str↔int）
  ⑨ 落盘：给了 logger ⇒ 每单一条记录，且 **T5/T6 审计通过**
  ⑩ 落盘被拒（StateError）**不被吞掉**（必须上抛，否则状态与日志会不一致）

运行：python CC_Compare/tools/test_role_c_replanner.py   （退出码 0 = 全过）
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import role_c_replanner as rp                                 # noqa: E402
from role_c_decision_log import DecisionLogger, StateError     # noqa: E402


class _V:
    def __init__(self, vid, status='idle', current_node=0):
        self.vehicle_id = vid
        self.status = status
        self.current_node = current_node
        self.mutable_suffix = None


class _Env:
    def __init__(self, reveal_time=None, reserved=()):
        self.reveal_time = reveal_time or {}
        self._reserved = set(reserved)

    def get_reserved_customers(self, vehicles):
        return set(self._reserved)


B = 100.0


def _mk(decide_fn=None, certify_fn=None, logger=None, **kw):
    """默认桩：接单并把订单插到 1 号车尾；认证恒通过（能耗 1.0）。"""
    def d(order, ctx):
        if decide_fn is not None:
            return decide_fn(order, ctx)
        p = {k: list(v) for k, v in ctx['plan'].items()}
        p.setdefault(1, []).append(order)
        return True, {'plan': p, 'day_id': ctx['inst_idx'], 'info_max_reveal': ctx['clock']}

    def c(plan):
        if certify_fn is not None:
            return certify_fn(plan)
        return True, 1.0

    return rp.RoleCReplanner(budget=B, capacity=50.0, booking_horizon=16.0, contract=None,
                             cooling_share=3.72, decide_fn=d, certify_fn=c,
                             logger=logger, **kw)


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- ① 因果性 fail-closed ----------
    try:
        rp.RoleCReplanner(budget=B, capacity=50.0, booking_horizon=None, contract=None,
                          cooling_share=3.72, decide_fn=lambda o, c: (False, {}),
                          certify_fn=lambda p: (True, 1.0))
        ck('① booking_horizon=None ⇒ 拒绝构造（因果性）', False, '没有抛错')
    except ValueError as e:
        ck('① booking_horizon=None ⇒ 拒绝构造（因果性）', 'has_future_reveal' in str(e), str(e))

    # ---------- ② / ③ 揭示事件：过滤 + 顺序 ----------
    seen = []

    def rec(order, ctx):
        seen.append(order)
        return False, {'day_id': 0, 'info_max_reveal': ctx['clock']}

    env = _Env(reveal_time={(0, 5): 3.0, (0, 7): 1.0, (0, 9): 2.0}, reserved=(9,))
    r = _mk(decide_fn=rec)
    r.on_reveal(env, 0, 5.0, [_V(1)], {5: False, 7: False, 9: False, 11: True},
                [5, 7, 9, 11])
    ck('② 排除：已服务(11) / 被预留(9)', 9 not in seen and 11 not in seen, seen)
    ck('③ 处理顺序 = 揭示时刻升序（7→5）', seen == [7, 5], seen)

    # ---------- ④ 事务：decide_fn 抛错 ⇒ 不标记、计划回滚 ----------
    def boom(order, ctx):
        raise RuntimeError('decide 内部异常')

    r = _mk(decide_fn=boom)
    before = copy.deepcopy(r._plan)
    try:
        r.on_reveal(_Env(reveal_time={(0, 3): 1.0}), 0, 1.0, [_V(1)], {3: False}, [3])
        ck('④ decide 抛错 ⇒ 异常上抛', False, '没有抛错')
    except RuntimeError:
        ck('④ 事务：decide_fn 抛错 ⇒ 该单不被标记且计划回滚',
           r._accepted == set() and r._rejected == set() and r._plan == before,
           'acc=%s rej=%s plan=%s' % (r._accepted, r._rejected, r._plan))

    # ---------- ⑤ 采纳规则 ----------
    r = _mk()                                        # 认证恒过 ⇒ 接单并采纳计划
    r.on_reveal(_Env(reveal_time={(0, 3): 1.0}), 0, 1.0, [_V(1)], {3: False}, [3])
    ck('⑤ 接单 ∧ 认证过 ⇒ 采纳候选计划且记入 accepted',
       r._accepted == {3} and r._plan.get(1) == [3], 'plan=%s' % r._plan)

    r = _mk(certify_fn=lambda p: (False, 999.0))     # 认证不过
    r.on_reveal(_Env(reveal_time={(0, 3): 1.0}), 0, 1.0, [_V(1)], {3: False}, [3])
    ck('⑤ 认证不过 ⇒ 回滚计划 + 记拒单（不采纳候选）',
       r._accepted == set() and r._rejected == {3} and r._plan == {},
       'acc=%s rej=%s plan=%s' % (r._accepted, r._rejected, r._plan))

    # ---------- ⑥ 上限 ----------
    r = _mk(max_total_accepts=1)
    env1 = _Env(reveal_time={(0, 3): 1.0, (0, 4): 1.0})
    r.on_reveal(env1, 0, 1.0, [_V(1)], {3: False, 4: False}, [3, 4])
    ck('⑥ max_total_accepts=1 ⇒ 第二单转拒单',
       len(r._accepted) == 1 and len(r._rejected) == 1, (r._accepted, r._rejected))

    # ---------- ⑦ plan() 回放 ----------
    r = _mk()
    r.on_reveal(_Env(reveal_time={(0, 3): 1.0}), 0, 1.0, [_V(1)], {3: False}, [3])
    v = _V(1, status='ready', current_node=0)
    r.plan(None, 0, 2.0, [v], {3: False}, [3])
    ck('⑦ route 非空 ⇒ mutable_suffix = route（**不含尾部 0**）',
       v.mutable_suffix == [3], v.mutable_suffix)

    v2 = _V(2, status='ready', current_node=5)       # 在途
    r.plan(None, 0, 2.0, [v2], {3: False}, [3])       # clock < booking_horizon ⇒ 有未来
    ck('⑦ 空 + 在途 + 有未来 ⇒ []（WAIT，不强制返回）', v2.mutable_suffix == [], v2.mutable_suffix)

    v3 = _V(3, status='ready', current_node=5)
    r.plan(None, 0, 99.0, [v3], {3: False}, [3])      # clock > booking_horizon ⇒ 无未来
    ck('⑦ 空 + 无未来 ⇒ [0]（返回仓）', v3.mutable_suffix == [0], v3.mutable_suffix)

    # ---------- ⑧ 快照往返 ----------
    st = r.export_state()
    r2 = _mk()
    r2.restore_state(st)
    ck('⑧ export/restore 往返一致（plan 键 str↔int 还原）',
       r2._accepted == r._accepted and r2._plan == r._plan
       and r2._plan.get(1) == [3], 'plan=%s' % r2._plan)
    try:
        r2.restore_state({'version': 'unknown'})
        ck('⑧ 未知快照版本 ⇒ 拒绝', False, '没有抛错')
    except ValueError:
        ck('⑧ 未知快照版本 ⇒ 拒绝', True)

    # ---------- ⑨ 落盘 + 审计 ----------
    lg = DecisionLogger(B)
    r = _mk(logger=lg)
    env2 = _Env(reveal_time={(0, 3): 1.0, (0, 4): 1.0})
    r.on_reveal(env2, 0, 1.0, [_V(1)], {3: False, 4: False}, [3, 4])
    ck('⑨ 每单一条记录', len(lg.records) == 2, len(lg.records))
    audit = lg.audit(log_repeat=[dict(x) for x in lg.records])
    ck('⑨ 落盘日志通过 T5/T6/T7 审计', audit['passed'] is True, audit['problems'])

    # ---------- ⑩ 落盘被拒不吞掉 ----------
    class _BadLogger(DecisionLogger):
        def log_event(self, **kw):
            raise StateError('故意拒绝写入')

    r = _mk(logger=_BadLogger(B))
    try:
        r.on_reveal(_Env(reveal_time={(0, 3): 1.0}), 0, 1.0, [_V(1)], {3: False}, [3])
        ck('⑩ logger 写入被拒 ⇒ 上抛（不吞）', False, '没有抛错')
    except StateError:
        ck('⑩ logger 写入被拒 ⇒ 上抛（不吞）', True)

    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
