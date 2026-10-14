#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**在线 replanner 骨架**（env 面向，**判决函数注入**）。

设计要点（为什么这样切）：

  * **判决逻辑不在这里**。本模块只做"状态机 + 计划读写 + 顺序 + 落盘"，
    具体接/拒由注入的 `decide_fn` 决定 ⇒ **不受尚未裁定的机制口径影响**
    （§2.6 的 B-shadow / B-density / 重新界定，换 `decide_fn` 即可）。
  * **接口对着真实源码接地**（`strict_online_env.Replanner`）：
      plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None)
      on_reveal(env, inst_idx, clock, vehicles, served_mask, visible_ids)
      export_state() / restore_state(state)
    计划表示沿用 `C0ReserveReplanner`：`self._plan = {vehicle_id: [order...]}`；
    `v.mutable_suffix = route`（**不含尾部 0**）；空则 `[]`（在途且有未来 ⇒ WAIT）
    或 `[0]`（无未来 ⇒ 返回仓）。
  * **因果性 fail-closed**：`booking_horizon=None` 时 `StrictOnlineEnv.has_future_reveal`
    会**读真实未来**（协议失效）⇒ 本类**拒绝构造**。
  * **顺序固定**（规格 §2.4）：候选评估 → `certify_fn` → 提交；**每单事务化**：
    任何一步抛错 ⇒ 该单**不**进入 accepted/rejected、计划**回滚**（不留半完成状态）。
  * 每个决策经 `DecisionLogger` 落盘（写入即校验）⇒ 可直接跑 T2–T7 审计。

本模块**不读效用、不跑实验**；`decide_fn` / `certify_fn` / `reveal_time_of` 由调用方注入。
"""
from __future__ import annotations

from role_c_decision_log import DecisionLogger, StateError


def _vid(k):
    """快照里的车辆键还原：JSON 会把 int 键变成 str ⇒ 能转回 int 就转回。"""
    try:
        return int(k)
    except (TypeError, ValueError):
        return k


class RoleCReplanner:
    """Role C 在线接拒骨架。

    参数
    ----
    budget, capacity, booking_horizon, contract, cooling_share : 与对照臂**同值**
    decide_fn(order, ctx) -> (accept: bool, diag: dict)
        `ctx` 提供 `{'inst_idx','clock','visible_ids','accepted','remaining_budget','plan'}`
    certify_fn(plan) -> (ok: bool, energy: float)
    reveal_time_of(inst_idx, order) -> float      # 缺省读 `env.reveal_time[inst_idx, o]`
    logger : DecisionLogger（可选；给了就落盘）
    """

    def __init__(self, *, budget, capacity, booking_horizon, contract, cooling_share,
                 decide_fn, certify_fn, reveal_time_of=None, logger=None,
                 max_total_accepts=None):
        if booking_horizon is None:
            raise ValueError(
                'booking_horizon 不得为 None：此时 StrictOnlineEnv.has_future_reveal 会'
                '**读真实未来**（源码自述"旧行为…非因果主张"）⇒ strict-online 协议失效。'
                '请传项目常量 BOOKING_HORIZON(=16.0)（P1A_CONTROL_FROZEN.json '
                'must_record_causal_flag）。')
        self.budget = float(budget)
        self.capacity = float(capacity)
        self.booking_horizon = float(booking_horizon)
        self.contract = contract
        self.cooling_share = float(cooling_share)
        self.decide_fn = decide_fn
        self.certify_fn = certify_fn
        self._reveal_time_of = reveal_time_of
        self.logger = logger
        self.max_total_accepts = max_total_accepts

        self._accepted = set()
        self._rejected = set()
        self._plan = {}                      # vehicle_id -> [order...]
        self._status_inst = None
        self._decisions = []                 # (clock, order, accept, diag)

    # ---------------- 状态 ----------------
    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = inst_idx
            self._accepted, self._rejected, self._plan = set(), set(), {}
            self._decisions = []
            if self.logger is not None:
                # 新实例 ⇒ 清空日志（一次运行一个实例；跨实例混记会让 day_id 配对失效）。
                # ⚠️ **原地清空**，不要 new 新对象 —— 否则调用方持有的 logger 句柄会失效
                # （初版就是这么错的：传入的 logger 被丢弃，records 永远为空）。
                self.logger.reset()

    def reveal_time(self, env, inst_idx, order):
        if self._reveal_time_of is not None:
            return float(self._reveal_time_of(inst_idx, order))
        return float(env.reveal_time[inst_idx, order])

    def remaining_budget(self, env, inst_idx, marginal_energy_fn=None):
        """剩余预算 = B − 已接受订单的边际能耗之和（与既有实现同口径）。"""
        if marginal_energy_fn is None:
            return self.budget
        used = sum(float(marginal_energy_fn(o)) for o in self._accepted)
        return self.budget - used

    # ---------------- 事件：揭示 ----------------
    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        """在一个揭示事件上对**新订单**逐个做接/拒决策；每单事务化。"""
        self._reset_if_new(inst_idx)
        reserved = set()
        if hasattr(env, 'get_reserved_customers'):
            reserved = set(int(i) for i in env.get_reserved_customers(vehicles))
        new_set = [int(i) for i in visible_ids
                   if not served_mask[i] and int(i) not in reserved
                   and int(i) not in self._accepted and int(i) not in self._rejected]
        # 处理顺序 = 揭示时刻升序（与既有实现一致；保证确定性）
        new_set.sort(key=lambda o: (self.reveal_time(env, inst_idx, o), o))
        for o in new_set:
            self._decide_one(env, inst_idx, clock, o, vehicles, served_mask, visible_ids)

    def _decide_one(self, env, inst_idx, clock, order, vehicles, served_mask, visible_ids):
        """单订单事务：**候选评估 → 认证 → 提交**；任何异常/未过认证 ⇒ 回滚且不采纳计划。

        `decide_fn` 通过 `diag['plan']` 返回**候选计划**（dict vehicle_id → [orders]）；
        本类**只在"接单 ∧ 通过认证"时采纳**它。路线搜索因此完全由注入层负责
        （§2.2：C 版用独立前瞻搜索，A/B 版用共享修复 —— 本骨架不预设）。
        """
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        remaining = self.remaining_budget(env, inst_idx)
        ctx = {'inst_idx': inst_idx, 'clock': float(clock), 'order': order,
               'visible_ids': list(visible_ids), 'accepted': sorted(self._accepted),
               'remaining_budget': remaining,
               'plan': {k: list(v) for k, v in self._plan.items()},
               'booking_horizon': self.booking_horizon}
        accept, diag = self.decide_fn(order, ctx)
        cand = (diag or {}).get('plan') if accept else None
        certify_ok, certify_energy = self.certify_fn(
            self._plan if cand is None else cand)
        if accept and certify_ok:
            if self.max_total_accepts is not None \
                    and len(self._accepted) >= int(self.max_total_accepts):
                accept, certify_ok = False, False          # 上限 ⇒ 转为拒单
        if accept and certify_ok:
            if cand is not None:                            # 采纳候选计划
                self._plan = {_vid(k): list(v) for k, v in cand.items()}
            self._accepted.add(order)
        else:
            self._plan = saved_plan                         # **回滚**（不采纳候选）
            self._rejected.add(order)
        self._decisions.append((float(clock), order, bool(accept), diag))
        if self.logger is not None:
            self._log(inst_idx, clock, order, accept, certify_ok, certify_energy, diag)
        return accept

    def _log(self, inst_idx, clock, order, accept, certify_ok, certify_energy, diag):
        import hashlib
        payload = '%s|%s|%s|%s' % (inst_idx, clock, order, accept)
        dh = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
        try:
            self.logger.log_event(
                inst_idx=inst_idx, day_id=diag.get('day_id', inst_idx), clock=clock,
                info_max_reveal=diag.get('info_max_reveal', clock),
                action='accept' if accept else 'reject', certified=bool(certify_ok),
                decision_hash=dh, order_id=(order if accept else None),
                certify_energy=(certify_energy if accept else None))
        except StateError:
            # 写入被拒 ⇒ 这是**日志与状态不一致**的信号，必须上抛，不能吞
            raise

    # ---------------- 计划：回放 ----------------
    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        """把持久计划回放到车辆的 `mutable_suffix`（**不改决策**）。"""
        self._reset_if_new(inst_idx)
        has_future = float(clock) < self.booking_horizon - 1e-6
        active = [v for v in vehicles if getattr(v, 'status', None) in ('idle', 'ready')
                  and (replan_ids is None or v.vehicle_id in replan_ids)]
        for v in active:
            route = [o for o in self._plan.get(v.vehicle_id, []) if not served_mask[o]]
            if route:
                v.mutable_suffix = route           # 不含尾部 0（由 plan_exhaustion 决定 WAIT/返回）
            else:
                v.mutable_suffix = ([] if (getattr(v, 'current_node', 0) != 0 and has_future)
                                    else [0])
        return None

    # ---------------- 快照 ----------------
    def export_state(self):
        return {'version': 'role-c-v1', 'budget': self.budget,
                'booking_horizon': self.booking_horizon,
                'accepted': sorted(self._accepted), 'rejected': sorted(self._rejected),
                'plan': {str(k): list(v) for k, v in self._plan.items()},
                'status_inst': self._status_inst}

    def restore_state(self, state):
        if not state:
            return
        if state.get('version') != 'role-c-v1':
            raise ValueError('未知快照版本 %r' % (state.get('version'),))
        self._accepted = set(state.get('accepted') or ())
        self._rejected = set(state.get('rejected') or ())
        self._plan = {_vid(k): list(v) for k, v in (state.get('plan') or {}).items()}
        self._status_inst = state.get('status_inst')
