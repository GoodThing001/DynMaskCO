"""A-v1 RRNCO-Ordering 接单决策适配（2026-09-29，用户授权「对比方法扩充」）。

角色：把冻结的 RRNCO-Ordering-RH-D 管线（CC_Compare/RRNCO/dcc_rh_v4，R1 seal rev1，
只读导入）接入 A-v1 accept/reject 协议的接单决策：
  每次揭示，对每个新订单 o（按揭示时间序）：
    pool = 已接受未服务 ∪ {o}；
    由 env 状态构造 duck-typed view（可见信息边界与求解器臂一致）→
    RRNCOGuidedAdapter.propose(view)（逐车子问题 → 真实 epoch_199.ckpt 排序 →
    确定性协调 → 独立认证 suffix）；
    接受 iff：无 fallback、无 deferred、覆盖全部 pool、真实 C0 certify_plan ≤ B；
    否则拒绝 o（保旧计划）；越限（timeout）= 保旧计划 + 拒单（2026-09-30 时限语义修复，
    旧版「先提交接单、再记录超时」已废止）。
决策语义与 OR-Tools/PyVRP 臂同为「myopic feasibility + 学习型路由器」，
隔离「学习排序 vs 距离优化」的路由强度差异。

运行环境：cc_compare conda env（torch 2.11/rl4co 0.6.0/tensordict 0.13.0，
checkpoint CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt，sha b1ff3191…）。
本文件是 A-v1 新代码，不修改任何冻结文件（dcc_rh_v4 只读导入）。
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RRNCO_DCC = os.path.normpath(os.path.join(_SCRIPTS, '..', '..', 'CC_Compare',
                                           'RRNCO', 'dcc_rh_v4'))
_RRNCO_ROOT = os.path.normpath(os.path.join(_SCRIPTS, '..', '..', 'CC_Compare', 'RRNCO'))
_CC_COMMON = os.path.normpath(os.path.join(_SCRIPTS, '..', '..', 'CC_Compare', 'common'))
for _p in (_RRNCO_DCC, _RRNCO_ROOT, _CC_COMMON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_energy_c0 import C0ReserveReplanner, certify_plan  # noqa: E402


class OrderingAcceptReplanner(C0ReserveReplanner):
    """通用「学习排序 + 全覆盖接单」replanner（2026-09-29 泛化）：
    任意 provider.order(sp) -> tuple[int,...]（可见池排序）经冻结 dcc_rh_v4 协调器
    打包成多车 suffix，全覆盖 + 真实 C0 certify ≤ B 才接单。provider 可换：
    RRNCO（真实 ckpt）、RouteFinder、CaDA、MVMoE、POMO/AM 等（各方法独立 provider 文件）。
    """

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 provider, time_limit=10.0):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share)
        from rrnco_guided_adapter import RRNCOGuidedAdapter

        self.time_limit = float(time_limit)
        self.timeouts = 0
        self.n_solves = 0          # = 决策次数（与求解器臂 meta 同名口径）
        self.n_fail = 0            # = fallback 次数
        self.solve_time_s = 0.0    # = provider 推理耗时
        self.max_decide_s = 0.0    # = 当日单决策最大墙钟（逐事件耗时审计）
        self._reject_reasons = {}
        self._adapter = RRNCOGuidedAdapter(provider)

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self.timeouts = 0
            self.n_solves = 0
            self.n_fail = 0
            self.solve_time_s = 0.0
            self.max_decide_s = 0.0
            self._reject_reasons = {}

    # ------------------------------------------------------------------ #
    # env 状态 → duck-typed view（subproblem.py 只取 pool+anchor，结构隔离）
    # ------------------------------------------------------------------ #
    def _build_view(self, env, inst_idx, clock, vehicles, pool):
        n = env.num_nodes
        speed = float(env.tw_speed)
        dist = env.dist_mat[inst_idx]
        veh_specs = []
        for v in vehicles:
            pos = self._vehicle_serve_state(env, inst_idx, v, clock)
            if pos is None:
                continue
            cur, t, load = pos
            veh_specs.append(SimpleNamespace(
                vehicle_id=int(v.vehicle_id),
                anchor_node_id=int(cur),
                anchor_idx=int(cur),      # 节点 id == 数组索引（真实节点空间）
                ready_time=float(t),
                load=float(load),
                status=v.status,
                mutable_suffix=tuple(),
            ))
        view = SimpleNamespace(
            node_ids=tuple(range(n)),
            coords=tuple(tuple(float(x) for x in env.coords[inst_idx, i]) for i in range(n)),
            demands=tuple(float(env.demands[inst_idx, i]) for i in range(n)),
            tw_start=tuple(float(env.tw_start[inst_idx, i]) for i in range(n)),
            tw_end=tuple(float(env.tw_end[inst_idx, i]) for i in range(n)),
            service_time=tuple(float(env.service_time[inst_idx, i]) for i in range(n)),
            dist_mat=tuple(tuple(float(dist[i, j]) for j in range(n)) for i in range(n)),
            travel_mat=tuple(tuple(float(dist[i, j] / speed) for j in range(n))
                             for i in range(n)),
            capacity=float(self.capacity),
            depot_tw_end=float(env.tw_end[inst_idx, 0]),
            pool_customer_ids=tuple(sorted(int(c) for c in pool)),
            vehicles=tuple(veh_specs),
            replan_ids=tuple(sorted(int(s.vehicle_id) for s in veh_specs)),
            has_future_reveal=bool(clock < self.booking_horizon - 1e-6),
        )
        return view

    def _decide(self, env, inst_idx, clock, vehicles, served_mask, o, t0):
        """单决策。2026-09-30 时限语义修复：越限=保旧计划+拒单（先检查后提交），
        t0 为 on_reveal 传入的决策起点；推理后与认证后各检查一次，
        杜绝「先提交接单、再记录超时」的旧缺口（用户复核报告第 2 点）。"""
        pool = sorted(set(int(x) for x in self._accepted
                          if not served_mask[x]) | {int(o)})
        # 2026-09-30 修复（duplicate_service，RouteFinder 诊断臂 36/40 天复现）：
        # 在途 committed_next 已开始执行（不可撤销、不可重分配），必须从重建池排除；
        # 否则协调器可能把它分给另一辆车 → 同一客户被服务两次 → 硬失败。
        # （reveal 时 mutable tail 已被 env 清空，故只需排除在途项；certify_plan 对
        # committed 车会把 committed_next 放回路线头部，服务次数仍为一次。）
        in_transit = {int(v.committed_next) for v in vehicles
                      if getattr(v, 'status', None) == 'committed'
                      and getattr(v, 'committed_next', None) not in (None, 0)}
        pool = sorted(set(pool) - in_transit)
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        view = self._build_view(env, inst_idx, clock, vehicles, pool)
        proposal = self._adapter.propose(view)
        self.n_solves += 1
        self.solve_time_s += float(proposal.model_runtime_s)
        elapsed = time.monotonic() - t0
        self.max_decide_s = max(self.max_decide_s, elapsed)
        if time.monotonic() > t0 + self.time_limit:   # 推理结束越限 → 保旧计划 + 拒单
            self.timeouts += 1
            self._plan = saved_plan
            return False, 'timeout'
        if proposal.fallback_triggered:
            self.n_fail += 1
            return False, 'fallback'
        deferred = list(proposal.solve_meta.get('deferred', []))
        if deferred:
            return False, 'deferred'
        plan = {int(vid): [int(c) for c in suffix if c != 0]
                for vid, suffix in proposal.suffixes.items()}
        covered = sorted(c for r in plan.values() for c in r)
        if covered != pool:
            return False, 'coverage_mismatch'
        ok, _ = certify_plan(env, inst_idx, clock, vehicles, served_mask, plan,
                             self.contract, self.budget)
        self.max_decide_s = max(self.max_decide_s, time.monotonic() - t0)
        if time.monotonic() > t0 + self.time_limit:   # 认证结束越限（含失败分支）→ 拒单
            self.timeouts += 1
            self._plan = saved_plan
            return False, 'timeout'
        if not ok:
            return False, 'certify_budget'
        self._plan = plan
        return True, 'accepted'

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = {int(i) for i in visible_ids if not served_mask[i]
                   and int(i) not in reserved and int(i) not in self._accepted
                   and int(i) not in self._rejected}
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            t0 = time.monotonic()
            ok, reason = self._decide(env, inst_idx, clock, vehicles, served_mask, o, t0)
            if ok:
                self._accepted.add(o)
            else:
                self._rejected.add(o)
                self._reject_reasons[reason] = self._reject_reasons.get(reason, 0) + 1


class RRNCOAcceptReplanner(OrderingAcceptReplanner):
    """RRNCO-Ordering 接单（真实 epoch_199.ckpt provider，见模块 docstring）。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 checkpoint_path, device='cuda', seed=0, time_limit=10.0):
        from rrnco_backend import BackendConfig, RRNCOPreferenceProvider
        provider = RRNCOPreferenceProvider(
            BackendConfig(checkpoint_path=checkpoint_path, device=device, seed=seed))
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         provider, time_limit=time_limit)
