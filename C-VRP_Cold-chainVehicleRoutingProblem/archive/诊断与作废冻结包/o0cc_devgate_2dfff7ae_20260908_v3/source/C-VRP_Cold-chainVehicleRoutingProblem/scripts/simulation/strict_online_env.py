"""
Strict Online Environment — 唯一事件引擎 + FleetState + execution trace（P0-SIM）。

所有方法（EDD / DynMaskCO / OR-Tools）共享同一事件循环、冻结前缀、车辆时间推进、
execution trace。方法只提供 Replanner（给定当前状态，为每辆 idle/ready 车决定下一步）。

核心语义（R1.5，2026-08-28）：
  - Event = Order Reveal ∪ Service Completion ∪ Return Depot。
  - 车辆服务完 committed 节点后变 'ready'（NOT closed），可在下一事件继续接 reveal 客户。
  - 只有真正返回 depot（returning → closed）才关闭；return 是显式 committed 移动。
  - P0-A（时间穿越修复）：idle 车第一次 dispatch 时 ready_time = clock，绝不再从 t=0 出发。
  - P0-B（跨车保留修复）：committed 车的 next 客户对其他车不可见（get_reserved_customers）。
  - P0-C（WAIT 语义）：ready 车无任务且有未来 reveal 时保持 ready（WAIT），不强制返回 depot；
    env 不做「自动 return」这个策略动作，只在最后 _force_return_all 统一收尾。
  - 所有 TW / capacity / cost / depot-return 从 execution trace 算（evaluate_execution_trace）。

Author: P0-Protocol Repair (R1.5)
Date: 2026-08-28
"""

import os
import sys

import numpy as np
from dataclasses import dataclass, field


_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLDCHAIN_ROOT = os.path.join(_SCRIPTS_ROOT, 'coldchain')
if _COLDCHAIN_ROOT not in sys.path:
    sys.path.insert(0, _COLDCHAIN_ROOT)

from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                             total_quality_loss, transition_segment)


@dataclass
class ServiceRecord:
    vehicle_id: int
    prev_node: int
    node: int
    depart_time: float      # 从 prev_node 出发时刻（= ready_time）
    arrival_time: float
    service_start: float
    service_finish: float
    coldchain_before: object = None
    coldchain_after: object = None
    picked_order_id: int = None
    segment_energy_kwh: float = 0.0
    segment_quality_loss: float = 0.0
    segment_distance_km: float = 0.0
    segment_thermal_violation_count: int = 0
    segment_thermal_violation_duration_h: float = 0.0


@dataclass
class VehicleTrace:
    vehicle_id: int
    dispatch_time: float = None
    services: list = field(default_factory=list)
    return_depart: float = None
    return_arrival: float = None
    final_coldchain_state: object = None
    depot_unload_records: list = field(default_factory=list)
    dispatch_preconditioning_energy_kwh: float = 0.0
    return_segment_energy_kwh: float = 0.0
    return_segment_quality_loss: float = 0.0
    return_segment_distance_km: float = 0.0
    return_segment_thermal_violation_count: int = 0
    return_segment_thermal_violation_duration_h: float = 0.0


@dataclass
class VehicleState:
    vehicle_id: int
    status: str = 'idle'            # idle / ready / committed / returning / closed
    current_node: int = 0
    ready_time: float = 0.0         # 服务完 current_node 可离开时刻（idle 车 = dispatch 时刻）
    current_load: float = 0.0
    committed_next: int = None
    committed_arrive: float = None
    committed_finish: float = None
    mutable_suffix: list = field(default_factory=list)   # 未 committed 计划（含末位 0=return）
    served_route: list = field(default_factory=list)
    dispatch_time: float = None
    return_finish: float = None     # returning 时到达 depot 时刻
    needs_replan: bool = False      # P0-CTRL：是否需要在下一决策点重规划（默认 False）
    replan_reason: str = None       # P0-M1：重规划触发原因（initial / reveal / plan_exhaustion）
    coldchain_state: object = None
    coldchain_updated_time: float = None
    committed_coldchain_before: object = None
    coldchain_accounting_anchor: object = None


class Replanner:
    """方法实现的接口。

    plan() 对每辆 'idle'/'ready' 车设置 v.mutable_suffix：
      - [c1, c2, ..., 0]：连续服务 c1,c2,... 后返回 depot（env 只 commit 第一个）。
      - [0]：立即返回 depot。
      - []：WAIT（保持 ready，当前不 commit 新 leg，等下一事件再 plan）。

    replan_ids（P0-CTRL）：本次需要重规划的车 id 集合。None = 所有 idle/ready 车。
    非 replan_ids 的车保留旧 mutable_suffix（plan persistence），不重规划。
    """
    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        raise NotImplementedError

    def export_state(self):
        """返回 replanner 的**决策状态**（影响后续 plan/ownership，随 snapshot 恢复）。

        审计状态（累计计数/日志）不在此返回——每条 rollout 分支独立，不跨分支串扰。
        默认无状态。
        """
        return None

    def restore_state(self, state):
        """恢复 replanner 决策状态（export_state 的逆操作）。默认无操作。"""
        return None


class StrictOnlineEnv:
    def __init__(self, dataset, capacity, tw_speed=1.0, num_vehicles=25, replanner=None,
                 coldchain_contract=None):
        self.dataset = dataset
        self.capacity = capacity
        self.tw_speed = tw_speed
        self.num_vehicles = num_vehicles
        self.replanner = replanner
        self.coldchain_contract = coldchain_contract

        self.coords = dataset['coords'].astype(np.float32)          # (N, nodes, 2)
        self.demands = dataset['demands'].astype(np.float32)        # (N, nodes)
        self.tw_start = dataset['tw_start'].astype(np.float32)      # (N, nodes)
        self.tw_end = dataset['tw_end'].astype(np.float32)          # (N, nodes)
        self.service_time = dataset.get(
            'service_time', np.zeros_like(self.demands, dtype=np.float32))
        self.reveal_time = dataset.get(
            'reveal_time', np.zeros_like(self.demands, dtype=np.float32))

        if self.coldchain_contract is not None:
            self.coldchain_contract.validate()
            expected_capacity = self.coldchain_contract.operational.shared_vehicle_capacity
            if abs(float(capacity) - float(expected_capacity)) > 1e-6:
                raise ValueError(
                    'env capacity %.6f != cold-chain contract shared capacity %.6f'
                    % (capacity, expected_capacity))
            if 'temp_class' not in dataset:
                raise ValueError('cold-chain execution requires dataset temp_class')
            self.temp_class = dataset['temp_class'].astype(np.int32)
            self.initial_quality = dataset.get(
                'initial_quality', np.ones_like(self.demands, dtype=np.float32)
            ).astype(np.float32)
        else:
            self.temp_class = None
            self.initial_quality = None

        self.num_instances = self.coords.shape[0]
        self.num_nodes = self.coords.shape[1]

        # A/B 验证指标：replanner 被调用的次数（plan persistence 后应从几十降到初始+reveal 次数）
        self.replan_count = 0
        # P0-M1：物理事件 id（audit 用，区分 clock 接近的不同事件）
        self.event_id = 0
        # R1.7-4：branch intervention —— (event_id, vehicle_id) -> 强制 suffix（覆盖 replanner 输出）。
        # 用于 long-horizon counterfactual：在指定决策点强制安装 incumbent/candidate 的完整 suffix。
        self.force_suffix = {}

        # P0-S：决策点快照钩子（recourse_snapshot.capture_recourse_snapshot 的测试/teacher 入口）。
        # 签名：hook(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers)
        self.snapshot_hook = None

        # O0：决策点干预钩子（sequential oracle 用）。在 _plan_and_commit 之前调用，
        # 可写 env.force_suffix 以覆盖 replanner 输出。签名同 snapshot_hook。
        self.oracle_hook = None

        # P1（dist_mat 支持）：dataset 有 dist_mat 就用它（asymmetric / real network）
        if 'dist_mat' in dataset:
            self.dist_mat = dataset['dist_mat'].astype(np.float32)
        else:
            diff = self.coords[:, :, None, :] - self.coords[:, None, :, :]
            self.dist_mat = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # ------------------------------------------------------------------
    def _travel(self, inst_idx, i, j):
        return self.dist_mat[inst_idx, i, j] / self.tw_speed

    def _coldchain_active_mask(self):
        if self.coldchain_contract is None:
            return ()
        return (True,) * len(self.coldchain_contract.thermal.supported_temp_classes)

    def _dispatch_coldchain(self, v, depart):
        if self.coldchain_contract is None or v.coldchain_state is not None:
            return
        initial = create_vehicle_state(self.coldchain_contract)
        v.coldchain_state = dispatch_vehicle(initial, self.coldchain_contract)
        v.coldchain_updated_time = float(depart)
        v.coldchain_accounting_anchor = v.coldchain_state

    def _advance_coldchain_to(self, v, clock):
        """Advance all in-cargo time, including travel, waiting and service."""
        if self.coldchain_contract is None or v.coldchain_state is None:
            return
        if v.coldchain_state.closed:
            return
        start = float(v.coldchain_updated_time)
        end = float(clock)
        if end < start - 1e-6:
            raise ValueError('cold-chain clock moved backwards')
        if end <= start + 1e-9:
            return
        v.coldchain_state, _ = transition_segment(
            v.coldchain_state,
            depart_time=start,
            arrival_time=end,
            service_finish=end,
            served_customer=None,
            active_zone_mask=self._coldchain_active_mask(),
            contract=self.coldchain_contract,
        )
        v.coldchain_updated_time = end

    def get_reserved_customers(self, vehicles):
        """committed 车的 committed_next + 所有车的 mutable_suffix（tail）都不可再分配。

        P0-B：committed leg 不可撤销。
        P0-CTRL-4：tail 也要 reserve，否则 reveal 后分批重规划时，先重规划的车会拿走
        别的车 tail 里的客户，导致 stale-tail duplicate。
        """
        reserved = set()
        for v in vehicles:
            if v.status == 'committed' and v.committed_next not in (None, 0):
                reserved.add(int(v.committed_next))
            for n in v.mutable_suffix:
                if n != 0:
                    reserved.add(int(n))
        return reserved

    def has_future_reveal(self, inst_idx, clock, served_mask):
        """是否还有未服务客户在未来 reveal（用于 WAIT vs CLOSE 决策）。P0-C。"""
        return any(not served_mask[i]
                   and self.reveal_time[inst_idx, i] > clock + 1e-6
                   for i in range(1, self.num_nodes))

    def _commit(self, inst_idx, clock, v, traces, node):
        """把 v commit 到 node（node 为 customer 或 0=return depot）。

        P0-A：depart = max(ready_time, clock)。idle 车第一次 dispatch 或 ready 车
        WAIT 后继续，实际 depart 不得早于当前 clock（消除时间穿越）。
        """
        depart = max(v.ready_time, clock)
        if node == 0:
            if v.current_node == 0:
                v.status = 'idle' if v.served_route == [] else 'closed'
                v.return_finish = depart
                return
            self._advance_coldchain_to(v, depart)
            v.return_finish = depart + self._travel(inst_idx, v.current_node, 0)
            v.status = 'returning'
            v.committed_next = 0
            v.committed_finish = v.return_finish
            v.ready_time = depart
            v.committed_coldchain_before = v.coldchain_state
            return
        # customer
        self._dispatch_coldchain(v, depart)
        if (self.coldchain_contract is not None and traces is not None
                and not traces[v.vehicle_id].services):
            traces[v.vehicle_id].dispatch_preconditioning_energy_kwh = float(
                v.coldchain_state.cumulative_energy_kwh)
        self._advance_coldchain_to(v, depart)
        arrive = depart + self._travel(inst_idx, v.current_node, node)
        service_start = max(arrive, self.tw_start[inst_idx, node])
        finish = service_start + self.service_time[inst_idx, node]
        v.committed_next = node
        v.committed_arrive = arrive
        v.committed_finish = finish
        v.status = 'committed'
        v.ready_time = depart  # 记录实际 depart（ServiceRecord 用）
        v.committed_coldchain_before = v.coldchain_state
        if v.dispatch_time is None and v.served_route == [] and v.current_node == 0:
            v.dispatch_time = depart
            if traces is not None:
                traces[v.vehicle_id].dispatch_time = float(depart)

    def _advance_fleet(self, inst_idx, clock, vehicles, traces, served_mask):
        """把 finish <= clock 的 committed / returning 车辆推进完成。"""
        for v in vehicles:
            self._advance_coldchain_to(v, clock)
            if v.status == 'committed':
                if v.committed_finish is not None and v.committed_finish <= clock + 1e-6:
                    node = v.committed_next
                    cc_before = (v.coldchain_accounting_anchor
                                 if self.coldchain_contract is not None else None)
                    segment_energy = 0.0
                    segment_quality = 0.0
                    segment_distance = 0.0
                    segment_violation_count = 0
                    segment_violation_duration = 0.0
                    cc_after = None
                    if self.coldchain_contract is not None:
                        pre_event = v.coldchain_state
                        v.coldchain_state, event_metrics = transition_segment(
                            pre_event,
                            depart_time=float(clock),
                            arrival_time=float(clock),
                            service_finish=float(clock),
                            served_customer=int(node),
                            active_zone_mask=self._coldchain_active_mask(),
                            contract=self.coldchain_contract,
                            order_quantity=float(self.demands[inst_idx, node]),
                            order_temp_class=int(self.temp_class[inst_idx, node]),
                            initial_quality=float(self.initial_quality[inst_idx, node]),
                            segment_distance_units=float(
                                self.dist_mat[inst_idx, v.current_node, node]),
                        )
                        v.coldchain_updated_time = float(clock)
                        cc_after = v.coldchain_state
                        v.coldchain_accounting_anchor = cc_after
                        segment_energy = (
                            cc_after.cumulative_energy_kwh
                            - cc_before.cumulative_energy_kwh)
                        segment_quality = (
                            total_quality_loss(cc_after, self.coldchain_contract)
                            - total_quality_loss(cc_before, self.coldchain_contract))
                        segment_distance = (
                            cc_after.cumulative_distance_km
                            - cc_before.cumulative_distance_km)
                        segment_violation_count = (
                            cc_after.thermal_violation_count
                            - cc_before.thermal_violation_count)
                        segment_violation_duration = (
                            cc_after.thermal_violation_duration_h
                            - cc_before.thermal_violation_duration_h)
                    traces[v.vehicle_id].services.append(ServiceRecord(
                        vehicle_id=v.vehicle_id,
                        prev_node=v.current_node,
                        node=node,
                        depart_time=v.ready_time,
                        arrival_time=v.committed_arrive,
                        service_start=max(v.committed_arrive, self.tw_start[inst_idx, node]),
                        service_finish=v.committed_finish,
                        coldchain_before=cc_before,
                        coldchain_after=cc_after,
                        picked_order_id=int(node) if self.coldchain_contract is not None else None,
                        segment_energy_kwh=float(segment_energy),
                        segment_quality_loss=float(segment_quality),
                        segment_distance_km=float(segment_distance),
                        segment_thermal_violation_count=int(segment_violation_count),
                        segment_thermal_violation_duration_h=float(segment_violation_duration),
                    ))
                    served_mask[node] = True
                    v.served_route.append(node)
                    v.current_node = node
                    v.ready_time = v.committed_finish
                    if self.coldchain_contract is not None:
                        v.current_load = v.coldchain_state.total_load
                    else:
                        v.current_load += self.demands[inst_idx, node]
                    v.committed_next = None
                    v.committed_arrive = None
                    v.committed_finish = None
                    v.committed_coldchain_before = None
                    v.status = 'ready'
                    # P0-M1：PlanExhaustion —— 当前 committed leg 完成后无旧 plan 可继续，
                    # 则下一决策点必须重规划（不覆盖更强的 reveal 原因）。
                    # 注意：只对「plan 用完」的 ready 车，不做「所有 ready 每次重规划」。
                    if not v.mutable_suffix and not v.needs_replan:
                        v.needs_replan = True
                        v.replan_reason = 'plan_exhaustion'
            elif v.status == 'returning':
                if v.return_finish is not None and v.return_finish <= clock + 1e-6:
                    if self.coldchain_contract is not None:
                        cc_before = v.coldchain_accounting_anchor
                        before_count = len(v.coldchain_state.delivered_to_depot)
                        v.coldchain_state, _ = transition_segment(
                            v.coldchain_state,
                            depart_time=float(clock),
                            arrival_time=float(clock),
                            service_finish=float(clock),
                            served_customer=None,
                            active_zone_mask=self._coldchain_active_mask(),
                            contract=self.coldchain_contract,
                            return_to_depot=True,
                            segment_distance_units=float(
                                self.dist_mat[inst_idx, v.current_node, 0]),
                        )
                        v.coldchain_updated_time = float(clock)
                        traces[v.vehicle_id].depot_unload_records.extend(
                            v.coldchain_state.delivered_to_depot[before_count:])
                        traces[v.vehicle_id].final_coldchain_state = v.coldchain_state
                        v.coldchain_accounting_anchor = v.coldchain_state
                        traces[v.vehicle_id].return_segment_energy_kwh = float(
                            v.coldchain_state.cumulative_energy_kwh
                            - cc_before.cumulative_energy_kwh)
                        traces[v.vehicle_id].return_segment_quality_loss = float(
                            total_quality_loss(v.coldchain_state, self.coldchain_contract)
                            - total_quality_loss(cc_before, self.coldchain_contract))
                        traces[v.vehicle_id].return_segment_distance_km = float(
                            v.coldchain_state.cumulative_distance_km
                            - cc_before.cumulative_distance_km)
                        traces[v.vehicle_id].return_segment_thermal_violation_count = int(
                            v.coldchain_state.thermal_violation_count
                            - cc_before.thermal_violation_count)
                        traces[v.vehicle_id].return_segment_thermal_violation_duration_h = float(
                            v.coldchain_state.thermal_violation_duration_h
                            - cc_before.thermal_violation_duration_h)
                    traces[v.vehicle_id].return_arrival = v.return_finish
                    traces[v.vehicle_id].return_depart = v.ready_time
                    v.status = 'closed'
                    v.current_node = 0
                    v.current_load = 0.0
                    v.ready_time = v.return_finish
                    v.return_finish = None
                    v.committed_coldchain_before = None

    def run(self, inst_idx, force_suffix=None):
        """运行单个实例。返回 (traces, served_mask)。

        P0-CTRL：物理事件（Reveal / ServiceCompletion / ReturnCompletion）≠ 重规划触发器。
        只有 t=0 / Reveal / PlanExhaustion / PlanInvalidation 才触发 replanner；
        无新信息的 service completion 直接延续旧 plan（pop mutable_suffix 下一个）。

        force_suffix（R1.7-4）：dict (event_id, vehicle_id) -> suffix，在对应决策点强制安装
        suffix（覆盖 replanner 输出，绕过 WAIT 逻辑）。用于 long-horizon counterfactual。
        """
        self.force_suffix = force_suffix or {}
        try:
            vehicles = [VehicleState(vehicle_id=v) for v in range(self.num_vehicles)]
            traces = [VehicleTrace(vehicle_id=v) for v in range(self.num_vehicles)]
            served_mask = np.zeros(self.num_nodes, dtype=bool)
            served_mask[0] = True

            all_customers = [i for i in range(1, self.num_nodes)
                             if self.demands[inst_idx, i] > 0]
            reveal_events = sorted(set(float(self.reveal_time[inst_idx, i])
                                         for i in all_customers
                                         if self.reveal_time[inst_idx, i] > 0))
            horizon = float(self.tw_end[inst_idx].max())

            clock = 0.0
            reveal_idx = 0
            self.replan_count = 0
            self.event_id = 0

            # P0-CTRL：初始 t=0，所有车需要首次规划
            for v in vehicles:
                v.needs_replan = True
                v.replan_reason = 'initial'
            self._maybe_snapshot(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                 all_customers)
            self._run_oracle_hook(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                  all_customers)
            self._plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers)
            return self._run_loop(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                  all_customers, reveal_events, horizon)
        finally:
            self.force_suffix = {}

    def _run_loop(self, inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                  all_customers, reveal_events, horizon):
        """从当前决策点状态继续事件循环直到终止。返回 (traces, served_mask)。

        P0-S：run() 与 run_resumed() 共享本循环，保证 resume 与原 rollout 后缀 exact parity。
        """
        max_iter = 10000
        for _ in range(max_iter):
            all_served = all(served_mask[c] for c in all_customers)
            no_pending = all(v.status in ('idle', 'closed') for v in vehicles)
            if all_served and no_pending:
                break

            # 下一个物理事件 = min(下一 reveal, 各 committed finish, 各 return finish)
            next_clock = horizon
            while reveal_idx < len(reveal_events) and reveal_events[reveal_idx] <= clock + 1e-6:
                reveal_idx += 1
            next_reveal = reveal_events[reveal_idx] if reveal_idx < len(reveal_events) else None
            if next_reveal is not None:
                next_clock = min(next_clock, next_reveal)
            for v in vehicles:
                if v.status == 'committed' and v.committed_finish is not None:
                    next_clock = min(next_clock, v.committed_finish)
                elif v.status == 'returning' and v.return_finish is not None:
                    next_clock = min(next_clock, v.return_finish)

            if next_clock > horizon - 1e-6:
                self._force_return_all(inst_idx, clock, vehicles, traces, served_mask)
                break

            clock = next_clock
            self.event_id += 1

            # 判断当前物理事件类型（决定是否触发重规划）
            is_reveal = (next_reveal is not None and abs(clock - next_reveal) < 1e-6)

            # 推进 fleet：service completion / return completion 更新 FleetState
            self._advance_fleet(inst_idx, clock, vehicles, traces, served_mask)

            # P0-CTRL：reveal = 新信息 → 所有未 closed 车 needs_replan=True。
            # committed 车保留标记，等完成当前不可撤销 leg 变 ready 后再重规划。
            if is_reveal:
                for v in vehicles:
                    if v.status != 'closed':
                        v.needs_replan = True
                        v.replan_reason = 'reveal'
                        v.mutable_suffix = []  # P0-CTRL-3：reveal invalidates mutable tail

            self._maybe_snapshot(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                 all_customers)
            self._run_oracle_hook(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                  all_customers)
            self._plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers)

        return traces, served_mask

    def _maybe_snapshot(self, inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                        all_customers):
        """P0-S：决策点快照钩子（每个 _plan_and_commit 前调用一次）。"""
        if self.snapshot_hook is not None:
            self.snapshot_hook(self, inst_idx, float(clock), int(self.event_id), int(reveal_idx),
                               vehicles, traces, served_mask, all_customers)

    def _run_oracle_hook(self, inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                         all_customers):
        """O0：决策点干预钩子。

        动作时机冻结：只在「存在需要重规划的 idle/ready 车」时触发（与 JF1-H replanner
        的 replan_ids 触发一致）；service-completion / return-completion 等无重规划需求的
        物理事件不触发 oracle 决策，保证 O0 与未来 M1 的决策频率一致。
        """
        if self.oracle_hook is None:
            return
        replan_ids = {v.vehicle_id for v in vehicles
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if not replan_ids:
            return
        self.oracle_hook(self, inst_idx, float(clock), int(self.event_id), int(reveal_idx),
                         vehicles, traces, served_mask, all_customers)

    def run_resumed(self, snapshot):
        """P0-S：从 recourse snapshot 恢复并继续运行到终止。返回 (traces, served_mask)。

        快照代表「决策点（_plan_and_commit 之前）」，因此恢复 = restore 状态 →
        重新执行本决策点的 plan_and_commit → 继续事件循环。与原 t=0 完整 rollout
        在相同 continuation 下后缀 trace/cost exact parity（Gate S）。
        """
        from recourse_snapshot import restore_recourse_snapshot, validate_snapshot_against_env
        validate_snapshot_against_env(self, snapshot)
        inst_idx = int(snapshot['instance_id'])
        vehicles, traces, served_mask = restore_recourse_snapshot(snapshot)
        if self.replanner is not None and getattr(self.replanner, 'restore_state', None) is not None:
            self.replanner.restore_state(snapshot.get('replanner_state'))
        clock = float(snapshot['clock'])
        reveal_idx = int(snapshot['reveal_idx'])
        self.event_id = int(snapshot['event_id'])
        self.replan_count = int(snapshot['replan_count'])
        self.force_suffix = {tuple(k): list(v) for k, v in snapshot['force_suffix']}

        all_customers = [int(c) for c in snapshot['customer_universe']]
        reveal_events = sorted(set(float(self.reveal_time[inst_idx, i])
                                     for i in all_customers
                                     if self.reveal_time[inst_idx, i] > 0))
        horizon = float(self.tw_end[inst_idx].max())
        try:
            self._maybe_snapshot(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                 all_customers)
            self._run_oracle_hook(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                  all_customers)
            self._plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers)
            return self._run_loop(inst_idx, clock, reveal_idx, vehicles, traces, served_mask,
                                  all_customers, reveal_events, horizon)
        finally:
            self.force_suffix = {}

    def prepare_decision_point(self, clock, vehicles):
        """决策点准备（P0-A）：idle 车第一次派车 ready_time=clock；ready 车 WAIT 后也把
        ready_time 抬到 clock（消除时间穿越：depart 不得早于当前事件时刻）。

        抽成公开方法，供 _plan_and_commit 与 counterfactual teacher 的 incumbent 计算
        共用，保证「no-op 候选 == 原 baseline」的锚点时间一致。
        """
        for v in vehicles:
            if v.status == 'idle':
                v.current_node = 0
                v.current_load = 0.0
                v.ready_time = float(clock)
            elif v.status == 'ready':
                v.ready_time = max(v.ready_time, float(clock))

    def _plan_and_commit(self, inst_idx, clock, vehicles, traces, served_mask, all_customers):
        self.prepare_decision_point(clock, vehicles)

        visible_ids = [i for i in all_customers
                       if self.reveal_time[inst_idx, i] <= clock + 1e-6]

        # P0-CTRL：只对 needs_replan 的 idle/ready 车调用 replanner；其余保留旧 plan
        replan_ids = {v.vehicle_id for v in vehicles
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if self.replanner is not None and replan_ids:
            self.replan_count += 1
            self.replanner.plan(self, inst_idx, clock, vehicles, served_mask,
                                visible_ids, replan_ids=replan_ids)

        # R1.7-4：force_suffix（branch intervention）—— 覆盖 replanner 输出的完整 suffix。
        # 只对 (event_id, vehicle_id) 命中的车生效，绕过 replanner 的 WAIT 逻辑（[0]=CLOSE 强制执行）。
        if self.force_suffix:
            for v in vehicles:
                key = (self.event_id, v.vehicle_id)
                if key in self.force_suffix:
                    v.mutable_suffix = list(self.force_suffix[key])
            # 同步 deferred：force 插入/重新分配的客户不再是 deferred（阶段 C 步骤 3）
            if self.replanner is not None and hasattr(self.replanner, 'sync_deferred_from_vehicles'):
                self.replanner.sync_deferred_from_vehicles(vehicles)

        # commit：对所有 idle/ready 车 pop mutable_suffix 第一个节点
        #（needs_replan 车用新 plan，非 needs_replan 车延续旧 plan）
        for v in vehicles:
            if v.status in ('idle', 'ready'):
                if v.mutable_suffix:
                    nxt = int(v.mutable_suffix[0])
                    v.mutable_suffix = v.mutable_suffix[1:]
                    self._commit(inst_idx, clock, v, traces, nxt)
                v.needs_replan = False  # 已处理，清除标记（committed 车保留）

    def _force_return_all(self, inst_idx, clock, vehicles, traces, served_mask):
        for v in vehicles:
            if v.status == 'ready' and v.current_node != 0:
                self._commit(inst_idx, clock, v, traces, 0)
                if v.status == 'returning':
                    finish = float(v.return_finish)
                    self._advance_fleet(inst_idx, finish, [v], traces, served_mask)


class GreedyReplanner(Replanner):
    """EDD / NN 贪心 replanner（顺序拍卖：车 0 先贪，车 1 贪剩下）。

    incumbent_builder='edd'：EDD（tw_end + 0.01*dist）。
    incumbent_builder='nn'：nearest-neighbor（dist）。
    支持 WAIT（P0-C）：ready@customer 车无可行客户且未来有 reveal 时返回 [] 等待。
    """

    def __init__(self, incumbent_builder='edd'):
        self.incumbent_builder = incumbent_builder

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved = env.get_reserved_customers(vehicles)  # P0-B
        unserved = (set(int(i) for i in visible_ids)
                    - set(np.where(served_mask)[0]) - reserved)
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        allocated = set()
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue  # P0-CTRL：非 needs_replan 车保留旧 plan
            remaining = [i for i in unserved if i not in allocated]
            suffix = self._greedy_suffix(env, inst_idx, v, remaining)
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []  # WAIT：有未来 reveal，暂不返回 depot
            else:
                v.mutable_suffix = suffix
            for o in suffix:
                if o != 0:
                    allocated.add(int(o))

    def _greedy_suffix(self, env, inst_idx, v, remaining):
        route = []
        current = v.current_node
        cur_time = v.ready_time
        load = float(v.current_load)
        unvisited = set(int(i) for i in remaining)
        while unvisited:
            best, best_score = None, float('inf')
            for j in unvisited:
                d = env.dist_mat[inst_idx, current, j]
                arr = cur_time + d / env.tw_speed
                arr = max(arr, env.tw_start[inst_idx, j])
                finish = arr + env.service_time[inst_idx, j]
                ret = finish + env.dist_mat[inst_idx, j, 0] / env.tw_speed
                # P0-B0：return-depot TW 检查（与 OR-fixed 的精确 TSPTW 对齐）。
                # 旧版漏检 ret <= tw_end[0]，导致贪心能「晚于 return horizon 回 depot」，
                # 与 OR-fixed 的 feasibility 语义不一致（B0 的 G_seq 负值部分来源）。
                if (arr <= env.tw_end[inst_idx, j] + 1e-6
                        and ret <= env.tw_end[inst_idx, 0] + 1e-6
                        and load + env.demands[inst_idx, j] <= env.capacity):
                    score = d if self.incumbent_builder == 'nn' \
                        else env.tw_end[inst_idx, j] + 0.01 * d
                    if score < best_score:
                        best_score, best = score, j
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            load += env.demands[inst_idx, best]
            cur_time = max(
                cur_time + env.dist_mat[inst_idx, current, best] / env.tw_speed,
                env.tw_start[inst_idx, best]) + env.service_time[inst_idx, best]
            current = best
        route.append(0)
        return route
