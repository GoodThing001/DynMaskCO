"""JF1-H-F：JF1-H + 完整计划层 repair（hard-feasible baseline，阶段 A）。

不改历史 JointAssignmentReplanner。流程：
  1. 原始 JF1-H joint assignment（min-travel + capacity-only 分配 pool 到各车）；
  2. 每车用 _greedy_sequence_report 排顺序并收集 dropped customer；
  3. 对每个 missing 客户，只在「本次可变计划的车」（replan_ids 的 idle/ready，含 reserved slack）
     内枚举插入位置（anchored + NEW_ROUTE），做 routing certificate，选最小增量 distance；
  4. 写回范围与枚举范围一致（只写 mutable_ids），并做完整车队 ownership partition 校验；
  5. 无合法插入 → 客户显式加入 deferred（延期），不静默消失。

显式延期语义（阶段 A Gate）：deferred 客户保留在 deferred_customers，仍是 unserved，
后续 reveal 重新进入 pool；被服务或重新分配后从 deferred 移除；终局 deferred 里仍未服务的
客户 = terminal_unresolved（audit 检测，不得静默）。

routing certificate 只检查 TW/capacity/return + 温区类型（routing 层面）；冷链 hard Gate
由 StrictOnlineEnv + coldchain_state 的整轨迹 evaluator 承担（短期口径）。
"""
from strict_online_env import Replanner
from joint_fleet import (JointAssignmentReplanner, get_fleet_anchors, get_global_pool,
                         get_vehicle_anchor_node)
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans,
                             apply_action, validate_ownership)


def make_continuation(slack_vehicles=1):
    """构造一个全新的 JF1-H-F continuation（决策/审计状态独立）。

    每条 rollout 分支（baseline / sequential / local / 每个候选）都要用独立实例，
    避免 deferred 集合与累计审计状态跨分支串扰（阶段 C 步骤 1）。
    """
    return JF1HRepairReplanner(slack_vehicles=slack_vehicles)


class JF1HRepairReplanner(JointAssignmentReplanner):
    """JF1-H + 完整计划层 repair。

    审计字段（per-instance，跨事件累计）：
      repair_status ∈ {'clean','repaired','partially_deferred','deferred','ownership_error'}
                    （最近一次 plan 的 repair 结果）
      deferred_customers：repair 中无合法插入、显式延期的客户集合
      repair_stats：attempts/successes/failures/reasons/deferred_then_reassigned/
                    deferred_then_served/ownership_violations/repair_distance_increment
      repair_events：逐事件 repair 记录（clock/missing/repaired/deferred/ownership_ok）
    """

    def __init__(self, slack_vehicles=1):
        super().__init__(score_mode='heuristic')
        # slack_vehicles：预留 K 辆 idle 车不参与 joint assignment，留给 late-reveal
        # tight-window 客户的 repair NEW_ROUTE。修复 greedy composition（t=0 把 25 辆车
        # 全派出去，late reveal 无车可用导致硬 fail）。默认 1：dev_cal 9×128 上 hard_fail
        # 3→0，无回归。
        self.slack_vehicles = int(slack_vehicles)
        self.repair_status = 'clean'
        self._status_inst = -1
        self.deferred_customers = set()
        self.repair_events = []
        self.repair_stats = {
            'attempts': 0, 'successes': 0, 'failures': 0,
            'reasons': {},
            'deferred_then_reassigned': 0,
            'deferred_then_served': 0,
            'ownership_violations': 0,
            'repair_distance_increment': 0.0,
        }

    def _reset_if_new_inst(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = int(inst_idx)
            self.repair_status = 'clean'
            self.deferred_customers = set()
            self.repair_events = []
            self.repair_stats = {'attempts': 0, 'successes': 0, 'failures': 0,
                                 'reasons': {}, 'deferred_then_reassigned': 0,
                                 'deferred_then_served': 0,
                                 'ownership_violations': 0, 'repair_distance_increment': 0.0}

    def export_state(self):
        """决策状态：deferred 集合 + 已跟踪实例 id（随 snapshot 恢复，供 candidate 隔离）。"""
        return {
            'version': 'jf1h-f-state-v1',
            'baseline': 'JF1-H-F',
            'slack_vehicles': self.slack_vehicles,
            'deferred_customers': sorted(int(c) for c in self.deferred_customers),
            'status_inst': self._status_inst,
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("JF1-H-F 新协议要求 snapshot 含 replanner_state dict；"
                             "legacy 快照缺失状态，不得沿用当前对象")
        if state.get('version') != 'jf1h-f-state-v1':
            raise ValueError(f"replanner_state 版本不符：got {state.get('version')!r}")
        if state.get('baseline') != 'JF1-H-F':
            raise ValueError(f"replanner_state baseline 不符：got {state.get('baseline')!r}")
        if state.get('slack_vehicles') != self.slack_vehicles:
            raise ValueError(f"replanner_state slack_vehicles 不符："
                             f"got {state.get('slack_vehicles')!r} != {self.slack_vehicles}")
        if 'deferred_customers' not in state:
            raise ValueError("replanner_state 缺 deferred_customers")
        if 'status_inst' not in state:
            raise ValueError("replanner_state 缺 status_inst")
        deferred = state['deferred_customers']
        if not isinstance(deferred, (list, tuple, set)):
            raise ValueError(f"deferred_customers 类型非法：{type(deferred)}")
        self.deferred_customers = set(int(c) for c in deferred)
        self._status_inst = int(state['status_inst'])
        # 重置分支审计状态（每条 rollout 分支独立，不跨分支累计）
        self.repair_status = 'clean'
        self.repair_events = []
        self.repair_stats = {'attempts': 0, 'successes': 0, 'failures': 0,
                             'reasons': {}, 'deferred_then_reassigned': 0,
                             'deferred_then_served': 0,
                             'ownership_violations': 0, 'repair_distance_increment': 0.0}

    def sync_deferred_from_vehicles(self, vehicles):
        """把已 committed 或已在 suffix 的客户从 deferred 移除（force 应用后、commit 前同步）。"""
        owned = set()
        for v in vehicles:
            if v.status == 'committed' and v.committed_next not in (None, 0):
                owned.add(int(v.committed_next))
            for n in v.mutable_suffix:
                if int(n) != 0:
                    owned.add(int(n))
        self.deferred_customers.difference_update(owned)

    def _anchor_load(self, env, inst_idx, a):
        ld = a.current_load
        if a.status == 'committed' and a.committed_next not in (None, 0):
            ld += float(env.demands[inst_idx, a.committed_next])
        return ld

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self._reset_if_new_inst(inst_idx)

        # 清理 deferred：已被服务 / 已被分配（committed 或 mutable tail）的客户不再是 deferred。
        # 覆盖「deferred 客户经正常 joint assignment 重新分配并 commit」而不触发 repair 的路径。
        owned = set()
        for v in vehicles:
            if v.status == 'committed' and v.committed_next not in (None, 0):
                owned.add(int(v.committed_next))
            for n in v.mutable_suffix:
                if int(n) != 0:
                    owned.add(int(n))
        for c in list(self.deferred_customers):
            if served_mask[int(c)]:
                self.deferred_customers.discard(c)
                self.repair_stats['deferred_then_served'] += 1
            elif int(c) in owned:
                self.deferred_customers.discard(c)
                self.repair_stats['deferred_then_reassigned'] += 1

        anchors = get_fleet_anchors(vehicles)
        pool, _locked = get_global_pool(vehicles, served_mask, visible_ids, replan_ids)
        # 预留 slack：把最高 vehicle_id 的 idle 车从 active 剔除，保持 idle 供 repair NEW_ROUTE
        reserved_ids = set()
        if self.slack_vehicles > 0:
            idle_sorted = sorted((a.vehicle_id for a in anchors if a.status == 'idle'),
                                 reverse=True)
            reserved_ids = set(idle_sorted[:self.slack_vehicles])
        active_ids = {a.vehicle_id for a in anchors
                      if a.status in ('idle', 'ready')
                      and a.vehicle_id not in reserved_ids
                      and (replan_ids is None or a.vehicle_id in replan_ids)}
        # mutable_ids：repair 写回范围 = 本次可变计划的车（idle/ready 且参与 replan）。
        # 含 reserved slack 车（其通过 NEW_ROUTE 参与 repair）；committed / 非 replan 车冻结。
        mutable_ids = {v.vehicle_id for v in vehicles
                       if v.status in ('idle', 'ready')
                       and (replan_ids is None or v.vehicle_id in replan_ids)}

        # 1. joint assignment（min-travel + capacity-only，与 JF1-H 一致）
        assignment = {vid: [] for vid in active_ids}
        assigned_load = {vid: 0.0 for vid in active_ids}
        unassigned = []  # joint assignment 容量不够、无法分配的客户（显式进入 missing，不静默消失）
        for j in pool:
            best_v, best_s = None, -float('inf')
            dj = float(env.demands[inst_idx, j])
            for a in anchors:
                if a.vehicle_id not in active_ids:
                    continue
                if (self._anchor_load(env, inst_idx, a)
                        + assigned_load[a.vehicle_id] + dj > env.capacity + 1e-6):
                    continue
                anchor_node = get_vehicle_anchor_node(a)
                s = -float(env.dist_mat[inst_idx, anchor_node, j])
                if s > best_s:
                    best_s, best_v = s, a.vehicle_id
            if best_v is not None:
                assignment[best_v].append(j)
                assigned_load[best_v] += dj
            else:
                unassigned.append(j)

        # 2. 每车 greedy sequence（report 版，收集 dropped）
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        missing = list(unassigned)  # 容量不够的 unassigned + greedy dropped
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = assignment.get(v.vehicle_id, [])
            suffix, dropped = self._greedy_sequence_report(env, inst_idx, v, assigned)
            missing.extend(dropped)
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix

        # 3. repair missing（完整计划层，只写回 mutable_ids 车）
        if missing:
            status = self._repair_missing(env, inst_idx, clock, vehicles, served_mask,
                                          missing, has_future, mutable_ids)
        else:
            status = 'clean'
        self.repair_status = status

    def _can_defer_to_committed(self, env, inst_idx, vehicles, customer):
        """是否存在「时间/容量/返仓上潜在可服务」的 committed 车（能则 defer，不消耗 reserved slack）。

        这只表示「未来可能可服务」，不是对未来服务的预留/承诺；deferred customer 仍留在池中，
        后续可由任意可用车辆服务。三个条件：
          - committed 完成 + committed_next→customer 直达 <= tw_end；
          - committed 完成时载重 + customer demand <= capacity；
          - customer 服务完成 + customer→depot <= depot horizon。
        """
        tw_end = float(env.tw_end[inst_idx, customer])
        depot_horizon = float(env.tw_end[inst_idx, 0])
        demand = float(env.demands[inst_idx, customer])
        service = float(env.service_time[inst_idx, customer])
        for v in vehicles:
            if v.status == 'committed' and v.committed_next not in (None, 0):
                node = int(v.committed_next)
                arrive = (float(v.committed_finish)
                          + env.dist_mat[inst_idx, node, customer] / env.tw_speed)
                if arrive > tw_end + 1e-6:
                    continue
                # 载重：committed 完成时（含已入舱 committed_next）+ 新客户
                post_load = float(v.current_load) + float(env.demands[inst_idx, node]) + demand
                if post_load > env.capacity + 1e-6:
                    continue
                service_start = max(arrive, float(env.tw_start[inst_idx, customer]))
                return_time = service_start + service + env.dist_mat[inst_idx, customer, 0] / env.tw_speed
                if return_time > depot_horizon + 1e-6:
                    continue
                return True
        return False

    def _repair_missing(self, env, inst_idx, clock, vehicles, served_mask, missing, has_future,
                        mutable_ids):
        """把 missing 客户按最小增量插入（只插入 mutable_ids 车），并验证完整车队 ownership。

        返回 repair_status ∈ {'repaired','partially_deferred','deferred','ownership_error'}。
        显式延期语义：无合法插入的客户加入 deferred_customers（不静默消失）。
        """
        committed = {int(v.committed_next) for v in vehicles
                     if v.status == 'committed' and v.committed_next not in (None, 0)}
        universe = [int(c) for c in range(1, env.num_nodes)
                    if env.demands[inst_idx, c] > 0
                    and not served_mask[int(c)]
                    and env.reveal_time[inst_idx, c] <= clock + 1e-6
                    and int(c) not in committed]

        plans = build_vehicle_plans(env, inst_idx, vehicles)
        # 清理 deferred：已被 committed 或当前 plans.suffix 持有的客户从 deferred 移除，
        # 避免 ownership 检查把它们同时算进两个 category 造成交叉（committed_and_deferred /
        # suffix_and_deferred）。
        owned = set(committed) | {int(x) for p in plans.values() for x in p.suffix}
        for c in list(self.deferred_customers):
            if c in owned:
                self.deferred_customers.discard(c)

        n_repaired = 0
        n_deferred = 0
        repaired_this_event = []
        deferred_this_event = []

        def _defer(customer, reason):
            nonlocal n_deferred
            self.repair_stats['failures'] += 1
            self.repair_stats['reasons'][reason] = \
                self.repair_stats['reasons'].get(reason, 0) + 1
            # 显式延期：customer 保持 unserved，后续 reveal 重新进入 pool
            self.deferred_customers.add(int(customer))
            deferred_this_event.append(int(customer))
            n_deferred += 1

        # 按 tw_end 升序（最紧迫先），确保 reserved slack 优先给最紧迫的客户
        for customer in sorted(missing, key=lambda c: (float(env.tw_end[inst_idx, c]), int(c))):
            self.repair_stats['attempts'] += 1
            cands, _ = enumerate_actions_from_plans(env, inst_idx, plans, customer,
                                                    allowed_vehicle_ids=mutable_ids)
            feasible = [c for c in cands if c.feasible]
            if not feasible:
                _defer(customer, 'no_feasible_insertion')
                continue
            anchored = [c for c in feasible if c.action.slot.kind == 'anchored']
            new_route = [c for c in feasible if c.action.slot.kind == 'new_route']
            if anchored:
                best = min(anchored, key=lambda c: (c.incremental_distance, c.action.action_id()))
            elif new_route and not self._can_defer_to_committed(env, inst_idx, vehicles, customer):
                # 无 anchored 且该客户不能被 committed 车稍后服务 → 用 reserved slack（NEW_ROUTE）
                best = min(new_route, key=lambda c: (c.incremental_distance, c.action.action_id()))
            else:
                # 无 anchored；new_route 可用但该客户能被 committed 车稍后服务 → 延期，
                # 把 reserved slack 留给真正无法等待的客户
                _defer(customer, 'deferred_committed_reachable')
                continue
            plans = apply_action(plans, best.action, allowed_vehicle_ids=mutable_ids)
            # 成功插入即被分配，不再是 deferred（覆盖「前一事件 deferred、本事件 repair 成功」路径）
            self.deferred_customers.discard(int(customer))
            self.repair_stats['successes'] += 1
            self.repair_stats['repair_distance_increment'] += best.incremental_distance
            repaired_this_event.append(int(customer))
            n_repaired += 1

        # 循环后：完整 ownership 校验（所有 missing 已插入或 deferred 后，基于与写回相同的 plans）
        ok = self._ownership_conserved(env, inst_idx, clock, plans, universe, committed,
                                       served_mask)
        if not ok:
            self.repair_stats['ownership_violations'] += 1

        # 写回：只写 mutable_ids 车（与枚举范围一致，避免写回 committed/非 replan 车）
        self._last_repair_plans = plans
        for v in vehicles:
            if v.vehicle_id not in mutable_ids:
                continue
            p = plans.get(v.vehicle_id)
            if p is None:
                continue
            if not p.suffix:
                v.mutable_suffix = [] if (p.anchor_node != 0 and has_future) else [0]
            else:
                v.mutable_suffix = list(p.suffix) + [0]

        self.repair_events.append({
            'clock': float(clock), 'missing': list(missing),
            'repaired': repaired_this_event, 'deferred': deferred_this_event,
            'ownership_ok': ok,
        })

        if not ok:
            return 'ownership_error'
        if n_deferred == 0:
            return 'repaired'
        if n_repaired == 0:
            return 'deferred'
        return 'partially_deferred'

    def _ownership_conserved(self, env, inst_idx, clock, plans, universe, committed, served_mask):
        """完整所有权 partition 校验（阶段 A 阻塞点 3）。

        对每个 ownership category 计数，强制：
          - 每个 expected（已揭示未服务非 committed）客户恰好一次；
          - 非 expected（served / future / extra）客户计数为 0；
          - category 之间无交叉（committed / suffix / deferred）。
        任一 violation 非空 → ownership failure。
        """
        from collections import Counter
        committed_counts = Counter(int(c) for c in committed)
        committed_set = set(committed_counts)
        deferred_set = set(int(c) for c in self.deferred_customers)
        suffix_counts = Counter()
        for p in plans.values():
            for x in p.suffix:
                suffix_counts[int(x)] += 1
        suffix_set = set(suffix_counts)

        expected = set(int(c) for c in universe)

        duplicate_committed = sorted(c for c, k in committed_counts.items() if k > 1)
        committed_in_suffix = sorted(committed_set & suffix_set)
        committed_and_deferred = sorted(committed_set & deferred_set)
        suffix_and_deferred = sorted(suffix_set & deferred_set)
        served_but_planned = sorted(c for c in (suffix_set | deferred_set)
                                    if served_mask[int(c)])
        future_but_planned = sorted(c for c in (suffix_set | deferred_set)
                                    if env.reveal_time[inst_idx, c] > clock + 1e-6)
        extra = sorted(c for c in (suffix_set | deferred_set)
                       if c <= 0 or c >= env.num_nodes or env.demands[inst_idx, c] <= 1e-9)
        duplicate_suffix = sorted(c for c, k in suffix_counts.items() if k > 1)
        planned = suffix_set | deferred_set
        missing = sorted(c for c in expected if c not in planned)

        self._last_dup = duplicate_suffix
        self._last_miss = missing
        self._last_extra = extra
        self._last_served_planned = served_but_planned
        self._last_future_planned = future_but_planned
        self._last_committed_overlap = committed_in_suffix + committed_and_deferred
        self._last_duplicate_committed = duplicate_committed

        return (not duplicate_committed and not committed_in_suffix
                and not committed_and_deferred
                and not suffix_and_deferred and not served_but_planned
                and not future_but_planned and not extra
                and not duplicate_suffix and not missing)
