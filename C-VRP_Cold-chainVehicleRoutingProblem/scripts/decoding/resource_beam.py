"""
Phase 3a: K-Beam Resource-State Feasible Decoder

替代 C++ greedy insertion 的 K-beam 搜索。
每条 beam 维护**实际**到达时间和载重 → 边可行性检查基于真实累积状态。
K 条并行 beam → 探索多样化的可行路线空间 → 避免 local dead-end。

核心循环:
  for each sampling_step:
      for each beam (K total):
          compute feasible next nodes from beam's CURRENT state
          score (beam, next_node) by model logit
      extend top K → beams for next step
      if beam's route full or load high → close route (return to depot)

在 generation 阶段替代 CVRPPartialInsertion.insert()。
在 search 阶段可与 mask-reconstruct 联用。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


def derive_seed(base, *values):
    """稳定的整数 seed 混合（不用 Python hash()，跨进程不保证稳定）。P0-M3。"""
    x = int(base) & 0xFFFFFFFF
    for v in values:
        v = int(v) & 0xFFFFFFFF
        x = (1664525 * (x ^ v) + 1013904223) & 0xFFFFFFFF
    return int(x)


@dataclass
class BeamState:
    """一条 beam = 一个部分构建的路线（可能的完整解）。"""
    route: List[int] = field(default_factory=list)    # 已访问节点序列 (含 depot)
    arrivals: List[float] = field(default_factory=list)  # 每个节点的到达时间
    current_node: int = 0       # 当前末端节点
    current_time: float = 0.0   # 当前到达时间
    current_load: float = 0.0   # 当前累计载重
    visited: set = field(default_factory=set)  # 已访集合
    score: float = 0.0          # 累计模型分数
    closed: bool = False        # 路线是否已关闭（已返回 depot）
    vehicle_count: int = 1      # 用过几辆车
    # 方向1: 热状态
    cabin_temp: float = 25.0       # 车厢温度 (°C)
    cumulative_energy: float = 0.0 # 累计制冷能耗 (kWh)
    quality_ambient: float = 1.0   # 常温货物完好率
    quality_refrig: float = 1.0    # 冷藏货物完好率
    quality_frozen: float = 1.0    # 冷冻货物完好率
    thermal_violations: int = 0    # 温度越界次数
    quality_loss_total: float = 0.0  # 累计品质损耗（weighted completion time，边际贡献累积）
    num_unsalable: int = 0        # 不可售货物数（品质低于可售阈值的货物计数，3 阶段品质模型）
    visited_K_sum: float = 0.0    # 已访问节点的 K 之和（边际贡献计算用）


class ResourceBeamSearcher:
    """
    K-beam 资源状态搜索器。
    在给定的 coords/TW/demands 上构建 feasible 路线。
    每条 beam 独立维护时间+载重+已访状态, 边检查基于实际状态。
    """

    def __init__(self, coords, tw_start, tw_end, service_time, demands,
                 capacity, speed=1.0, K=8, tw_margin=0.0,
                 enable_thermal=False, temp_class=None,
                 enable_quality=False, lambda_q=0.1,
                 quality_salable_threshold=0.1, dist_mat=None,
                 enable_constraint_cutting=False, constraint_mode='fast',
                 constraint_check_top_k=None,
                 visible_mask=None,
                 depot_dispatch_time=0.0):
        """
        Args:
            coords: (N+1, 2) single-instance
            tw_start/tw_end: (N+1,)
            service_time: (N+1,)
            demands: (N+1,)
            capacity: float
            K: beam width
            enable_thermal: 方向1 - 启用温度约束
            temp_class: (N+1,) int, 各节点温度类别 (0=常温/1=冷藏/2=冷冻)
            enable_quality: 品质感知 score（温度驱动路由，冷链多资源核心）
            lambda_q: 品质损耗权重（score -= lambda_q × quality_loss）
            enable_constraint_cutting: Week 1 v6 - 启用约束切割
            constraint_mode: 'fast' (heuristic) or 'ortools' (CP-SAT, not implemented)
            constraint_check_top_k: 只检查得分最高的前 K 个候选（None=检查全部）
            visible_mask: (N+1,) bool，1=可见，0=未来订单。None=clairvoyant（全可见）。
                修复 2026-08-26：严格 non-anticipatory，beam 搜索只服务可见客户。
        """
        self.coords = coords
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_time = service_time
        self.demands = demands
        self.capacity = capacity
        self.speed = speed
        self.K = K
        self.margin = tw_margin
        # 回 depot 后新车辆出发时刻：static=0（多车从 t=0），online=当前 clock（P0-3）
        self.depot_dispatch_time = depot_dispatch_time
        self.N = len(coords)  # nodes + depot

        # P0-M2/P1：内部 shuffle audit（默认空；record_shuffle_audit=True 时由
        # generate_from_state 填入）。unit test 用，正式 VAL128 不开启。
        self.last_shuffle_audit = []

        # 严格 non-anticipatory：future 节点对 search 不可见
        # visible_mask[j]=0 的客户在 beam 搜索中被跳过，只服务已揭示客户
        self.visible_mask = visible_mask  # (N+1,) bool or None=clairvoyant

        # 方向1: 热约束参数（品质感知也需温度追踪 cabin_temp）
        # 品质感知（multi-compartment）不依赖车厢温度追踪，故 enable_quality 不再强制 enable_thermal。
        # 否则 toy 验证时热约束会错误地把冷冻节点（target=-18°C）判为不可行。
        self.enable_thermal = enable_thermal and temp_class is not None
        self.temp_class = temp_class if temp_class is not None else np.zeros(self.N, dtype=int)
        # T_target per class: 0=25, 1=4, 2=-18
        self.T_TARGET = np.array([25.0, 4.0, -18.0], dtype=np.float32)
        self.T_MARGIN = 30.0  # Large margin at start: allow warm cabin to serve frozen after cooling en route

        # 品质感知参数（基于 docs/论文参考/v1 的 03_冷链品质模型）
        # 品质劣化 = 一阶反应（Arrhenius）：Q = 1 - exp(-k(T)t)，k(T) 随车厢温度变化
        # 依据：Aung & Chang 2013（温度是品质最重要因素）+ ML 温度管理 2024（一阶反应）
        self.enable_quality = enable_quality
        self.lambda_q = lambda_q
        # 可售品质阈值：货物剩余品质低于 (1 - threshold) 时视为不可售（3 阶段品质模型的寿命阈值）
        self.quality_salable_threshold = quality_salable_threshold
        self.K_QUALITY = np.array([0.01, 0.002, 0.0002], dtype=np.float32)  # 常温/冷藏/冷冻 基准衰减速率 (1/h)
        # 温度敏感系数：车厢温度每偏离目标 1°C，品质损耗速率变化 exp(α×1) 倍
        # 依据：Arrhenius 方程 k(T) = k_0 exp(-E_a/RT)，温度升高 → 速率指数上升
        self.ALPHA_TEMP = 0.05  # 1/°C，温度敏感系数
        # scale 校准：品质损耗数值 (~0.00002-0.05) 放大到与 edge_logit (~0.06/边) 同量级
        self.QUALITY_SCALE = 100.0

        # 所有客户节点的 K 之和（用于 weighted completion time 的边际贡献计算）
        self.K_TOTAL = sum(
            self.K_QUALITY[min(int(self.temp_class[k]), 2)]
            for k in range(1, self.N)
        )

        # Precompute distance matrix（支持手动传入 dist_mat，用于 controlled toy instance）
        if dist_mat is not None:
            self.dist = np.asarray(dist_mat, dtype=np.float32)
        else:
            diff = coords[:, None, :] - coords[None, :, :]
            self.dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
        self.travel_mat = self.dist / speed  # (N, N)

        # Week 1 v6: Constraint Cutting
        self.enable_constraint_cutting = enable_constraint_cutting
        self.constraint_mode = constraint_mode
        self.constraint_check_top_k = constraint_check_top_k
        self.constraint_checker = None
        if self.enable_constraint_cutting:
            # 使用绝对导入避免相对导入问题
            import sys
            import os
            _constraint_checker_path = os.path.join(os.path.dirname(__file__), 'constraint_checker.py')
            import importlib.util
            spec = importlib.util.spec_from_file_location("constraint_checker", _constraint_checker_path)
            constraint_checker_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(constraint_checker_module)
            ConstraintChecker = constraint_checker_module.ConstraintChecker

            self.constraint_checker = ConstraintChecker(
                coords=coords,
                demands=demands,
                tw_start=tw_start,
                tw_end=tw_end,
                service_time=service_time,
                capacity=capacity,
                speed=speed,
                time_margin=tw_margin
            )

    # ── feasibility checks ──
    def _can_go(self, i: int, j: int, ready_i: float, load_i: float, beam_state=None) -> bool:
        """检查从 i 到 j 的单边可行性（ready_i = 服务完 i 可离开的时刻）。"""
        if i == j:
            return False
        # arrival at j = ready_i + travel（不再加 service[i]，P0-2 修复 service 双重计数）
        arrive_j = ready_i + self.travel_mat[i, j]
        start_j = max(arrive_j, self.tw_start[j])
        if start_j > self.tw_end[j] - self.margin:
            return False
        # capacity
        if j != 0 and load_i + self.demands[j] > self.capacity:
            return False
        # round-trip: after serving j, can return to depot?
        if j != 0:
            ready_j = start_j + self.service_time[j]
            ret = ready_j + self.travel_mat[j, 0]
            if ret > self.tw_end[0] - self.margin:
                return False
        # depot return: can i → 0?
        if j == 0:
            ret = ready_i + self.travel_mat[i, 0]
            if ret > self.tw_end[0] - self.margin:
                return False
        # 方向1: 热约束 — 货物温度不能超过目标温度 + 5°C
        if j != 0 and self.enable_thermal and beam_state is not None:
            t_target = self.T_TARGET[min(int(self.temp_class[j]), 2)]
            if beam_state.cabin_temp > t_target + self.T_MARGIN:
                return False
        return True

    def _apply(self, beam: BeamState, j: int, edge_logit: float) -> BeamState:
        """将节点 j 追加到 beam, 创建新 beam state (不修改原 beam)。"""
        if j == 0:
            # close route: return to depot, dispatch new vehicle at depot_dispatch_time
            arr_0 = beam.current_time + self.travel_mat[beam.current_node, 0]
            new_route = beam.route + [0]
            new_arrivals = beam.arrivals + [arr_0]
            return BeamState(
                route=new_route,
                arrivals=new_arrivals,
                current_node=0, current_time=self.depot_dispatch_time, current_load=0.0,
                visited=beam.visited.copy(),
                score=beam.score + edge_logit, closed=False,
                vehicle_count=beam.vehicle_count + 1,
                # 新车从 depot 出发：热状态重置 + 货物品质重置为新鲜（1.0）
                cabin_temp=25.0, cumulative_energy=beam.cumulative_energy,
                quality_ambient=1.0,
                quality_refrig=1.0,
                quality_frozen=1.0,
                thermal_violations=beam.thermal_violations,
                # 累计量跨车辆累加，不能清零（修复：旧版在此清零导致 Num 恒为 0）
                quality_loss_total=beam.quality_loss_total,
                num_unsalable=beam.num_unsalable,
                visited_K_sum=beam.visited_K_sum,
            )
        else:
            # 品质口径用「行驶 + 服务」时间（不含等待），与数据 cum_time 一致。
            # 等待时间（早到等 tw_start）会让 dt 虚高、品质衰减被放大。
            travel_service_dt = self.service_time[beam.current_node] + self.travel_mat[beam.current_node, j]
            # arrival = ready + travel（不再加 service[current]，P0-2 修复）
            arrive_j = beam.current_time + self.travel_mat[beam.current_node, j]
            start_j = max(arrive_j, self.tw_start[j])
            ready_j = start_j + self.service_time[j]
            load_j = beam.current_load + self.demands[j]
            new_visited = beam.visited | {j}
            # 方向1: update thermal state from beam
            tc = int(self.temp_class[j]) if j < len(self.temp_class) else 0
            t_target = self.T_TARGET[min(tc, 2)]
            dt = start_j - beam.current_time  # travel + waiting (thermal, 不再含 service[current])
            new_temp = beam.cabin_temp
            thermal_v = beam.thermal_violations
            if self.enable_thermal:
                if beam.cabin_temp > t_target + self.T_MARGIN:
                    thermal_v += 1
                # Newton 升温：向环境 25°C 靠拢
                new_temp += (25.0 - beam.cabin_temp) * (1.0 - np.exp(-0.5 * dt))
                # 主动制冷：向目标温度 t_target 靠拢（制冷速率比升温快）
                if new_temp > t_target:
                    new_temp -= (new_temp - t_target) * (1.0 - np.exp(-1.5 * dt))
                # Service at j: door opening → fast warming
                new_temp += (25.0 - new_temp) * (1.0 - np.exp(-2.0 * self.service_time[j]))
            # 品质损耗：温度依赖的 Arrhenius 一阶反应（品质感知 score，冷链多资源核心）
            new_ql = beam.quality_loss_total
            new_q_ambient = beam.quality_ambient
            new_q_refrig = beam.quality_refrig
            new_q_frozen = beam.quality_frozen
            num_unsalable = beam.num_unsalable
            new_visited_K = beam.visited_K_sum
            if self.enable_quality:
                # === 多温区（multi-compartment）品质模型 ===
                # 每温区在各自目标温度下独立衰减，k = K_QUALITY[cls]（常温 0.01 / 冷藏 0.002 / 冷冻 0.0002）。
                decay_ambient = 1.0 - np.exp(-self.K_QUALITY[0] * travel_service_dt)
                decay_refrig = 1.0 - np.exp(-self.K_QUALITY[1] * travel_service_dt)
                decay_frozen = 1.0 - np.exp(-self.K_QUALITY[2] * travel_service_dt)
                new_q_ambient *= (1.0 - decay_ambient)
                new_q_refrig *= (1.0 - decay_refrig)
                new_q_frozen *= (1.0 - decay_frozen)
                # 品质损耗增量 = 边际贡献（归一化）× travel_dt × scale
                # weighted completion time ∑K_i·t_i 的贪心增量：
                #   服务 j 让所有未服务节点都延迟 p_j，其 K 之和 = K_total − 已访问 K 之和。
                #   「高 K 先服务」→ 已访问 K 之和更大 → 后续边际贡献更小 → 全局更优。
                # 归一化修正（2026-08-18）：
                #   旧版绝对值 (K_total - visited_K) × dt 在展开初期 >> edge_logit，
                #   压制了模型的全局路由信息，导致 50-node 品质优先失效。
                #   改为相对值 (K_total - visited_K) / K_total ∈ [0,1]，
                #   让品质项权重不随展开阶段变化，λ_q 能稳定控制品质 vs 距离权衡。
                marginal_fraction = (self.K_TOTAL - beam.visited_K_sum) / self.K_TOTAL
                new_ql += marginal_fraction * travel_service_dt * self.QUALITY_SCALE
                new_visited_K = beam.visited_K_sum + self.K_QUALITY[tc]
                # === 3 阶段品质（新鲜→劣化→不可售）：剩余品质 < 可售阈值 → 不可售 ===
                if tc == 0 and new_q_ambient < (1.0 - self.quality_salable_threshold):
                    num_unsalable += 1
                elif tc == 1 and new_q_refrig < (1.0 - self.quality_salable_threshold):
                    num_unsalable += 1
                elif tc == 2 and new_q_frozen < (1.0 - self.quality_salable_threshold):
                    num_unsalable += 1
            return BeamState(
                route=beam.route + [j],
                arrivals=beam.arrivals + [arrive_j],
                current_node=j, current_time=ready_j, current_load=load_j,
                visited=new_visited,
                score=beam.score + edge_logit, closed=False,
                vehicle_count=beam.vehicle_count,
                # 方向1: propagate thermal state
                cabin_temp=new_temp,
                cumulative_energy=beam.cumulative_energy,
                quality_ambient=new_q_ambient,
                quality_refrig=new_q_refrig,
                quality_frozen=new_q_frozen,
                thermal_violations=thermal_v,
                quality_loss_total=new_ql,
                num_unsalable=num_unsalable,
                visited_K_sum=new_visited_K,
            )

    # ── main generation ──
    def generate(self, edge_logits: np.ndarray, max_steps: int = 100):
        """
        K-beam 生成初始解。

        Args:
            edge_logits: (N, N) float, 模型输出的边 logits (原始, softmax 前)
            max_steps: 最大展开步数

        Returns:
            best_route: list[int] 节点序列, 或 None (无完整可行解)
            best_score: float
        """
        # Start from depot
        beams = [BeamState(route=[0], arrivals=[0.0])]
        # 严格 non-anticipatory：只服务可见客户（future 节点不参与搜索）
        all_unvisited = set(range(1, self.N))
        if self.visible_mask is not None:
            all_unvisited = {j for j in all_unvisited if self.visible_mask[j]}

        for step in range(max_steps):
            # Check: any beam completed?（修复：closed 状态写回 list）
            for idx, b in enumerate(beams):
                if b.visited == all_unvisited and not b.closed:
                    # Close route: return to depot
                    if self._can_go(b.current_node, 0, b.current_time, b.current_load, beam_state=b):
                        b = self._apply(b, 0, 0.0)
                        b.closed = True
                        beams[idx] = b  # 写回 list

            active = [b for b in beams if not b.closed]
            if not active:
                break

            # For each active beam, find feasible next nodes
            candidates = []  # list of (new_beam, edge_logit)
            for b in active:
                i = b.current_node
                arr_i = b.current_time
                load_i = b.current_load

                for j in range(self.N):
                    if self.visible_mask is not None and j != 0 and not self.visible_mask[j]:
                        continue  # 未来节点：严格 non-anticipatory，search 不可见
                    if j in b.visited and j != 0:
                        continue
                    if not self._can_go(i, j, arr_i, load_i, beam_state=b):
                        continue
                    logit = edge_logits[i, j]
                    new_b = self._apply(b, j, logit)
                    # 2-step lookahead: does j still have feasible next options?
                    if j != 0 and not new_b.closed:
                        has_future = False
                        for k in range(self.N):
                            if self.visible_mask is not None and k != 0 and not self.visible_mask[k]:
                                continue  # 未来节点不可见
                            if k in new_b.visited and k != 0:
                                continue
                            if self._can_go(j, k, new_b.current_time, new_b.current_load, beam_state=new_b):
                                has_future = True
                                break
                        # Also: can always return to depot
                        if self._can_go(j, 0, new_b.current_time, new_b.current_load):
                            has_future = True
                        if not has_future:
                            continue  # dead-end, skip
                    candidates.append((new_b, logit))

            # Week 1 v6: Constraint Cutting - prune candidates that cannot be extended to feasible solutions
            if self.enable_constraint_cutting and self.constraint_checker is not None and candidates:
                # Sort candidates by score to apply constraint checking to top-K only (if specified)
                if self.constraint_check_top_k is not None:
                    candidates_sorted = sorted(candidates, key=lambda x: x[0].score, reverse=True)
                    check_candidates = candidates_sorted[:self.constraint_check_top_k]
                    skip_candidates = candidates_sorted[self.constraint_check_top_k:]
                else:
                    check_candidates = candidates
                    skip_candidates = []

                # Check extendability for selected candidates
                filtered_candidates = []
                for new_b, logit in check_candidates:
                    if new_b.closed or len(new_b.visited) == self.N:
                        # Already complete, no need to check
                        filtered_candidates.append((new_b, logit))
                        continue

                    # Prepare state for constraint checker
                    visited_mask = np.zeros(self.N, dtype=bool)
                    for v in new_b.visited:
                        visited_mask[v] = True

                    # Check if this partial state can be extended to feasible solution
                    is_feasible, reason = self.constraint_checker.is_extendable_to_feasible(
                        visited=visited_mask,
                        current_node=new_b.current_node,
                        current_time=new_b.current_time,
                        remaining_capacity=self.capacity - new_b.current_load
                    )

                    if is_feasible:
                        filtered_candidates.append((new_b, logit))
                    # else: prune this branch (not extendable to feasible solution)

                # Merge filtered top-K with unchecked candidates (if any)
                candidates = filtered_candidates + skip_candidates

            if not candidates:
                # No feasible extensions → close existing beams
                for b in active:
                    if self._can_go(b.current_node, 0, b.current_time, b.current_load, beam_state=b):
                        closed_b = self._apply(b, 0, 0.0)
                        closed_b.closed = True
                        candidates.append((closed_b, 0.0))

            if not candidates:
                break

            # Sort by score（品质感知：score -= lambda_q × quality_loss），keep top K
            def _beam_rank(item):
                b = item[0]
                return b.score - self.lambda_q * b.quality_loss_total

            candidates.sort(key=_beam_rank, reverse=True)
            beams = candidates[:self.K]
            beams = [c[0] for c in beams]

        # Select best: prefer complete + high score（品质感知）
        complete = [b for b in beams if b.visited == all_unvisited and b.closed]
        if complete:
            best = max(complete, key=lambda b: b.score - self.lambda_q * b.quality_loss_total)
        else:
            # fallback: beam with most nodes visited
            best = max(beams, key=lambda b: len(b.visited))

        self.last_num_unsalable = best.num_unsalable
        return best.route, best.score

    def _feasible_next_nodes(self, beam, all_unvisited, init_visited, single_route):
        """production 候选边生成：复用当前 generate 的真实可行性条件（P0-M2）。"""
        i = beam.current_node
        ready_i = beam.current_time
        load_i = beam.current_load
        feasible_js = []
        for j in range(self.N):
            # 1. future invisible
            if self.visible_mask is not None and j != 0 and not self.visible_mask[j]:
                continue
            # 2. already visited
            if j in beam.visited and j != 0:
                continue
            # 3. single-route 不能中途回 depot
            if single_route and j == 0 and beam.visited != (all_unvisited | init_visited):
                continue
            # 4. 当前资源状态硬可行性
            if not self._can_go(i, j, ready_i, load_i, beam_state=beam):
                continue
            # 5. 2-step lookahead
            probe = self._apply(beam, j, 0.0)
            if j != 0 and not probe.closed:
                remaining_after = [k for k in all_unvisited if k not in probe.visited]
                if remaining_after:
                    has_future = any(
                        self._can_go(j, k, probe.current_time, probe.current_load, beam_state=probe)
                        for k in remaining_after)
                else:
                    has_future = self._can_go(j, 0, probe.current_time, probe.current_load,
                                              beam_state=probe)
                if not has_future:
                    continue
            feasible_js.append(j)
        return feasible_js

    def _derive_shuffle_seed(self, base_seed, step, route):
        """按 beam state identity 派生 shuffle seed（不是 beam_idx，避免分叉后错位）。"""
        x = derive_seed(base_seed, step, len(route))
        for node in route:
            x = derive_seed(x, int(node))
        return x

    def generate_from_state(self, edge_logits, current_node, current_time, current_load,
                            served_mask, max_steps=100, single_route=False,
                            logit_mode='real', shuffle_seed=None, return_meta=False,
                            record_shuffle_audit=False):
        """
        严格 non-anticipatory 在线重规划：从真实当前车辆状态开始 beam 搜索。

        与 generate() 的区别（修复 2026-08-26，P0-4 完整落地）：
          1. 初始 beam 从当前车辆状态开始（不是 depot/t=0/load=0）
          2. 已服务节点（served_mask）进入 visited，不再服务
          3. 搜索空间 = 可见 & 未服务（visible_mask & ~served）
          4. 找不到完整可行 suffix 时返回 success=False（不再 fallback 到 most-visited）

        logit_mode（P0-M2）：'real'=原 logits；'shuffle'=只对当前 beam state 下真正
          可行候选边的 score 做 permutation（严格保持 multiset 不变）。
        return_meta（P0-M3）：返回 meta dict（initial_actionable_count 等）。

        Args:
            edge_logits: (N, N) 模型输出的边 logits
            current_node: 当前车辆位置
            current_time: 当前 ready_time（服务完可离开）
            current_load: 当前载重
            served_mask: (N,) bool，已服务的节点（含已执行 prefix + 其他车已分配）
            max_steps: 最大展开步数
            single_route: True=单 segment（fleet 语义：不中途回 depot 派新车，容量满就停）

        Returns:
            (suffix, success) 或 (suffix, success, meta)（return_meta=True）
        """
        current_node = int(current_node)
        # 初始 beam 从当前状态开始
        init_route = [current_node]
        init_visited = set(int(v) for v in np.where(served_mask)[0])
        init_visited.add(0)  # depot 已访问
        if current_node != 0:
            init_visited.add(current_node)

        beams = [BeamState(
            route=init_route,
            current_node=current_node,
            current_time=float(current_time),
            current_load=float(current_load),
            visited=init_visited,
        )]

        # 搜索空间 = 可见 & 未服务
        all_unvisited = set(range(1, self.N))
        if self.visible_mask is not None:
            all_unvisited = {j for j in all_unvisited if self.visible_mask[j]}
        all_unvisited = all_unvisited - init_visited

        meta = {'initial_actionable_count': 0, 'beam_steps': 0, 'shuffle_applied_count': 0}
        shuffle_audit = []

        for step in range(max_steps):
            # 检查 completed（修复：closed 状态写回 beams list，原实现只改局部变量）
            for idx, b in enumerate(beams):
                if b.visited == (all_unvisited | init_visited) and not b.closed:
                    if self._can_go(b.current_node, 0, b.current_time, b.current_load, beam_state=b):
                        b = self._apply(b, 0, 0.0)
                        b.closed = True
                        beams[idx] = b  # 写回 list

            active = [b for b in beams if not b.closed]
            if not active:
                break

            meta['beam_steps'] = step + 1

            candidates = []
            for b in active:
                i = b.current_node

                # P0-M2：用 production helper 得到真正 feasible 候选边
                candidate_js = self._feasible_next_nodes(
                    b, all_unvisited, init_visited, single_route)
                if not candidate_js:
                    continue

                if step == 0:
                    meta['initial_actionable_count'] = len(candidate_js)

                raw_scores = np.asarray(
                    [edge_logits[i, j] for j in candidate_js], dtype=np.float32)
                used_scores = raw_scores.copy()

                # H3：只对当前 beam state 下真正可行候选边 shuffle（严格 multiset 不变）
                if logit_mode == 'shuffle' and len(candidate_js) > 1:
                    base = shuffle_seed if shuffle_seed is not None else 0
                    local_seed = self._derive_shuffle_seed(base, step, b.route)
                    rng = np.random.default_rng(local_seed)
                    perm = rng.permutation(len(candidate_js))
                    used_scores = raw_scores[perm]
                    meta['shuffle_applied_count'] += 1
                    if record_shuffle_audit:
                        shuffle_audit.append({
                            'step': step, 'current_node': i, 'route': tuple(b.route),
                            'candidate_js': tuple(candidate_js),
                            'scores_before': tuple(raw_scores.tolist()),
                            'scores_after': tuple(used_scores.tolist()),
                        })

                for j, score in zip(candidate_js, used_scores):
                    new_b = self._apply(b, j, float(score))
                    candidates.append((new_b, float(score)))

            if not candidates:
                for b in active:
                    if self._can_go(b.current_node, 0, b.current_time, b.current_load, beam_state=b):
                        closed_b = self._apply(b, 0, 0.0)
                        closed_b.closed = True
                        candidates.append((closed_b, 0.0))

            if not candidates:
                break

            def _beam_rank(item):
                b = item[0]
                return b.score - self.lambda_q * b.quality_loss_total

            candidates.sort(key=_beam_rank, reverse=True)
            beams = [c[0] for c in candidates[:self.K]]

        self.last_shuffle_audit = shuffle_audit

        # 明确 complete 判断 + success 标志
        target = all_unvisited | init_visited
        complete = [b for b in beams if b.visited == target and b.closed]
        if complete:
            best = max(complete, key=lambda b: b.score - self.lambda_q * b.quality_loss_total)
            self.last_num_unsalable = best.num_unsalable
            if return_meta:
                return best.route, True, meta
            return best.route, True
        if single_route:
            # single_route：返回服务最多订单的 partial beam（不完整），success=False
            # 服务数量相同时，用模型 score 打破平局（Fix：partial tie-break）
            best = max(beams, key=lambda b: (
                len(b.visited),
                b.score - self.lambda_q * b.quality_loss_total,
            ))
            if len(best.visited) > len(init_visited):
                self.last_num_unsalable = best.num_unsalable
                if return_meta:
                    return best.route, False, meta
                return best.route, False
            self.last_num_unsalable = 0
            if return_meta:
                return None, False, meta
            return None, False
        self.last_num_unsalable = 0
        if return_meta:
            return None, False, meta
        return None, False

    # ── reconstruction (search phase) ──
    def reconstruct(self, frozen_edges: List[Tuple[int, int]],
                    edge_logits: np.ndarray, max_steps: int = 50):
        """
        从冻结边出发进行 beam 重构（search phase 用）。

        Args:
            frozen_edges: [(from, to), ...] 保留的边
            edge_logits: (N, N) model logits
            max_steps: max expansion steps

        Returns:
            route: list[int] 完整路线
        """
        # Build initial beam from frozen edges
        frozen_set = set(frozen_edges)
        # Find depot-departing edges
        starts = [(i, j) for (i, j) in frozen_edges if i == 0]
        if not starts:
            starts = [(0, j) for j in range(1, self.N) if (0, j) in frozen_set]

        initial_beams = []
        for (i, j) in starts:
            if self._can_go(0, j, 0.0, 0.0, beam_state=BeamState()):
                b = BeamState(route=[0, j], arrivals=[0.0, 0.0])
                arr_j = max(self.tw_start[j],
                            self.service_time[0] + self.travel_mat[0, j])
                b.arrivals[-1] = arr_j
                b.current_node = j
                b.current_time = arr_j
                b.current_load = self.demands[j]
                b.visited = {0, j}
                initial_beams.append(b)

        if not initial_beams:
            initial_beams = [BeamState(route=[0], arrivals=[0.0])]

        # Follow frozen chain as far as possible
        for b in initial_beams:
            cur = b.current_node
            while cur != 0 and cur > 0:
                nxt = None
                for (fi, fj) in frozen_edges:
                    if fi == cur:
                        nxt = fj
                        break
                if nxt is None or nxt == 0:
                    break
                b = self._apply(b, nxt, 0.0)
                cur = nxt
            # if chain ends without depot return, try to close
            if b.current_node > 0:
                if self._can_go(b.current_node, 0, b.current_time, b.current_load, beam_state=b):
                    b = self._apply(b, 0, 0.0)

        # Continue with beam search for remaining nodes
        beams = initial_beams
        return self.generate(edge_logits, max_steps)[0]
