"""
CVRPTW 解码 / 推理脚本。

基于 decoding/cvrp.py 扩展，核心改动：
1. 加载 CVRPTWModel（5D 输入）替代 CVRPModel
2. 在候选边插入前增加时间窗可行性过滤
3. 2-opt 阶段后增加时间窗约束违反检查

设计原则：尽量复用原始 decoding/cvrp.py 的 mask-and-reconstruct 循环，
通过修改 _encode 和 _decode_step 来适配 CVRPTW。
"""

import sys
import os
# P0-M：从唯一目录合同解析上游与扩展路径，不依赖当前工作目录。
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT, SCRIPTS_ROOT
_MASKCO_ROOT = str(MASKCO_ROOT)
_CVRPTW_SCRIPTS = str(SCRIPTS_ROOT)
sys.path.insert(0, _MASKCO_ROOT)
sys.path.insert(0, os.path.join(_CVRPTW_SCRIPTS, 'models'))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import threading
import typing as tp
import tqdm
from concurrent.futures import ThreadPoolExecutor

# 原始 MaskCO 模块
from helpers.coord_transform import normalize, cdist
from cvrptw_utils import coord_normalize_visible
from training import load_ckpt
from lib import cvrp_two_opt, CVRPPartialInsertion, cvrp_eval_cost
from decoding.utils import (
    convert_heatmap_dtype,
    CoordDynamicAugment as DynamicAugment,
    compute_num_depot,
)

# CVRPTW 扩展模块
_CVRPTW_ROOT = str(EXTENSION_ROOT)
sys.path.insert(0, _CVRPTW_ROOT)
sys.path.insert(0, os.path.join(_CVRPTW_ROOT, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW_ROOT, 'scripts', 'evaluation'))
from CVRPTWModel import (
    CVRPTWModel, CVRPTWModelConfig,
    compute_tw_feasibility_matrix,
)
try:
    from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
except ImportError:
    DynamicColdChainModel = None
    DynamicColdChainModelConfig = None

try:
    from DynamicColdChainModelEdgeState import DynamicColdChainModelEdgeState
except ImportError:
    DynamicColdChainModelEdgeState = None

# Step 4: 尝试加载 C++ 扩展
_CPP = None
try:
    sys.path.insert(0, os.path.join(_CVRPTW_ROOT, 'scripts', 'lib'))
    import cvrptw_ops as _CPP
    print("[CVRPTW] C++ operators loaded")
except ImportError:
    print("[CVRPTW] C++ operators not found, using Python fallback")


def _cpp_repair_edd(sols, coords_batch, tw_start_batch, tw_end_batch,
                     service_time_batch, speed):
    """EDD 修复（C++ 版）。"""
    _r = np.ascontiguousarray(sols.astype(np.int32))
    _CPP.repair_edd(_r,
        np.ascontiguousarray(coords_batch, dtype=np.float32),
        np.ascontiguousarray(tw_start_batch, dtype=np.float32),
        np.ascontiguousarray(tw_end_batch, dtype=np.float32),
        np.ascontiguousarray(service_time_batch, dtype=np.float32),
        speed)
    return np.array(_r)

_cpp_seed_counter = 0

def _cpp_two_opt(sols, dist_mat, coords_batch, tw_start_batch, tw_end_batch,
                 service_time_batch, num_steps, speed):
    """TW-aware 2-opt（C++ 版）。"""
    global _cpp_seed_counter
    _cpp_seed_counter += 1
    _r = np.ascontiguousarray(sols.astype(np.int32))
    _CPP.two_opt(_r,
        np.ascontiguousarray(dist_mat, dtype=np.float32),
        np.ascontiguousarray(coords_batch, dtype=np.float32),
        np.ascontiguousarray(tw_start_batch, dtype=np.float32),
        np.ascontiguousarray(tw_end_batch, dtype=np.float32),
        np.ascontiguousarray(service_time_batch, dtype=np.float32),
        num_steps, 50, speed,
        42 + _cpp_seed_counter * 777)
    return np.array(_r)

def _cpp_quality_two_opt(sols, dist_mat, coords_batch, tw_start_batch, tw_end_batch,
                          service_time_batch, quality_loss_batch, energy_mat_batch,
                          num_steps, speed, lambda_q=0.1, lambda_e=0.01):
    """品质感知 TW 2-opt（C++ 版，P0-4）。"""
    global _cpp_seed_counter
    _cpp_seed_counter += 1
    _r = np.ascontiguousarray(sols.astype(np.int32))
    _CPP.quality_two_opt(_r,
        np.ascontiguousarray(dist_mat, dtype=np.float32),
        np.ascontiguousarray(coords_batch, dtype=np.float32),
        np.ascontiguousarray(tw_start_batch, dtype=np.float32),
        np.ascontiguousarray(tw_end_batch, dtype=np.float32),
        np.ascontiguousarray(service_time_batch, dtype=np.float32),
        np.ascontiguousarray(quality_loss_batch, dtype=np.float32),
        np.ascontiguousarray(energy_mat_batch, dtype=np.float32),
        lambda_q, lambda_e,
        num_steps, 50, speed,
        42 + _cpp_seed_counter * 777)
    return np.array(_r)

# ============================================================
# TW 可行性评估工具（纯 NumPy，用于推理后评估）
# ============================================================

def evaluate_tw_feasibility(
    routes: np.ndarray,         # (batch, route_len) int
    coords: np.ndarray,         # (batch, nodes, 2)
    tw_start: np.ndarray,       # (batch, nodes)
    tw_end: np.ndarray,         # (batch, nodes)
    service_time: np.ndarray,   # (batch, nodes)
    speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    评估一批解的时间窗可行性。

    Returns:
        feasible_mask: (batch,) bool — 整条路径是否 TW 可行
        violation_counts: (batch,) int — 每条路径的 TW 违反次数
    """
    batch_size = routes.shape[0]
    feasible = np.ones(batch_size, dtype=bool)
    violations = np.zeros(batch_size, dtype=np.int32)

    for b in range(batch_size):
        route = routes[b]
        current_time = 0.0
        prev_node = 0

        for pos in range(1, len(route)):
            node = int(route[pos])
            if node == 0:
                # 检查返回 depot 是否超时（修复 2026-08-26：原实现漏查）
                if prev_node != 0:
                    dx = coords[b, prev_node, 0] - coords[b, 0, 0]
                    dy = coords[b, prev_node, 1] - coords[b, 0, 1]
                    travel_back = np.sqrt(dx * dx + dy * dy) / speed
                    return_time = current_time + service_time[b, prev_node] + travel_back
                    if return_time > tw_end[b, 0] + 1e-6:
                        feasible[b] = False
                        violations[b] += 1
                current_time = 0.0
                prev_node = 0
                continue
            if node == prev_node:
                continue

            dx = coords[b, prev_node, 0] - coords[b, node, 0]
            dy = coords[b, prev_node, 1] - coords[b, node, 1]
            travel_time = np.sqrt(dx * dx + dy * dy) / speed

            arrive_time = current_time + service_time[b, prev_node] + travel_time
            arrive_time = max(arrive_time, tw_start[b, node])

            if arrive_time > tw_end[b, node] + 1e-6:
                feasible[b] = False
                violations[b] += 1

            current_time = arrive_time
            prev_node = node

        # 末尾若未返回 depot，补查返回 depot 时间
        if prev_node != 0:
            dx = coords[b, prev_node, 0] - coords[b, 0, 0]
            dy = coords[b, prev_node, 1] - coords[b, 0, 1]
            travel_back = np.sqrt(dx * dx + dy * dy) / speed
            return_time = current_time + service_time[b, prev_node] + travel_back
            if return_time > tw_end[b, 0] + 1e-6:
                feasible[b] = False
                violations[b] += 1

    return feasible, violations


# ============================================================
# Exp-07: TW-preserving 局部搜索
# ============================================================

def tw_preserving_two_opt(
    sols: np.ndarray,              # (batch, route_len) int
    dist_mat: np.ndarray,          # (batch, nodes, nodes)
    demands: np.ndarray,           # (batch, nodes)
    capacity: int,
    penalty: float,
    num_steps: int,
    num_workers: int,
    coords: np.ndarray,            # (batch, nodes, 2)
    tw_start: np.ndarray,          # (batch, nodes)
    tw_end: np.ndarray,            # (batch, nodes)
    service_time: np.ndarray,      # (batch, nodes)
    speed: float = 1.0,
) -> np.ndarray:
    """
    TW-preserving 2-opt 包装器。

    1. 记录 2-opt 前的 TW 违规数
    2. 运行标准 C++ 2-opt（优化距离）
    3. 检查 2-opt 后的 TW 违规数
    4. 对 TW 违规增加的实例，回退到 2-opt 前的解

    这样既能享受 2-opt 的距离优化，又不会因 2-opt 破坏 TW 可行性。
    """
    batch_size = sols.shape[0]

    # 保存原始解 & 评估 TW
    sols_before = sols.copy()
    _, tw_viol_before = evaluate_tw_feasibility(
        sols_before, coords, tw_start, tw_end, service_time, speed
    )

    # 标准 C++ 2-opt
    sols_after = cvrp_two_opt(
        sols, dist_mat, demands, capacity, penalty, num_steps, num_workers
    )

    # 评估 2-opt 后的 TW
    _, tw_viol_after = evaluate_tw_feasibility(
        sols_after, coords, tw_start, tw_end, service_time, speed
    )

    # 逐实例回退：如果 TW 违规增加，保留 2-opt 前的解
    revert_count = 0
    for b in range(batch_size):
        if tw_viol_after[b] > tw_viol_before[b]:
            sols_after[b] = sols_before[b]
            revert_count += 1

    if revert_count > 0:
        print(f'  [TW-2opt] reverted {revert_count}/{batch_size} instances '
              f'(TW viol would have increased)')

    return sols_after


def tw_aware_two_opt_py(
    sols: np.ndarray,              # (batch, route_len) int
    dist_mat: np.ndarray,          # (batch, nodes, nodes)
    demands: np.ndarray,           # (batch, nodes)
    capacity: int,
    penalty: float,
    num_steps: int,
    coords: np.ndarray,            # (batch, nodes, 2)
    tw_start: np.ndarray,          # (batch, nodes)
    tw_end: np.ndarray,            # (batch, nodes)
    service_time: np.ndarray,      # (batch, nodes)
    speed: float = 1.0,
    trials_per_step: int = 20,
) -> np.ndarray:
    """
    Python TW-aware 2-opt（Phase 2 P0）。

    每步随机采样 trials_per_step 个候选边交换，只接受：
    1. TW 违规不增加
    2. 路径总距离减少

    比完整 2-opt 简单但正确的方向——TW 约束优先。
    """
    batch_size, route_len = sols.shape
    rng = np.random.default_rng()

    for b in range(batch_size):
        sol = sols[b].copy()
        coords_b = coords[b]
        tw_s_b, tw_e_b = tw_start[b], tw_end[b]
        st_b = service_time[b]
        dist_b = dist_mat[b]

        # 评估初始状态
        _, best_viol = evaluate_single_tw(
            sol, coords_b, tw_s_b, tw_e_b, st_b, speed
        )
        best_cost = _eval_single_route_cost(
            sol, dist_b, demands[b], capacity, penalty
        )
        if not _check_capacity_feasible(sol, demands[b], capacity):
            continue  # 容量不可行的初始解，2-opt 无法修复，跳过

        for _ in range(num_steps):
            improved = False
            # 随机采样候选交换
            indices = rng.integers(1, route_len - 2, size=(trials_per_step, 2))
            for i, j in indices:
                i, j = int(i), int(j)
                if i >= j:
                    i, j = j, i
                if j - i < 1:
                    continue
                # 跳过涉及 depot 的交换（会改变路线结构）
                if sol[i] == 0 or sol[i + 1] == 0:
                    continue
                if sol[j] == 0 or sol[min(j + 1, route_len - 1)] == 0:
                    continue

                # 2-opt 交换：反转 [i+1, j]
                new_sol = sol.copy()
                new_sol[i + 1:j + 1] = new_sol[j:i:-1]

                # TW 检查
                _, new_viol = evaluate_single_tw(
                    new_sol, coords_b, tw_s_b, tw_e_b, st_b, speed
                )
                if new_viol > best_viol:
                    continue  # TW 变差，拒绝

                # 距离检查
                new_cost = _eval_single_route_cost(
                    new_sol, dist_b, demands[b], capacity, penalty
                )
                if new_cost < best_cost - 1e-6:
                    sol = new_sol
                    best_cost = new_cost
                    best_viol = new_viol
                    improved = True
                    break  # 找到改进，跳出内层循环

            if not improved:
                break  # 没有改进，提前结束

        sols[b] = sol

    return sols


def evaluate_single_tw(
    route: np.ndarray,          # (route_len,) int
    coords: np.ndarray,         # (nodes, 2)
    tw_start: np.ndarray,       # (nodes,)
    tw_end: np.ndarray,         # (nodes,)
    service_time: np.ndarray,   # (nodes,)
    speed: float = 1.0,
) -> tuple[bool, int]:
    """单实例 TW 可行性评估。返回 (feasible, violation_count)。"""
    feasible = True
    violations = 0
    current_time = 0.0
    prev_node = 0

    for pos in range(1, len(route)):
        node = int(route[pos])
        if node == 0:
            current_time = 0.0
            prev_node = 0
            continue
        if node == prev_node:
            continue

        dx = coords[prev_node, 0] - coords[node, 0]
        dy = coords[prev_node, 1] - coords[node, 1]
        travel_time = np.sqrt(dx * dx + dy * dy) / speed
        arrive_time = current_time + service_time[prev_node] + travel_time
        arrive_time = max(arrive_time, tw_start[node])

        if arrive_time > tw_end[node] + 1e-6:
            feasible = False
            violations += 1

        current_time = arrive_time
        prev_node = node

    return feasible, violations


def _check_capacity_feasible(
    route: np.ndarray,   # (route_len,) int
    demands: np.ndarray, # (nodes,)
    capacity: int,
) -> bool:
    """检查单条路线是否容量可行（任意时刻载重不超容量）。

    与 `_eval_single_route_cost` 的容量判定逻辑一致，但只返回 bool，
    用于 tw_aware_2opt_py 的初始可行判断（跳过容量不可行的初始解）。
    """
    current_load = 0
    for pos in range(len(route)):
        node = int(route[pos])
        if node == 0:
            current_load = 0
            continue
        current_load += demands[node]
        if current_load > capacity:
            return False
    return True


def _eval_single_route_cost(
    route: np.ndarray,       # (route_len,) int
    dist_mat: np.ndarray,    # (nodes, nodes)
    demands: np.ndarray,     # (nodes,)
    capacity: int,
    penalty: float,
) -> float:
    """单实例 CVRP 路径成本评估（含容量软惩罚）。

    注意：此函数**仅用于** Python TW-aware 2-opt 的**内部边选择**
    （决定是否接受一次 2-opt 交换）。其容量软惩罚 `penalty × (load-capacity)`
    用于引导搜索远离超载解。

    最终 cost 报告**统一使用 `_eval_distmat_cost`（纯距离）**，任何惩罚
    均不进入报告的 cost 数值——这是跨方法指标一致性的关键约定。
    """
    total_cost = 0.0
    current_load = 0
    prev_node = 0

    for pos in range(len(route)):
        node = int(route[pos])
        if node == 0:
            total_cost += dist_mat[prev_node, 0]
            current_load = 0
            prev_node = 0
            continue
        if node == prev_node:
            continue

        total_cost += dist_mat[prev_node, node]
        current_load += demands[node]
        if current_load > capacity:
            total_cost += penalty * (current_load - capacity) * 100  # 大惩罚
        prev_node = node

    if prev_node != 0:
        total_cost += dist_mat[prev_node, 0]

    return total_cost


# ============================================================
# Phase 2: TW 修复（EDD 重排 + 局部交换）
# ============================================================

def tw_repair_edd(
    sols: np.ndarray,              # (batch, route_len) int
    coords: np.ndarray,            # (batch, nodes, 2)
    tw_start: np.ndarray,          # (batch, nodes)
    tw_end: np.ndarray,            # (batch, nodes)
    service_time: np.ndarray,      # (batch, nodes)
    speed: float = 1.0,
) -> np.ndarray:
    """
    TW 修复：对每条子路径按最早截止时间（EDD）重排节点。

    经典的 OR 启发式——Earliest Due Date first 最小化最大延迟。
    仅对 TW 违规 > 0 的子路径执行重排，保持可行子路径不变。
    """
    batch_size, route_len = sols.shape
    repaired_count = 0

    for b in range(batch_size):
        sol = sols[b].copy()
        coords_b = coords[b]
        tw_s_b, tw_e_b = tw_start[b], tw_end[b]
        st_b = service_time[b]

        # 拆分路径段（depot=0 分隔）
        segments = []
        current_seg = []
        for pos in range(route_len):
            node = int(sol[pos])
            if node == 0:
                if current_seg:
                    segments.append(current_seg)
                    current_seg = []
            elif node != 0:
                current_seg.append((pos, node))
        if current_seg:
            segments.append(current_seg)

        # 评估每条子路径的 TW 违规
        improved = False
        for seg in segments:
            if len(seg) <= 1:
                continue

            nodes = [node for _, node in seg]
            positions = [pos for pos, _ in seg]

            # 检查该子路径 TW 违规
            tw_ok = _check_segment_tw(
                nodes, coords_b, tw_s_b, tw_e_b, st_b, speed
            )
            if tw_ok:
                continue  # 该段已可行，不修改

            # 按 EDD 排序（tw_end 早的优先）
            sorted_nodes = sorted(nodes, key=lambda n: tw_e_b[n])

            # 检查排序后是否改善 TW
            if sorted_nodes == nodes:
                continue

            new_tw_ok = _check_segment_tw(
                sorted_nodes, coords_b, tw_s_b, tw_e_b, st_b, speed
            )
            if not new_tw_ok:
                continue  # EDD 排序后仍不可行

            # 应用排序
            for i, pos in enumerate(positions):
                sol[pos] = sorted_nodes[i]
            improved = True

        if improved:
            sols[b] = sol
            repaired_count += 1

    if repaired_count > 0:
        print(f'  [TW-Repair] EDD reordered {repaired_count}/{batch_size} instances')

    return sols


def _check_segment_tw(
    nodes: list,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service_time: np.ndarray,
    speed: float,
) -> bool:
    """检查子路径是否 TW 可行。"""
    current_time = 0.0
    prev_node = 0  # depot

    for node in nodes:
        dx = coords[prev_node, 0] - coords[node, 0]
        dy = coords[prev_node, 1] - coords[node, 1]
        travel_time = np.sqrt(dx * dx + dy * dy) / speed

        arrive_time = current_time + service_time[prev_node] + travel_time
        arrive_time = max(arrive_time, tw_start[node])

        if arrive_time > tw_end[node] + 1e-6:
            return False

        current_time = arrive_time
        prev_node = node

    # 返回 depot
    dx = coords[prev_node, 0] - coords[0, 0]
    dy = coords[prev_node, 1] - coords[0, 1]
    travel_time = np.sqrt(dx * dx + dy * dy) / speed
    return_time = current_time + service_time[prev_node] + travel_time
    if return_time > tw_end[0] + 1e-6:
        return False

    return True


# ============================================================
# TW Attention Bias（Exp-05：将 TW 兼容性注入 decoder attention）
# ============================================================

def compute_tw_attn_bias(
    coords: np.ndarray,          # (batch, nodes, 2) — 未归一化坐标
    tw_start: np.ndarray,        # (batch, nodes) — 未归一化 TW
    tw_end: np.ndarray,          # (batch, nodes) — 未归一化 TW
    service_time: np.ndarray,    # (batch, nodes)
    speed: float = 1.0,
    penalty: float = 5.0,
) -> np.ndarray:
    """
    计算 TW 兼容性 attention bias 矩阵。

    对每对节点 (i, j):
    - 若 TW 兼容: bias = 0（不惩罚）
    - 若 TW 不兼容: bias = -penalty（软惩罚，降低 attention 权重）

    depot (idx 0) 始终被视为与所有节点兼容。

    Returns:
        bias: (batch, nodes, nodes) float32
    """
    batch_size, num_nodes = coords.shape[:2]

    # 距离矩阵
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
    travel_time = dist / speed

    # 从 i 出发到达 j 的时间: tw_start[i] + service[i] + travel(i,j)
    ready_at_j = tw_start[:, :, None] + service_time[:, :, None] + travel_time
    feasible = ready_at_j <= tw_end[:, None, :]  # (batch, nodes_from, nodes_to)

    # depot 始终兼容
    feasible[:, 0, :] = True
    feasible[:, :, 0] = True

    bias = np.where(feasible, 0.0, -penalty).astype(np.float32)
    return bias


def cvrptw_searching_decode(
    dataset: dict[str, np.ndarray],
    capacity: int,
    penalty: float,
    model: CVRPTWModel,
    sampling_steps: int,
    cycles: int,
    keep_rate: float,
    batch_size: int,
    runs: int,
    two_opt_steps: int,
    disable_gumbel: bool,
    gumbel_scale_factor: float,
    heatmap_dtype: jax.typing.DTypeLike,
    topk: int | None,
    augment_level: int,
    padding_policy: tp.Literal['none', 'auto'],
    threads_over_batches: int | None,
    seed: int,
    # --- CVRPTW 特有参数 ---
    enable_tw_filter: bool = False,   # 是否在插入时启用 TW 过滤
    enable_tw_attn_bias: bool = False,  # Exp-05: TW attention bias
    tw_attn_penalty: float = 5.0,     # TW attention bias 的惩罚系数
    enable_tw_preserving_2opt: bool = False,  # Exp-07: TW-preserving 2-opt
    enable_tw_aware_2opt_py: bool = False,    # Phase 2: Python TW-aware 2-opt
    enable_tw_repair_edd: bool = False,       # Phase 2: EDD TW修复
    tw_speed: float = 1.0,            # 时间单位下的行驶速度
    tw_max: float = 1.0,               # 时间窗归一化分母
    # --- Phase 2: Dynamic-Aware MaskCO (D1-D6) ---
    enable_event_mask: bool = False,       # D1: 事件驱动局部重建
    enable_frozen_prefix: bool = False,    # D2: 冻结前缀
    enable_constraint_decoder: bool = False,  # D3: 解码器原生约束
    enable_adaptive_mask: bool = False,    # D4: 自适应mask
    enable_anytime: bool = False,          # D6: anytime solver
    enable_resource_decoder: bool = False, # Phase 3a: resource-state feasible decoder
    beam_width: int = 16,                  # Phase 3a: K for beam search
    tw_margin: float = 0.05,               # Phase 3a: TW 安全裕度（0=无裕度，cost 更低）
    # Week 1 v6: Constraint Cutting
    enable_constraint_cutting: bool = False,  # Week 1 v6: 启用约束切割
    constraint_mode: str = 'fast',            # Week 1 v6: 约束检查模式 (fast/ortools)
    constraint_check_top_k: int | None = None,  # Week 1 v6: 只检查 top-K 候选 (None=全部)
    # Week 3 v6: In-Cycle Local Search
    in_cycle_2opt: bool = False,              # Week 3 v6: 启用 in-cycle 2-opt
    in_cycle_2opt_frequency: int = 5,         # Week 3 v6: 每 N 个 cycle 执行一次
    in_cycle_2opt_steps: int = 2,             # Week 3 v6: cycle 内 2-opt 步数
    in_cycle_2opt_start: int = 10,            # Week 3 v6: 从第几个 cycle 开始
    # Quality-aware routing
    enable_quality: bool = False,          # 品质感知 score（温度驱动路由，冷链多资源核心）
    lambda_q: float = 0.1,                 # 历史 proxy 权重（非 C0 目标）
    quality_salable_threshold: float = 0.1,  # 历史 proxy 阈值（非 C0 可售判定）
    save_routes: str | None = None,        # 保存最终路线到 .npz（路线分析用，如 --save_routes routes.npz）
    time_budget_ms: int = 200,             # D6: 时间预算
    frozen_prefix_len: int = 0,              # D2: 冻结前缀长度
):
    num_workers = 1
    np.random.seed(seed)

    # --- 数据提取 ---
    coords = dataset['coords']
    unnormalized_demands = dataset['demands']
    tw_start = dataset['tw_start']  # (N, nodes)
    tw_end = dataset['tw_end']      # (N, nodes)
    service_time = dataset.get('service_time',
                               np.zeros_like(tw_start, dtype=np.float32))

    num_instances = coords.shape[0]
    num_nodes = coords.shape[1] - 1  # 不含 depot

    # --- 距离矩阵 (优先使用数据集中的非对称矩阵) ---
    if 'dist_mat' in dataset:
        dist_mat_np = dataset['dist_mat'].astype(np.float32)
        print(f"[CVRPTW] Using precomputed dist_mat from dataset "
              f"{'(asymmetric)' if dataset.get('asymmetric', False) else '(symmetric)'}")
    else:
        dist_mat_np = np.array(jax.jit(cdist)(coords))

    # --- 最优成本 ---
    if 'opt_costs' in dataset.keys():
        opt_costs = dataset['opt_costs']
        mean_opt_cost = opt_costs.mean().item()
    else:
        opt_costs = None
        mean_opt_cost = None

    # --- TW 归一化：与训练时 CVRPTWDataloader 的逻辑一致 ---
    if tw_max is None:
        tw_max = float(tw_end.max())
    print(f'[CVRPTW] tw_max = {tw_max:.2f} (auto-detected from data)')

    # --- 特征: [x, y, demand, tw_start, tw_end] + 可选 temp_class ---
    feature_arrays = [
        coords,
        unnormalized_demands[..., None] / capacity,
        tw_start[..., None] / tw_max,
        tw_end[..., None] / tw_max,
    ]
    if 'temp_class' in dataset:
        temp_class = dataset['temp_class'].astype(np.float32)
        feature_arrays.append(temp_class[..., None] / 2.0)  # normalize [0,2]→[0,1]
    if 'reveal_time' in dataset:
        reveal_time = dataset['reveal_time'].astype(np.float32)
        feature_arrays.append(reveal_time[..., None] / tw_max)
    if 'quality_loss' in dataset:
        feature_arrays.append(dataset['quality_loss'].astype(np.float32)[..., None])
    # auto-detected via model.init_proj shape check below
    raw_features = np.concatenate(feature_arrays, axis=-1).astype(np.float32)
    # 安全截断：若特征维度>8，取前8维（兼容旧7D模型）
    if raw_features.shape[-1] > 8:
        print(f"[CVRPTW] Truncating features: {raw_features.shape[-1]}D → 8D")
        raw_features = raw_features[..., :8]

    # --- 可见性掩码：将未来订单特征置零 (P0-2 修复) ---
    visible_mask_ds = None
    if 'visible_mask' in dataset:
        visible_mask_ds = dataset['visible_mask'].astype(np.float32)
        vis = visible_mask_ds[..., None]  # (N, nodes, 1)
        # 不可见节点的非坐标特征置零，坐标设为 depot (0.5, 0.5)
        raw_features[..., 2:] = raw_features[..., 2:] * vis
        raw_features[..., :2] = raw_features[..., :2] * vis + (1.0 - vis) * 0.5

    # --- Padding policy ---
    if padding_policy == 'none':
        expected_padded_length = None
    elif padding_policy == 'auto':
        expected_padded_length = compute_num_depot(
            total_demand=unnormalized_demands.sum(axis=-1).max().item(),
            capacity=capacity,
        ) + coords.shape[1] - 1
    else:
        raise ValueError()

    # --- dist_mat 成本计算 (修复 C++ cost=0) ---
    def _eval_distmat_cost(sols, dm):
        """最终 cost 报告：纯行驶距离，无任何惩罚。

        与 `_eval_single_route_cost`（含容量软惩罚，仅 2-opt 内部边选择用）
        不同，此函数是**所有解码路径的统一最终口径**——可行解只比纯距离，
        不可行性通过独立的 TW Feas / TW viol 指标报告，不混入 cost。
        dm: (batch, nodes, nodes)

        修复（2026-08-26）：包含每辆车返回 depot 的距离（原实现漏算）。
        """
        batch_cost = np.zeros(sols.shape[0])
        for b in range(sols.shape[0]):
            prev, c = 0, 0.0
            for pos in range(sols.shape[1]):
                node = int(sols[b, pos])
                if node == 0:
                    c += dm[b, prev, 0]  # 返回 depot 的距离
                    prev = 0
                    continue
                c += dm[b, prev, node]
                prev = node
            if prev != 0:
                c += dm[b, prev, 0]  # 末尾补返回 depot
            batch_cost[b] = c
        return batch_cost

    # --- 预计算时间窗可行性矩阵（用于候选边过滤） ---
    tw_feasible = None
    if enable_tw_filter:
        tw_feasible = np.array(compute_tw_feasibility_matrix(
            coords, tw_start, tw_end, service_time, speed=tw_speed
        ))
        tw_feasible[:, 0, :] = True   # depot → any is always feasible
        tw_feasible[:, :, 0] = True   # any → depot is always feasible

    # --- 预计算 TW attention bias（Exp-05：注入 decoder attention） ---
    tw_attn_bias = None
    if enable_tw_attn_bias:
        tw_attn_bias = compute_tw_attn_bias(
            coords, tw_start, tw_end, service_time,
            speed=tw_speed, penalty=tw_attn_penalty,
        )
        print(f'[CVRPTW] TW attention bias enabled (penalty={tw_attn_penalty})')

    # --- 历史 P0-4 品质/能耗代理输入（仅复现旧 checkpoint） ---
    # Historical model/beam feature proxies. These are not C0 trace metrics.
    quality_loss_ds = None
    energy_mat_ds = None
    if 'quality_loss' in dataset:
        quality_loss_ds = dataset['quality_loss'].astype(np.float32)
    if 'energy_mat' in dataset:
        energy_mat_ds = dataset['energy_mat'].astype(np.float32)

    # --- JIT 编译 ---
    _MODEL_IN = int(model.init_proj.kernel.shape[0])  # JIT外计算, 捕获为闭包

    def _encode(raw_features: jax.Array, visible_mask: jax.Array | None = None,
                edge_feat: jax.Array | None = None):
        raw_features = raw_features.at[..., :2].set(
            coord_normalize_visible(raw_features[..., :2], visible_mask)
        )
        if hasattr(model, 'type_embed'):
            features = model.encode(raw_features[..., :_MODEL_IN],
                                    visible_mask=visible_mask, edge_feat=edge_feat)
        else:
            features = model.encode(raw_features[..., :_MODEL_IN])
        return features

    def _decode_step(
        features: jax.Array,
        neighbors: jax.Array,
        gumbel_key: jax.Array | None = None,
        tw_bias: jax.Array | None = None,  # Exp-05: (batch, nodes, nodes)
        # === D3: dynamic TW constraint ===
        tw_data: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
        # === Phase 3a: resource-state feasible decoder ===
        resource_mask: jax.Array | None = None,  # (B, N+1, N+1) bool, from insertion
    ):
        """tw_data = (coords, tw_start, tw_end, service_time) for runtime TW check."""
        @jax.vmap
        def neighbors2adj(neighbors: jax.Array):
            neighbors = jnp.where(
                neighbors < 0,
                num_nodes * 2,
                neighbors,
            )
            adjmat = jnp.zeros([num_nodes + 1, num_nodes + 1], dtype=jnp.int8)
            _arange = jnp.arange(neighbors.shape[0])
            adjmat = adjmat.at[_arange[1:], neighbors[1:, 0]].set(1, mode='drop')
            adjmat = adjmat.at[_arange[1:], neighbors[1:, 1]].add(1, mode='drop')
            adjmat = adjmat.at[0, :].set(adjmat[:, 0])
            return adjmat

        adjmat = neighbors2adj(neighbors)

        timestep = (neighbors >= 0).sum(axis=[-1, -2]) / (2 * num_nodes)

        def denoised_fn(adjmat: jax.Array, timestep: jax.Array):
            if tw_bias is not None:
                adjmat = adjmat.astype(jnp.float32) + tw_bias
            logits = model.decode(features, timestep, adjmat.astype(jnp.float32))
            if resource_mask is not None:
                logits = jnp.where(resource_mask, logits, -1e9)
            if tw_data is not None and enable_constraint_decoder:
                coords_d3, tw_s, tw_e, svc_t = tw_data
                B, N1 = coords_d3.shape[0], coords_d3.shape[1]
                coords_i = coords_d3[..., :2]
                diff = coords_i[:, :, None, :] - coords_i[:, None, :, :]
                dist_d3 = jnp.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
                travel_d3 = dist_d3 / jnp.maximum(tw_speed, 0.01)
                arrival_lb = tw_s[:, :, None] + svc_t[:, :, None] + travel_d3
                feas_dyn = arrival_lb <= tw_e[:, None, :] + 1e-6
                feas_dyn = feas_dyn.at[:, :, 0].set(True)
                feas_dyn = feas_dyn.at[:, 0, :].set(True)
                logits = jnp.where(feas_dyn, logits, -1e9)
            if gumbel_key is not None:
                logits = logits + jax.random.gumbel(
                    gumbel_key, logits.shape, logits.dtype
                ) * gumbel_scale_factor
            adjmat = jax.nn.softmax(logits, axis=-1) * 2
            return adjmat

        adjmat_pred = denoised_fn(adjmat, timestep)
        adjmat_pred = jnp.where(adjmat, 0., adjmat_pred)
        adjmat_pred = adjmat_pred.at[:, 0].set(0)
        adjmat_pred = convert_heatmap_dtype(adjmat_pred, dtype=heatmap_dtype)

        if topk is None or topk <= 0:
            candidate_edges = jnp.argsort(
                adjmat_pred.reshape(batch_size, -1),
                axis=-1, descending=True, stable=False,
            )
        else:
            _, candidate_edges = jax.lax.top_k(
                adjmat_pred.reshape(batch_size, -1), k=topk,
            )
        candidate_edges = jnp.stack(
            jnp.divmod(candidate_edges, num_nodes + 1), axis=-1
        )
        return candidate_edges

    encode = jax.jit(_encode)
    decode_step = jax.jit(_decode_step)

    # Phase 3a: raw logit decode for beam search (bypass softmax/gumbel/sort)
    def _decode_logits(
        features: jax.Array,
        neighbors: jax.Array,
        tw_bias: jax.Array | None = None,
        resource_mask: jax.Array | None = None,
    ):
        """Return raw (B, N+1, N+1) logits before softmax — for beam search."""
        @jax.vmap
        def _neighbors2adj(nb):
            nb = jnp.where(nb < 0, num_nodes * 2, nb)
            am = jnp.zeros([num_nodes + 1, num_nodes + 1], dtype=jnp.int8)
            ar = jnp.arange(nb.shape[0])
            am = am.at[ar[1:], nb[1:, 0]].set(1, mode='drop')
            am = am.at[ar[1:], nb[1:, 1]].add(1, mode='drop')
            am = am.at[0, :].set(am[:, 0])
            return am

        adjmat = _neighbors2adj(neighbors)
        timestep = (neighbors >= 0).sum(axis=[-1, -2]) / (2 * num_nodes)

        if tw_bias is not None:
            adjmat = adjmat.astype(jnp.float32) + tw_bias
        logits = model.decode(features, timestep, adjmat.astype(jnp.float32))
        if resource_mask is not None:
            logits = jnp.where(resource_mask, logits, -1e9)
        return logits

    decode_logits = jax.jit(_decode_logits)

    # --- Phase 2 动态扩展 (D1-D6) ---
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "maskco_dynamic",
        os.path.join(os.path.dirname(__file__), "maskco_dynamic.py"))
    _dyn = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_dyn)

    scheduler = None
    if enable_anytime:
        scheduler = _dyn.AnytimeScheduler(
            total_budget_ms=time_budget_ms, min_cycles=2, max_cycles=cycles)

    # D3: 约束解码器使用 tw_feasible_batch (per-batch, 在 inference_fn 中传递)

    # D4: 首次 adaptive keep_rate（generation 阶段用 base）
    adaptive_keep_probs = None

    # --- 调度参数 ---

    num_kept_edges = round(keep_rate * num_nodes)
    generation_node_times_schedule = np.linspace(
        0, num_nodes * 2, num=sampling_steps + 1, dtype=np.float32
    ).astype(np.int32)
    searching_node_times_schedule = np.linspace(
        num_kept_edges * 2, num_nodes * 2, num=sampling_steps + 1, dtype=np.float32
    ).astype(np.int32)

    # --- 推理函数 ---
    def inference_fn(seed: int, raw_features_batch: np.ndarray,
                     dist_mat: np.ndarray, unnormalized_demands_batch: np.ndarray,
                     tw_start_batch: np.ndarray, tw_end_batch: np.ndarray,
                     service_time_batch: np.ndarray,
                     tw_feasible_batch: np.ndarray | None,
                     tw_attn_bias_batch: np.ndarray | None,
                     visible_mask_batch: np.ndarray | None = None,
                     quality_loss_batch: np.ndarray | None = None,
                     energy_mat_batch: np.ndarray | None = None,
                     temp_class_batch: np.ndarray | None = None):
        coords_batch = raw_features_batch[..., :2]
        normalized_demands = raw_features_batch[..., 2:3]
        normalized_tw = raw_features_batch[..., 3:]

        features_manager = DynamicAugment(
            lambda x: encode(
                np.concatenate([x, normalized_demands, normalized_tw], axis=-1),
                visible_mask=jnp.array(visible_mask_batch) if visible_mask_batch is not None else None,
                edge_feat=jnp.array(energy_mat_batch) if energy_mat_batch is not None else None,
            ),
            coords_batch,
            augment_level=augment_level,
        )
        storage: dict[int, np.ndarray] = {}
        tw_storage: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        unsalable_storage: dict[int, np.ndarray] = {}  # beam 路线一致 num_unsalable（--enable_quality）
        # Historical delivery-style route proxy; never report as C0 truth.
        final_unsalable_storage: dict[int, np.ndarray] = {}
        route_storage: dict[int, np.ndarray] = {}  # 最终路线（search+2opt 后），路线分析用

        def _single_run(r: int):
            generator = np.random.default_rng(seed + r * 77)
            insertion = CVRPPartialInsertion(
                batch_size, num_nodes + 1, num_workers=num_workers
            )
            unsalable_counts = np.zeros(batch_size, dtype=np.float32)

            # D1: precompute affected segments (default: none)
            affected_mask_batch = None

            # D2: frozen prefix positions
            frozen_positions_batch = None
            if enable_frozen_prefix and frozen_prefix_len > 0:
                frozen_positions_batch = set(range(frozen_prefix_len))

            # D3: dynamic arrival-time TW constraint data
            tw_data_for_d3 = None
            if enable_constraint_decoder:
                tw_data_for_d3 = (
                    jnp.array(coords_batch), jnp.array(tw_start_batch),
                    jnp.array(tw_end_batch), jnp.array(service_time_batch),
                )

            # D4: adaptive keep_rate
            adaptive_keep_probs = None

            # Phase 3a: resource-state feasible decoder helper
            _build_resource_mask = None
            if enable_resource_decoder:
                import importlib.util as _iu
                _spec_rm = _iu.spec_from_file_location(
                    "resource_mask",
                    os.path.join(os.path.dirname(__file__), "resource_mask.py"))
                _rm = _iu.module_from_spec(_spec_rm)
                _spec_rm.loader.exec_module(_rm)
                def _build_resource_mask(neighbors_np):
                    m = _rm.compute_resource_feasibility_mask(
                        neighbors_np, coords_batch, tw_start_batch, tw_end_batch,
                        service_time_batch, unnormalized_demands_batch, capacity, tw_speed)
                    return jnp.array(m)

            # generation phase
            if enable_resource_decoder and _build_resource_mask is not None:
                # --- Phase 3a: Beam search generation ---
                import importlib.util as _iu2
                _spec_rb = _iu2.spec_from_file_location(
                    "resource_beam",
                    os.path.join(os.path.dirname(__file__), "resource_beam.py"))
                _rb = _iu2.module_from_spec(_spec_rb)
                _spec_rb.loader.exec_module(_rb)

                pad_to = expected_padded_length if expected_padded_length else num_nodes * 2 + 5
                beam_sols = np.zeros((batch_size, pad_to), dtype=np.int32)

                for b_idx in range(batch_size):
                    logits_step = decode_logits(
                        features_manager(generator),
                        insertion.neighbors,
                    )
                    logits_b = np.array(logits_step[b_idx])
                    logits_b[:, 0] = -1e9
                    np.fill_diagonal(logits_b, -1e9)

                    # 反推原始 temp_class（int 0/1/2），供品质感知 score 用
                    tc_batch = None
                    if 'temp_class' in dataset:
                        tc_batch = np.round(raw_features_batch[b_idx, :, 5] * 2).astype(int)

                    searcher = _rb.ResourceBeamSearcher(
                        coords_batch[b_idx], tw_start_batch[b_idx],
                        tw_end_batch[b_idx], service_time_batch[b_idx],
                        unnormalized_demands_batch[b_idx], capacity, tw_speed,
                        K=beam_width, tw_margin=tw_margin,
                        enable_quality=enable_quality, lambda_q=lambda_q,
                        quality_salable_threshold=quality_salable_threshold,
                        temp_class=tc_batch,
                        # Week 1 v6: Constraint Cutting
                        enable_constraint_cutting=enable_constraint_cutting,
                        constraint_mode=constraint_mode,
                        constraint_check_top_k=constraint_check_top_k,
                        # 严格 non-anticipatory：beam 只服务可见客户（修复 2026-08-26）
                        visible_mask=(visible_mask_batch[b_idx].astype(bool)
                                      if visible_mask_batch is not None else None),
                    )
                    route, score = searcher.generate(logits_b, max_steps=200)
                    unsalable_counts[b_idx] = searcher.last_num_unsalable
                    if route is None or len(route) <= 2:
                        route = [0] + list(range(1, num_nodes)) + [0]
                    padded = np.array(route, dtype=np.int32)
                    padded = np.pad(padded, (0, max(0, pad_to - len(padded))),
                                   constant_values=0)[:pad_to]
                    beam_sols[b_idx] = padded
                sols = beam_sols
                if expected_padded_length is not None and expected_padded_length > sols.shape[-1]:
                    sols = np.ascontiguousarray(np.pad(
                        sols, [(0, 0), (0, expected_padded_length - sols.shape[-1])],
                        mode='constant'))
            else:
                # --- Original generation phase ---
                for i in range(sampling_steps):
                    rmask = _build_resource_mask(insertion.neighbors) if _build_resource_mask else None
                    candidate_edges = decode_step(
                        features_manager(generator),
                        insertion.neighbors,
                        None if disable_gumbel else
                        generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
                        tw_bias=jnp.array(tw_attn_bias_batch) if tw_attn_bias_batch is not None else None,
                        tw_data=tw_data_for_d3,
                        resource_mask=rmask,
                    )
                    candidate_edges = np.array(candidate_edges)

                    # [CVRPTW] TW 过滤
                    if enable_tw_filter and tw_feasible_batch is not None:
                        feasible_mask = tw_feasible_batch[
                            np.arange(batch_size)[:, None],
                            candidate_edges[..., 0],
                            candidate_edges[..., 1],
                        ]
                        for b in range(batch_size):
                            feasible_edges = candidate_edges[b][feasible_mask[b]]
                            if len(feasible_edges) < len(candidate_edges[b]):
                                n_missing = len(candidate_edges[b]) - len(feasible_edges)
                                filler = np.tile(
                                    np.array([[0, num_nodes]], dtype=np.int32),
                                    (n_missing, 1)
                                )
                                feasible_edges = np.concatenate(
                                    [feasible_edges, filler], axis=0
                                )
                            candidate_edges[b] = feasible_edges[:len(candidate_edges[b])]

                    insertion.insert(
                        np.array(candidate_edges),
                        generation_node_times_schedule[i + 1]
                    )

                sols = insertion.get_sols()
            # Phase 2: TW 修复（在 2-opt 之前）
            if enable_tw_repair_edd:
                if _CPP is not None:
                    sols = _cpp_repair_edd(sols, coords_batch, tw_start_batch, tw_end_batch,
                                           service_time_batch, tw_speed)
                else:
                    sols = tw_repair_edd(
                        sols, coords_batch, tw_start_batch, tw_end_batch,
                        service_time_batch, speed=tw_speed,
                    )
            if enable_tw_aware_2opt_py:
                if _CPP is not None:
                    sols = _cpp_two_opt(
                        sols, dist_mat, coords_batch, tw_start_batch, tw_end_batch,
                        service_time_batch, two_opt_steps, tw_speed,
                    )
                else:
                    sols = tw_aware_two_opt_py(
                        sols, dist_mat, unnormalized_demands_batch,
                        capacity, penalty, two_opt_steps,
                        coords_batch, tw_start_batch, tw_end_batch,
                        service_time_batch, speed=tw_speed,
                    )
            elif enable_tw_preserving_2opt:
                sols = tw_preserving_two_opt(
                    sols, dist_mat, unnormalized_demands_batch,
                    capacity, penalty, two_opt_steps, num_workers,
                    coords_batch, tw_start_batch, tw_end_batch,
                    service_time_batch, speed=tw_speed,
                )
            else:
                sols = cvrp_two_opt(
                    sols, dist_mat, unnormalized_demands_batch,
                    capacity, penalty, num_steps=two_opt_steps,
                    num_workers=num_workers,
                )
            costs = cvrp_eval_cost(
                sols, dist_mat, unnormalized_demands_batch,
                capacity, num_workers=num_workers
            )
            # Fix: 用 dist_mat 重算成本 (C++ 用欧式坐标距离)
            costs = _eval_distmat_cost(sols, dist_mat)
            min_costs = costs
            # P0-8: 统一 incumbent。用 list 存储（每实例变长路线），避免 beam(105) 与
            # C++ insertion(~100) 路线长度不一致导致的 shape mismatch。
            incumbent_routes = [sols[b].copy() for b in range(sols.shape[0])]

            # searching phase (mask-and-reconstruct) — Phase 2 动态扩展
            cycle_idx = 0
            for _ in range(cycles - 1):
                # D6: Anytime solver — 时间预算检查
                if scheduler is not None and not scheduler.should_continue(cycle_idx):
                    break

                # D1+D2+D4 动态 mask（替代原静态随机 mask）
                if enable_event_mask or enable_adaptive_mask:
                    affected = affected_mask_batch if (enable_event_mask and affected_mask_batch is not None) else None
                    frozen_pos = frozen_positions_batch if enable_frozen_prefix else None
                    adp_probs = adaptive_keep_probs if enable_adaptive_mask else None
                    kept_batch = _dyn.dynamic_mask_reconstruct(
                        sols, num_nodes, keep_rate, cycle_idx,
                        affected_mask=affected,
                        frozen_prefix_positions=frozen_pos,
                        adaptive_keep_probs=adp_probs,
                    )
                    edges_for_insert = []
                    for b in range(batch_size):
                        arr = kept_batch[b]
                        padded = np.pad(arr, ((0, max(0, num_kept_edges - len(arr))), (0, 0)),
                                        constant_values=num_nodes * 2)[:num_kept_edges]
                        edges_for_insert.append(padded)
                    edges = np.stack(edges_for_insert, axis=0)
                    insertion.set_state(np.ascontiguousarray(edges))
                else:
                    edges = sols * (num_nodes + 1) + np.roll(sols, shift=1, axis=-1)
                    generator.permuted(edges, axis=1, out=edges)
                    edges = np.stack(np.divmod(edges, num_nodes + 1), axis=-1)
                    edges[:, num_kept_edges:] = num_nodes * 2
                    insertion.set_state(
                        np.ascontiguousarray(edges[:, :num_kept_edges])
                    )

                cycle_idx += 1
                for i in range(sampling_steps):
                    rmask = _build_resource_mask(insertion.neighbors) if _build_resource_mask else None
                    candidate_edges = decode_step(
                        features_manager(generator),
                        insertion.neighbors,
                        None if disable_gumbel else
                        generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
                        tw_bias=jnp.array(tw_attn_bias_batch) if tw_attn_bias_batch is not None else None,
                        tw_data=tw_data_for_d3,
                        resource_mask=rmask,
                    )
                    insertion.insert(
                        np.array(candidate_edges),
                        searching_node_times_schedule[i + 1]
                    )
                sols = insertion.get_sols()
                if expected_padded_length is not None:
                    if expected_padded_length > sols.shape[-1]:
                        sols = np.ascontiguousarray(np.pad(
                            sols,
                            [(0, 0), (0, expected_padded_length - sols.shape[-1])],
                            mode='constant'
                        ))

                # Week 3 v6: In-Cycle Local Search
                # 在 mask-reconstruct 过程中插入 2-opt，早期发现局部优化机会
                if (in_cycle_2opt and
                    cycle_idx >= in_cycle_2opt_start and
                    cycle_idx % in_cycle_2opt_frequency == 0):
                    # 执行轻量级 2-opt（步数少于最终 2-opt）
                    if enable_tw_aware_2opt_py:
                        if _CPP is not None:
                            sols = _cpp_two_opt(
                                sols, dist_mat, coords_batch, tw_start_batch, tw_end_batch,
                                service_time_batch, in_cycle_2opt_steps, tw_speed,
                            )
                        else:
                            sols = tw_aware_two_opt_py(
                                sols, dist_mat, unnormalized_demands_batch,
                                capacity, penalty, in_cycle_2opt_steps,
                                coords_batch, tw_start_batch, tw_end_batch,
                                service_time_batch, speed=tw_speed,
                            )
                    else:
                        # 使用标准 2-opt（如果未启用 TW-aware）
                        sols = cvrp_two_opt(
                            sols, dist_mat, unnormalized_demands_batch,
                            capacity, penalty, num_steps=in_cycle_2opt_steps,
                            num_workers=num_workers,
                        )

                # Phase 2: TW 修复（在最终 2-opt 之前）
                if enable_tw_repair_edd:
                    if _CPP is not None:
                        sols = _cpp_repair_edd(sols, coords_batch, tw_start_batch, tw_end_batch,
                                               service_time_batch, tw_speed)
                    else:
                        sols = tw_repair_edd(
                            sols, coords_batch, tw_start_batch, tw_end_batch,
                            service_time_batch, speed=tw_speed,
                        )
                if enable_tw_aware_2opt_py:
                    if _CPP is not None:
                        sols = _cpp_two_opt(
                            sols, dist_mat, coords_batch, tw_start_batch, tw_end_batch,
                            service_time_batch, two_opt_steps, tw_speed,
                        )
                    else:
                        sols = tw_aware_two_opt_py(
                            sols, dist_mat, unnormalized_demands_batch,
                            capacity, penalty, two_opt_steps,
                            coords_batch, tw_start_batch, tw_end_batch,
                            service_time_batch, speed=tw_speed,
                        )
                elif enable_tw_preserving_2opt:
                    sols = tw_preserving_two_opt(
                        sols, dist_mat, unnormalized_demands_batch,
                        capacity, penalty, two_opt_steps, num_workers,
                        coords_batch, tw_start_batch, tw_end_batch,
                        service_time_batch, speed=tw_speed,
                    )
                else:
                    sols = cvrp_two_opt(
                        sols, dist_mat, unnormalized_demands_batch,
                        capacity, penalty, num_steps=two_opt_steps,
                        num_workers=num_workers,
                    )
                costs = cvrp_eval_cost(
                    sols, dist_mat, unnormalized_demands_batch,
                    capacity, num_workers=num_workers
                )
                # Fix: 用 dist_mat 重算成本
                costs = _eval_distmat_cost(sols, dist_mat)
                improved = costs < min_costs  # P0-8: 只更新更优的实例
                min_costs = np.minimum(min_costs, costs)
                for b in range(sols.shape[0]):
                    if improved[b]:
                        incumbent_routes[b] = sols[b].copy()

            # P0-8: 从 list 重建统一 incumbent（pad 到最长），供 TW/route/quality 同解使用
            max_len = max(len(r) for r in incumbent_routes)
            incumbent_sols = np.zeros((sols.shape[0], max_len), dtype=np.int32)
            for b in range(sols.shape[0]):
                incumbent_sols[b, :len(incumbent_routes[b])] = incumbent_routes[b]

            storage[r] = min_costs

            # [CVRPTW] TW 可行性评估（在统一 incumbent 上，P0-8）
            tw_feas, tw_viol = evaluate_tw_feasibility(
                incumbent_sols, coords_batch, tw_start_batch, tw_end_batch,
                service_time_batch, speed=tw_speed,
            )
            tw_storage[r] = (tw_feas, tw_viol)
            unsalable_storage[r] = unsalable_counts

            # 历史 delivery-style num_unsalable 代理。仅用于旧消融复现；
            # 它不是 pickup-to-depot 的 C0 cargo-manifest 执行轨迹指标。
            final_unsalable = np.zeros(batch_size, dtype=np.float32)
            if temp_class_batch is not None:
                K_TEMP_ = np.array([0.01, 0.002, 0.0002], dtype=np.float32)
                for b in range(batch_size):
                    num, cum, prev = 0, 0.0, 0
                    for x in incumbent_sols[b]:
                        node = int(x)
                        if node == 0:
                            cum = 0.0; prev = 0; continue
                        if node >= dist_mat.shape[1]:
                            continue
                        cum += dist_mat[b, prev, node] + service_time_batch[b, prev]
                        k = K_TEMP_[min(int(temp_class_batch[b, node]), 2)]
                        if 1.0 - np.exp(-k * cum) > quality_salable_threshold:
                            num += 1
                        prev = node
                    final_unsalable[b] = num
            final_unsalable_storage[r] = final_unsalable
            # 保存最终路线（统一 incumbent，P0-8），用于路线分析（②机制验证 / ①route-changing）
            route_storage[r] = incumbent_sols.copy()

        threads = [threading.Thread(target=_single_run, args=[r])
                    for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        min_costs = storage[0]
        for i in range(1, runs):
            min_costs = np.minimum(min_costs, storage[i])

        # [CVRPTW] 合并各 run 的 TW 可行性（任何 run 的 best 都是 feasible）
        tw_feas_all = tw_storage[0][0]
        tw_viol_all = tw_storage[0][1]
        for i in range(1, runs):
            tw_feas_all = tw_feas_all & tw_storage[i][0]
            tw_viol_all = np.minimum(tw_viol_all, tw_storage[i][1])

        # [legacy proxy] 合并各 run 的 beam num_unsalable（平均）
        unsalable_all = unsalable_storage[0]
        for i in range(1, runs):
            unsalable_all = unsalable_all + unsalable_storage[i]
        unsalable_all = unsalable_all / runs

        # [legacy proxy] 合并各 run 的最终路线 num_unsalable（平均）
        final_unsalable_all = final_unsalable_storage[0]
        for i in range(1, runs):
            final_unsalable_all = final_unsalable_all + final_unsalable_storage[i]
        final_unsalable_all = final_unsalable_all / runs

        # 最终路线取 run 0（各 run 收敛后路线近似，用于路线分析）
        route_all = route_storage[0]

        return min_costs, tw_feas_all, tw_viol_all, unsalable_all, final_unsalable_all, route_all

    def run_batch(i: int) -> dict:
        cost, tw_feas, tw_viol, unsalable, final_unsalable, routes = inference_fn(
            i * 888,
            raw_features[i * batch_size:(i + 1) * batch_size],
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
            unnormalized_demands[i * batch_size:(i + 1) * batch_size],
            tw_start[i * batch_size:(i + 1) * batch_size],
            tw_end[i * batch_size:(i + 1) * batch_size],
            service_time[i * batch_size:(i + 1) * batch_size],
            tw_feasible[i * batch_size:(i + 1) * batch_size] if tw_feasible is not None else None,
            tw_attn_bias[i * batch_size:(i + 1) * batch_size] if tw_attn_bias is not None else None,
            visible_mask_batch=visible_mask_ds[i * batch_size:(i + 1) * batch_size] if visible_mask_ds is not None else None,
            quality_loss_batch=quality_loss_ds[i * batch_size:(i + 1) * batch_size] if quality_loss_ds is not None else None,
            energy_mat_batch=energy_mat_ds[i * batch_size:(i + 1) * batch_size] if energy_mat_ds is not None else None,
            temp_class_batch=temp_class[i * batch_size:(i + 1) * batch_size] if 'temp_class' in dataset else None,
        )
        return {
            'cost': cost.mean().item(),
            'tw_feas_rate': tw_feas.mean().item(),
            'tw_viol_avg': tw_viol.mean().item(),
            'num_unsalable': unsalable.mean().item(),
            'num_unsalable_final': final_unsalable.mean().item(),
            'routes': routes,
        }

    # --- Triton/JIT warmup ---
    for _ in range(5):
        i = 0
        generator = np.random.default_rng(i)
        features = encode(
            raw_features[i * batch_size:(i + 1) * batch_size],
            visible_mask=jnp.array(visible_mask_ds[i * batch_size:(i + 1) * batch_size]) if visible_mask_ds is not None else None,
            edge_feat=jnp.array(energy_mat_ds[i * batch_size:(i + 1) * batch_size]) if energy_mat_ds is not None else None,
        )
        candidate_edges = decode_step(
            features,
            jnp.full([batch_size, num_nodes + 1, 2], num_nodes * 2, dtype=jnp.int32),
            None if disable_gumbel else
            generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
            tw_bias=jnp.array(tw_attn_bias[i * batch_size:(i + 1) * batch_size]) if tw_attn_bias is not None else None,
            tw_data=(jnp.array(coords[i * batch_size:(i + 1) * batch_size]), jnp.array(tw_start[i * batch_size:(i + 1) * batch_size]), jnp.array(tw_end[i * batch_size:(i + 1) * batch_size]), jnp.array(service_time[i * batch_size:(i + 1) * batch_size])) if enable_constraint_decoder else None,
        )
        del features, candidate_edges

    # --- 主循环 ---
    costs: list[float] = []
    tw_feas_rates: list[float] = []
    tw_viols: list[float] = []
    num_unsalables: list[float] = []
    num_unsalable_finals: list[float] = []
    all_routes: list[np.ndarray] = []  # 每个 batch 的最终路线 (batch_size, pad_len)
    if threads_over_batches is None or threads_over_batches == 1:
        for i in tqdm.tqdm(range(num_instances // batch_size)):
            result = run_batch(i)
            print(f'batch {i} cost: {result["cost"]:.6f} | '
                  f'TW feas: {result["tw_feas_rate"]:.1%} | '
                  f'TW viol/inst: {result["tw_viol_avg"]:.1f}')
            costs.append(result['cost'])
            tw_feas_rates.append(result['tw_feas_rate'])
            tw_viols.append(result['tw_viol_avg'])
            num_unsalables.append(result['num_unsalable'])
            num_unsalable_finals.append(result['num_unsalable_final'])
            all_routes.append(result['routes'])
    else:
        assert num_instances % (batch_size * threads_over_batches) == 0
        with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
            for j in tqdm.tqdm(
                range(num_instances // batch_size // threads_over_batches)
            ):
                batch_id_start = j * threads_over_batches
                batch_id_end = batch_id_start + threads_over_batches
                submit_results = list(
                    pool.map(run_batch, range(batch_id_start, batch_id_end))
                )
                for i, result in zip(
                    range(batch_id_start, batch_id_end), submit_results
                ):
                    print(f'batch {i} cost: {result["cost"]:.6f} | '
                          f'TW feas: {result["tw_feas_rate"]:.1%} | '
                          f'TW viol/inst: {result["tw_viol_avg"]:.1f}')
                    costs.append(result['cost'])
                    tw_feas_rates.append(result['tw_feas_rate'])
                    tw_viols.append(result['tw_viol_avg'])
                    num_unsalables.append(result['num_unsalable'])
                    num_unsalable_finals.append(result['num_unsalable_final'])
                    all_routes.append(result['routes'])

    mean_cost = sum(costs) / len(costs)
    mean_tw_feas = sum(tw_feas_rates) / len(tw_feas_rates)
    mean_tw_viol = sum(tw_viols) / len(tw_viols)
    print(f'--- Overall Results ---')
    print(f'mean cost:       {mean_cost:.6f}')
    print(f'TW feas rate:    {mean_tw_feas:.1%}')
    print(f'Avg TW viol/inst: {mean_tw_viol:.1f}')

    # --- 权威评估：capacity + completion + depot return（修复 2026-08-26）---
    # 审查报告 P0：原实现只检查客户 TW，没检查 capacity/completion/return depot。
    # 用独立权威 evaluator 重新评估最终路线，统一输出三层可行性。
    try:
        from authoritative_evaluator import evaluate_solution
        cap_feas_list, complete_list, depot_return_list = [], [], []
        for batch_idx, batch_routes in enumerate(all_routes):
            for b in range(batch_routes.shape[0]):
                gi = batch_idx * batch_size + b
                if gi >= num_instances:
                    break
                r = evaluate_solution(
                    batch_routes[b], coords[gi], tw_start[gi], tw_end[gi],
                    service_time[gi], unnormalized_demands[gi], capacity,
                )
                cap_feas_list.append(r['capacity_feasible'])
                complete_list.append(r['complete'])
                depot_return_list.append(r['depot_return_feasible'])
        if cap_feas_list:
            print(f'Cap feas rate:   {np.mean(cap_feas_list):.1%}  (权威评估)')
            print(f'Complete rate:   {np.mean(complete_list):.1%}  (权威评估)')
            print(f'Depot return feas: {np.mean(depot_return_list):.1%}  (权威评估)')
    except Exception as e:
        print(f'[WARN] 权威评估失败: {e}')

    if mean_opt_cost is not None:
        print(f'opt cost:        {mean_opt_cost:.6f}')
        print(f'Gap_ref (vs reference): {(mean_cost - mean_opt_cost) / mean_opt_cost * 100:.6f} %')
    if quality_loss_ds is not None:
        num_unsalable = (quality_loss_ds > quality_salable_threshold).sum(axis=-1).mean()
        print(f'[历史代理/非C0] mean delivery-reference quality proxy: '
              f'{quality_loss_ds.sum(axis=-1).mean():.4f}')
        print(f'[历史代理/非C0] Num unsalable (proxy > {quality_salable_threshold:.2f}): '
              f'{num_unsalable:.2f}/inst')
    if num_unsalable_finals:
        mean_final = sum(num_unsalable_finals) / len(num_unsalable_finals)
        print(f'[历史代理/非C0] Num unsalable (decoded-route delivery proxy): '
              f'{mean_final:.2f}/inst  (阈值 {quality_salable_threshold:.2f})')
    if enable_quality and num_unsalables:
        mean_num_unsalable = sum(num_unsalables) / len(num_unsalables)
        print(f'[历史代理/非C0] Num unsalable (beam proxy): '
              f'{mean_num_unsalable:.2f}/inst')

    # 保存最终路线（路线分析用）
    if save_routes and all_routes:
        max_len = max(r.shape[-1] for r in all_routes)
        padded = [np.pad(r, ((0, 0), (0, max_len - r.shape[-1])), constant_values=0)
                  for r in all_routes]
        routes_all = np.concatenate(padded, axis=0)  # (num_instances, max_len)
        np.savez(save_routes, routes=routes_all)
        print(f'[路线保存] {routes_all.shape[0]} 条最终路线 → {save_routes}')

    return mean_cost


def compute_coldchain_metrics(routes, quality_loss, energy_mat):
    """计算旧版代理量；此函数明确不属于 C0 权威评估。

    Args:
        routes: (N, pad_len) int32 路线数组
        quality_loss: (N, nodes) float32 每节点品质衰减
        energy_mat: (N, nodes, nodes) float32 制冷能耗矩阵
    Returns:
        dict: 带 schema/authoritative 标记的历史代理均值
    """
    N = routes.shape[0]
    total_ql = np.zeros(N)
    total_energy = np.zeros(N)
    for b in range(N):
        ql, en, prev = 0.0, 0.0, 0
        for pos in range(routes.shape[1]):
            node = int(routes[b, pos])
            if node == 0: prev = 0; continue
            ql += quality_loss[b, node]
            en += energy_mat[b, prev, node]
            prev = node
        total_ql[b] = ql
        total_energy[b] = en
    return {
        'schema_version': 'legacy-delivery-proxy-v1',
        'authoritative_coldchain_physics': False,
        'legacy_mean_quality_proxy': total_ql.mean(),
        'legacy_mean_energy_proxy': total_energy.mean(),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # 标准参数（与 decoding/cvrp.py 一致）
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--penalty', type=float, default=3.)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_gumbel', action='store_true', default=False)
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--augment_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--gumbel_scale_factor', type=float, required=True)
    parser.add_argument('--padding_policy', type=str, default='none')
    parser.add_argument('--model_config', type=str, default='softcap_fn',
                        help='模型配置预设名（仅当从 CVRP checkpoint 迁移时使用）')
    parser.add_argument('--use_edge_state', action='store_true', default=False,
                        help='Phase A1: 用 DynamicColdChainModelEdgeState 评估（5D 边特征 + EdgeBiasProjector）')

    # CVRPTW 特有参数
    parser.add_argument('--enable_tw_filter', action='store_true', default=False)
    parser.add_argument('--enable_tw_attn_bias', action='store_true', default=False,
                        help='Exp-05: 将 TW 兼容性作为 attention bias 注入 decoder')
    parser.add_argument('--tw_attn_penalty', type=float, default=5.0,
                        help='TW attention bias 的惩罚系数（默认 5.0）')
    parser.add_argument('--enable_tw_preserving_2opt', action='store_true', default=False,
                        help='Exp-07: TW-preserving 2-opt（违规增加则回退）')
    parser.add_argument('--enable_tw_aware_2opt_py', action='store_true', default=False,
                        help='Phase 2: Python TW-aware 2-opt（只接受不增加TW违规的交换）')
    parser.add_argument('--enable_tw_repair_edd', action='store_true', default=False,
                        help='Phase 2: EDD TW修复（最早截止时间优先重排）')
    parser.add_argument('--lambda_q', type=float, default=0.1,
                        help='品质损失权重 (default 0.1)')
    # === Phase 2: Dynamic-Aware MaskCO (D1-D6) ===
    parser.add_argument('--enable_event_mask', action='store_true', default=False,
                        help='D1: 事件驱动局部重建 (只mask受影响segment)')
    parser.add_argument('--enable_frozen_prefix', action='store_true', default=False,
                        help='D2: 冻结已执行路径前缀 (不可修改)')
    parser.add_argument('--enable_constraint_decoder', action='store_true', default=False,
                        help='D3: 解码器原生TW约束 (logit=-inf for illegal edges)')
    parser.add_argument('--enable_adaptive_mask', action='store_true', default=False,
                        help='D4: 特征自适应mask概率 (tw_slack/quality/confidence)')
    parser.add_argument('--enable_anytime', action='store_true', default=False,
                        help='D6: 时间预算自适应 (anytime solver)')
    parser.add_argument('--enable_resource_decoder', action='store_true', default=False,
                        help='Phase 3a: 资源状态可行性解码器 (TW+容量约束, softmax前mask)')
    parser.add_argument('--beam_width', type=int, default=16,
                        help='Phase 3a: K for beam search (default 16, more=lower cost)')
    parser.add_argument('--tw_margin', type=float, default=0.05,
                        help='Phase 3a: TW 安全裕度 (default 0.05, 0=无裕度)')
    # Week 1 v6: Constraint Cutting
    parser.add_argument('--enable_constraint_cutting', action='store_true', default=False,
                        help='Week 1 v6: Enable constraint cutting in beam search')
    parser.add_argument('--constraint_mode', type=str, default='fast', choices=['fast', 'ortools'],
                        help='Week 1 v6: Constraint checking mode')
    parser.add_argument('--constraint_check_top_k', type=int, default=None,
                        help='Week 1 v6: Only check top-K candidates (None = all)')
    parser.add_argument('--enable_quality', action='store_true', default=False,
                        help='历史品质代理 score（仅复现旧实验，非 C0 权威目标）')
    parser.add_argument('--quality_salable_threshold', type=float, default=0.1,
                        help='历史品质代理阈值（仅复现旧实验，非 C0 可售判定）')
    parser.add_argument('--save_routes', type=str, default=None,
                        help='保存最终路线到 .npz（路线分析用，如 --save_routes /tmp/routes.npz）')
    parser.add_argument('--time_budget_ms', type=int, default=200,
                        help='D6: 总时间预算 (ms, default 200)')
    parser.add_argument('--frozen_prefix_len', type=int, default=0,
                        help='D2: 冻结前缀长度 (>0 启用)')
    parser.add_argument('--tw_speed', type=float, default=1.0)
    parser.add_argument('--tw_max', type=float, default=None,
                        help='TW 归一化分母。None=自动使用 tw_end.max()。'
                             '训练和推理必须使用相同的 tw_max！')
    # Week 3 v6: In-Cycle Local Search
    parser.add_argument('--in_cycle_2opt', action='store_true', default=False,
                        help='Week 3 v6: Enable in-cycle 2-opt local search')
    parser.add_argument('--in_cycle_2opt_frequency', type=int, default=5,
                        help='Week 3 v6: Perform in-cycle 2-opt every N cycles (default 5)')
    parser.add_argument('--in_cycle_2opt_steps', type=int, default=2,
                        help='Week 3 v6: Number of 2-opt steps per in-cycle search (default 2)')
    parser.add_argument('--in_cycle_2opt_start', type=int, default=10,
                        help='Week 3 v6: Start in-cycle 2-opt from cycle N (default 10)')

    args = parser.parse_args()

    # --- 加载模型 ---
    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    if isinstance(model_config, (CVRPTWModelConfig,)) or \
       (DynamicColdChainModelConfig is not None and
        isinstance(model_config, DynamicColdChainModelConfig)):
        pass  # checkpoint compatible
    else:
        print("WARNING: Loading CVRP checkpoint into CVRPTW model.")
        model_config = CVRPTWModelConfig.get_config(args.model_config)
    model_config.dtype = 'float32'
    if args.use_edge_state and DynamicColdChainModelEdgeState is not None:
        # Phase A1: EdgeState 模型（5D 边特征 + EdgeBiasProjector）
        model = DynamicColdChainModelEdgeState(**vars(model_config))
    else:
        model = model_config.construct_model()

    # 尝试加载 — 不匹配的层（如 init_proj）会保持随机初始化
    try:
        graphdef = nnx.graphdef(model)
        model = nnx.merge(graphdef, params)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Partial load: {e}")
        print("Model will use random init for new layers.")

    dataset = dict(np.load(args.data))

    cvrptw_searching_decode(
        dataset, args.capacity, args.penalty, model,
        sampling_steps=args.sampling_steps,
        cycles=args.cycles, keep_rate=args.keep_rate,
        batch_size=args.batch_size, runs=args.runs,
        two_opt_steps=args.two_opt_steps,
        disable_gumbel=args.disable_gumbel,
        gumbel_scale_factor=args.gumbel_scale_factor,
        padding_policy=args.padding_policy,
        heatmap_dtype=args.heatmap_dtype,
        topk=args.topk,
        augment_level=args.augment_level,
        threads_over_batches=args.threads_over_batches,
        seed=args.seed,
        enable_tw_filter=args.enable_tw_filter,
        enable_tw_attn_bias=args.enable_tw_attn_bias,
        tw_attn_penalty=args.tw_attn_penalty,
        enable_tw_preserving_2opt=args.enable_tw_preserving_2opt,
        enable_tw_aware_2opt_py=args.enable_tw_aware_2opt_py,
        enable_tw_repair_edd=args.enable_tw_repair_edd,
        tw_speed=args.tw_speed,
        tw_max=args.tw_max,
        # Phase 2
        enable_event_mask=args.enable_event_mask,
        enable_frozen_prefix=args.enable_frozen_prefix,
        enable_constraint_decoder=args.enable_constraint_decoder,
        enable_adaptive_mask=args.enable_adaptive_mask,
        enable_anytime=args.enable_anytime,
        enable_resource_decoder=args.enable_resource_decoder,
        beam_width=args.beam_width,
        tw_margin=args.tw_margin,
        # Week 1 v6: Constraint Cutting
        enable_constraint_cutting=args.enable_constraint_cutting,
        constraint_mode=args.constraint_mode,
        constraint_check_top_k=args.constraint_check_top_k,
        # Week 3 v6: In-Cycle Local Search
        in_cycle_2opt=args.in_cycle_2opt,
        in_cycle_2opt_frequency=args.in_cycle_2opt_frequency,
        in_cycle_2opt_steps=args.in_cycle_2opt_steps,
        in_cycle_2opt_start=args.in_cycle_2opt_start,
        enable_quality=args.enable_quality,
        lambda_q=args.lambda_q,
        quality_salable_threshold=args.quality_salable_threshold,
        save_routes=args.save_routes,
        time_budget_ms=args.time_budget_ms,
        frozen_prefix_len=args.frozen_prefix_len,
    )
