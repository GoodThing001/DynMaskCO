# -*- coding: utf-8 -*-
"""CaDA / RouteFinder 共享的 MTVRP 精简环境与输入构造（纯 torch，独立于作者代码）。

两个作者的官方环境（CaDA 100/envs/env.py 与 routefinder/envs/mtvrp/env.py）对
C/VRPTW 语义逐行一致（本目录已逐行比对）：step/get_action_mask/get_reward 逻辑相同，
仅 CaDA 旧版没有 backhaul_class 字段。本模块把作者环境的核心转移逻辑**逐行移植**为
纯 torch 实现（不依赖 torchrl/tensordict），并附实例特征构造与路线切分。

口径（与作者官方 CVRPLIB/VRPTW 推理路径一致）：
  * 坐标：仓库+客户整体各向同性归一化（CaDA 作者 trainer.test_lib 的 normalize_coord：
    逐轴减各自最小值、除以公共最大跨度 scale）→ 训练域 [0,1]²；各向同性保证模型内
    距离 = 真实距离 / scale，时间窗可行性与真实问题严格等价；
  * 需求：demand / capacity（vehicle_capacity=1，作者 scale_demand 口径）；
  * 时窗（VRPTW）：时间轴统一缩放 s = max_time / depot_window_end（CaDA/RouteFinder
    训练 max_time=4.6），TW/service/depot 窗口均乘 s；模型速度 speed=1/(s*scale)，
    使模型内时间 = s × 真实时间（到达、最早/最晚时窗、回仓约束全部严格等价）；
  * CVRP：TW 取默认 [0, inf]、service=0、speed=1（作者默认，掩码中 TW/距离限制失效）；
  * open_route=False、distance_limit=inf（标准 CVRP/VRPTW 无开放路线与距离上限）。
"""
from __future__ import annotations

import numpy as np
import torch

from _vendor.tensordict import TensorDict

# CaDA/RouteFinder 训练生成器的 TW 上界（depot 窗口末端，见两作者 generator.py）
MAX_TIME = 4.6


# ---------------------------------------------------------------- 张量工具
def get_distance(x, y):
    """欧氏距离（作者 rl4co/CaDA utils 同款）。"""
    return (x - y).norm(p=2, dim=-1)


def gather_by_index(src, idx, dim=1, squeeze=True):
    """按索引 gather（作者同款）。"""
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def batchify(x, shape):
    """批维复制展平（作者 utils/functions.py 同款；TensorDict shim 支持 expand/view）。"""
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        if s > 0:
            bs = x.shape
            x = x.expand(s, *bs).contiguous().view(bs[0] * s, *bs[1:])
    return x


def dihedral_8_augmentation(xy):
    """POMO 二面体群 8 增广（作者 envs/transformer.py 同款：4 旋转反射 × 2 转置）。"""
    x, y = xy.split(1, dim=2)
    z0 = torch.cat((x, y), dim=2)
    z1 = torch.cat((1 - x, y), dim=2)
    z2 = torch.cat((x, 1 - y), dim=2)
    z3 = torch.cat((1 - x, 1 - y), dim=2)
    z4 = torch.cat((y, x), dim=2)
    z5 = torch.cat((1 - y, x), dim=2)
    z6 = torch.cat((y, 1 - x), dim=2)
    z7 = torch.cat((1 - y, 1 - x), dim=2)
    return torch.cat((z0, z1, z2, z3, z4, z5, z6, z7), dim=0)


def augment_td_8(td):
    """作者 StateAugmentation(dihedral8, num_augment=8) 同款：8× 复制后替换 locs。

    注意作者的 dihedral_8_augmentation_wrapper 先取第一个批样本（reduce=True），
    再做 8 个二面体变换拼接，最终 locs 形状仍为 [B, N, 2]（B=8 增广×原批）。
    二面体群为等距变换，action_mask 只依赖距离与时窗，增广后仍严格有效（作者语义）。
    """
    td_aug = batchify(td, 8)
    first = td_aug["locs"][: td_aug["locs"].shape[0] // 8]
    td_aug["locs"] = dihedral_8_augmentation(first)
    return td_aug


# ---------------------------------------------------------------- 精简环境
class MiniMTVRPEnv:
    """CaDA/RouteFinder 共用的 MTVRP 环境（转移语义与作者环境逐行一致）。

    节点编号：0=仓库，1..n=客户（与 standard_vrp 实例一致）。
    只实现推理路径需要的方法：reset/step/get_action_mask/get_reward/
    select_start_nodes/get_num_starts。
    """

    name = "mtvrp"

    def reset(self, td):
        """给定完整特征 td（含 depot 行）→ 初始状态 td（克隆特征 + 清零状态字段）。"""
        batch_size = td.batch_size
        td_reset = TensorDict(
            {
                "locs": td["locs"].clone(),
                "demand_backhaul": td["demand_backhaul"].clone(),
                "demand_linehaul": td["demand_linehaul"].clone(),
                "backhaul_class": td["backhaul_class"].clone(),
                "distance_limit": td["distance_limit"].clone(),
                "service_time": td["service_time"].clone(),
                "open_route": td["open_route"].clone(),
                "time_windows": td["time_windows"].clone(),
                "speed": td["speed"].clone(),
                "vehicle_capacity": td["vehicle_capacity"].clone(),
                "capacity_original": td["capacity_original"].clone(),
                "current_node": torch.zeros((*batch_size,), dtype=torch.long),
                "current_route_length": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "current_time": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "used_capacity_backhaul": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "used_capacity_linehaul": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "visited": torch.zeros((*batch_size, td["locs"].shape[-2]), dtype=torch.bool),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def step(self, td):
        """执行一步（作者 env._step 逐行移植；原地更新并返回 {'next': td}）。"""
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        service_time = gather_by_index(td["service_time"], curr_node, dim=1, squeeze=False)
        start_times = gather_by_index(td["time_windows"], curr_node, dim=1, squeeze=False)[..., 0]
        # 到点等待：current_time = max(到达, 最早时窗) + 服务时间；回仓库时间清零
        curr_time = (curr_node[:, None] != 0) * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance
        )
        selected_demand_linehaul = gather_by_index(td["demand_linehaul"], curr_node, dim=1, squeeze=False)
        selected_demand_backhaul = gather_by_index(td["demand_backhaul"], curr_node, dim=1, squeeze=False)
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done).float()

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return {"next": td}

    @staticmethod
    def get_action_mask(td):
        """可行动作掩码（作者 env.get_action_mask 逐行移植，含 backhaul_class 分支）。"""
        curr_node = td["current_node"]
        locs = td["locs"]
        d_ij = get_distance(gather_by_index(locs, curr_node)[..., None, :], locs)
        d_j0 = get_distance(locs, locs[..., 0:1, :])

        early_tw, late_tw = td["time_windows"][..., 0], td["time_windows"][..., 1]
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        can_reach_customer = arrival_time < late_tw
        # 闭合路线必须能返回仓库：开始时刻 + 服务 + 回程 < 仓库最晚时窗
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[..., 0:1]

        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[..., None]
        is_carrying_backhaul = (
            gather_by_index(td["demand_backhaul"], curr_node, dim=1, squeeze=False) > 0
        )
        # 回程类 1（经典 backhaul：先送后取）：本任务 demand_backhaul 恒 0，
        # 该公式退化为纯 linehaul 容量约束（与 CaDA 旧版掩码完全一致）
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))
        # 回程类 2（混合取送）
        cannot_serve_linehaul = (
            td["demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )
        # 仓库掩码：从仓库出发且仍有可访问客户时不立即返回
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    def get_reward(self, td, actions):
        """回报 = -总行程（作者 get_reward 同款；闭合路线含回仓库边）。"""
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)
        distances = get_distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)
        return -tour_length

    def get_num_starts(self, td):
        """多起点数 = 客户数（POMO 全起点；作者 AllSelectStartNodes 口径）。"""
        return td["locs"].shape[-2] - 1

    def select_start_nodes(self, td, num_starts):
        """POMO 起点选择：每个批样本轮转 1..n（作者 AllSelectStartNodes 同款）。"""
        num_loc = td["locs"].shape[-2] - 1
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
            + 1
        )
        return selected


# ---------------------------------------------------------------- 输入构造
def build_mtvrp_td(inst, objective):
    """standard_vrp 实例 → 作者训练域 TensorDict（batch=1，depot 在 0 行）。

    inst: CVRPInstance / VRPTWInstance；objective: 'cvrp' | 'vrptw'。
    """
    n = inst.dimension
    capacity = float(inst.capacity)

    # ---- 坐标：各向同性归一化（CaDA 作者 normalize_coord 口径） ------------
    coords = np.array([inst.coords[i] for i in range(n + 1)], dtype=np.float32)
    lo = coords.min(axis=0)
    span = max(float(coords[:, 0].max() - coords[:, 0].min()),
               float(coords[:, 1].max() - coords[:, 1].min()))
    if span <= 1e-9:
        raise ValueError("实例坐标跨度为 0，无法归一化")
    locs = (coords - lo) / span

    # ---- 需求 / 容量（作者 scale_demand 口径：demand/capacity，容量=1） ------
    demand = np.array([inst.demands[i] for i in range(1, n + 1)], dtype=np.float32)
    demand_linehaul = demand / capacity

    # ---- 时窗 / 服务时间 ----------------------------------------------------
    if objective == "vrptw":
        depot_end = float(inst.depot_window[1])
        if depot_end <= 0:
            raise ValueError("仓库最晚时窗非正，无法做时间缩放")
        s = MAX_TIME / depot_end  # 时间轴缩放：模型时间 = s × 真实时间
        tw = np.array([inst.windows[i] for i in range(n + 1)], dtype=np.float32)
        service = np.array([inst.service[i] for i in range(n + 1)], dtype=np.float32)
        time_windows = tw * s
        service_time = service * s
        speed = 1.0 / (s * span)  # 模型内时间增量 = (d/span)/speed = s*d
    else:
        time_windows = np.zeros((n + 1, 2), dtype=np.float32)
        time_windows[:, 1] = float("inf")
        service_time = np.zeros(n + 1, dtype=np.float32)
        speed = 1.0

    td = TensorDict(
        {
            # 注意：与真实 tensordict 相同，张量首维即批维（batch=1）；
            # 形状与作者环境完全一致：demand/service 为 2-D [B, n+1]（无尾维 1），
            # time_windows 为 3-D [B, n+1, 2]（作者 generator/env 口径）。
            "locs": torch.from_numpy(locs)[None],                      # [1, n+1, 2]
            "demand_linehaul": torch.from_numpy(
                np.concatenate([[0.0], demand_linehaul]).astype(np.float32)
            )[None],                                                   # [1, n+1]
            "demand_backhaul": torch.zeros(1, n + 1, dtype=torch.float32),
            "backhaul_class": torch.ones(1, 1, dtype=torch.float32),
            "distance_limit": torch.full((1, 1), float("inf"), dtype=torch.float32),
            "service_time": torch.from_numpy(service_time)[None],      # [1, n+1]
            "open_route": torch.zeros(1, 1, dtype=torch.bool),
            "time_windows": torch.from_numpy(time_windows)[None],      # [1, n+1, 2]
            "speed": torch.full((1, 1), float(speed), dtype=torch.float32),
            "vehicle_capacity": torch.ones(1, 1, dtype=torch.float32),
            "capacity_original": torch.full((1, 1), capacity, dtype=torch.float32),
        },
        batch_size=[1],
    )
    return td


# ---------------------------------------------------------------- 输出切分
def actions_to_routes(actions, n):
    """解码动作序列 → 路线元组。0=仓库分段点，节点号 1..n 即 standard_vrp 客户 id。

    防御性覆盖补齐：作者解码必然访问全部客户（done=全部 visited），该分支仅兜底，
    若触发说明环境/模型路径异常，应由调用方检查。
    """
    routes = []
    cur = []
    seen = set()
    for t in actions:
        t = int(t)
        if t == 0:
            if cur:
                routes.append(tuple(cur))
                cur = []
        elif 1 <= t <= n:
            cur.append(t)
            seen.add(t)
    if cur:
        routes.append(tuple(cur))
    for c in range(1, n + 1):
        if c not in seen:
            routes.append((c,))
    return tuple(routes)
