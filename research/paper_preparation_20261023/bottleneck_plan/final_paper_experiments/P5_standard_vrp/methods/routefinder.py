# -*- coding: utf-8 -*-
"""RouteFinder（ai4co/routefinder, TMLR 2025，MTVRP 基础模型）CVRP100/VRPTW100 推理适配。

作者资产（远端只读，未做任何修改）：
  assets/neural_assets/RouteFinder/{policy.pt, provenance.json, configs/, routefinder/}
  policy.pt = 作者 rf-transformer.ckpt 的 policy 模块 state_dict（61 张量，与
  routefinder/models/policy.py + encoder/nn 模块结构一一对应）。

依赖说明：RouteFinder 作者代码依赖 rl4co/torchrl/tensordict。远端 MASKCO_env
（torch 2.6.0+cpu）未安装这些包，且任何其他 conda env 都不含完整 torch 栈；
为不改动 MASKCO_env，本目录 _vendor/ 内嵌了：
  * tensordict 最小兼容层（仅推理路径用到的 API，纯 torch）；
  * rl4co 0.6.0（MIT，ai4co/rl4co）推理最小子集（AM 解码器/指针网络/解码策略，
    逐字取自服务器 cc_compare 环境同版本 site-packages 源码，只裁剪未用模块）。
两处垫片仅影响本进程 sys.path，不改任何全局环境。

模型结构（作者 configs/experiment/main/rf/rf-transformer-100.yaml）：
  RouteFinderPolicy(normalization='rms', encoder_use_prenorm=True,
                    encoder_use_post_layers_norm=True,
                    parallel_gated_kwargs={'mlp_activation': 'silu'})
  其余取作者默认：embed_dim=128、num_heads=8、encoder 6 层、ff 512、
  use_graph_context=False（图上下文在解码中恒为 0，与训练一致）。

数据口径（与作者 eval-cvrplib notebook / MTVRP 训练域一致，见 _mtvrp_common.py）：
  * 坐标：各向同性 min-shift / max-span 归一化到 [0,1] 训练域
    （作者 notebook 用逐轴 min-max；此处采用 CaDA 作者的各向同性口径，
    因逐轴缩放会破坏度量/时窗等价性——VRPTW 掩码必须与真实约束严格一致）；
  * 需求：demand / capacity（车辆容量=1）；
  * VRPTW：时间轴 s = max_time(4.6) / depot_window_end、speed = 1/(s*scale)；
  * CVRP：TW 默认 [0,inf]、service=0、speed=1。

解码配置（作者默认 rollout）：
  * policy(phase='test', decode_type='greedy', num_starts=100, return_actions=True)
    —— 与作者 routefinder/utils.py evaluate 完全一致；
  * 8 个二面体增广（作者 eval-cvrplib notebook num_augment=8）；
  * 800 条并行轨迹按作者 get_reward（负归一化行程）取最优 → 切分为车辆路线；
  * 贪心解码确定性，seed 仅固定 torch 随机源（接口统一）。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线；
  不支持（任务, 规模）组合抛 NotImplementedError（登记 UNSUPPORTED）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- 远端路径常量（作者资产只读） -----------------------------------------
_HERE = Path(__file__).resolve().parent
_VENDOR = str(_HERE / "_vendor")                      # tensordict/rl4co 垫片
_ASSETS = "/home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets/neural_assets"
_RF_ROOT = _ASSETS + "/RouteFinder"                   # routefinder/ 包
_POLICY_PT = _ASSETS + "/RouteFinder/policy.pt"

# ---- 作者 rf-transformer-100.yaml 的 policy 超参（其余取作者默认） -----------
_POLICY_KWARGS = dict(
    normalization="rms",
    encoder_use_prenorm=True,
    encoder_use_post_layers_norm=True,
    parallel_gated_kwargs={"mlp_activation": "silu"},
    env_name="mtvrp",
)

# ---- 模块级缓存（同一进程多次 solve 只加载一次权重） ------------------------
_CACHE = None  # (policy, env)


def _setup_paths():
    """把本目录、垫片目录与作者资产目录插到 sys.path 最前（只影响本进程）。"""
    for p in (str(_HERE), _VENDOR, _RF_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load():
    """懒加载：作者 RouteFinderPolicy（作者超参）+ policy.pt 权重（strict）。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import torch

    torch.set_num_threads(1)  # 单核验证：限制线程避免空耗

    _setup_paths()
    # 作者代码（只读导入，不修改）：routefinder 包
    from routefinder.models.policy import RouteFinderPolicy

    policy = RouteFinderPolicy(**_POLICY_KWARGS)
    state = torch.load(_POLICY_PT, map_location="cpu")
    policy.load_state_dict(state, strict=True)
    policy.eval()

    from _mtvrp_common import MiniMTVRPEnv
    env = MiniMTVRPEnv()
    _CACHE = (policy, env)
    return _CACHE


def solve(inst, objective, seed):
    """标准接口：standard_vrp 实例 → 完整覆盖路线（客户 id 1..dimension）。"""
    # ---- 支持矩阵：100 基础模型（作者资产仅 100 检查点） --------------------
    if objective not in ("cvrp", "vrptw"):
        raise NotImplementedError(
            "RouteFinder 不支持 objective=%r（仅 'cvrp'/'vrptw'），登记 UNSUPPORTED"
            % objective)
    if getattr(inst, "dimension", None) != 100:
        raise NotImplementedError(
            "RouteFinder 仅支持 n=100（作者资产只有 100 检查点），收到 n=%r，"
            "登记 UNSUPPORTED" % getattr(inst, "dimension", None))

    import torch

    torch.manual_seed(int(seed))  # 贪心解码确定性；seed 仅固定随机源（接口统一）
    torch.set_num_threads(1)      # 单核验证

    from _mtvrp_common import (actions_to_routes, augment_td_8, build_mtvrp_td)

    policy, env = _load()

    # ---- 实例 → 作者训练域输入（batch=1，含 depot 行）→ 重置 → 8 增广 --------
    td = build_mtvrp_td(inst, objective)
    td = env.reset(td)
    td_aug = augment_td_8(td)  # 作者 num_augment=8（dihedral8）口径

    # ---- 作者默认 rollout：greedy + POMO 全起点（num_starts=100） -------------
    with torch.inference_mode():
        out = policy(td_aug, env, phase="test", decode_type="greedy",
                     return_actions=True, num_starts=100)
    actions = out["actions"]   # [800, T]
    rewards = out["reward"]    # [800]（负归一化行程）
    best = int(rewards.argmax())

    return actions_to_routes(actions[best].tolist(), inst.dimension)
