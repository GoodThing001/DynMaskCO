# -*- coding: utf-8 -*-
"""CaDA（Cross-Problem Routing Solver with Constraint-Aware Dual-Attention,
CIAM-Group/CaDA, arXiv:2412.00346）CVRP100/VRPTW100 推理适配。

重要说明（口径勘误）：任务描述中 "CaDA=Collaborative Diffusion Model / 扩散采样步数"
与作者仓库实际不符。作者 provenance.json 与 README 明确 CaDA = 约束感知双注意力
跨问题路由求解器（ICML 2025），其官方推理路径是 **POMO 式多起点 + 贪心解码**，
仓库内不存在任何扩散采样代码（本适配未使用、也不需要扩散噪声种子）。
确定性由「固定权重 + 贪心解码 + 固定 8 增广 + 固定 POMO 起点」保证，seed 仅固定
torch 随机源（接口统一），不改变解码结果。

作者资产（远端只读，未做任何修改）：
  assets/neural_assets/CaDA/{policy.pt, provenance.json, utils/, 100/{model.py, config.yaml}}
  其中 100/model.py、100/config.yaml、utils/functions.py 已与本地归档字节核验一致
  （SHA256 相同），policy.pt = 作者 checkpoint-300.pt 的 model_state_dict（162 张量）。

数据口径（与作者 trainer.test_lib 的 CVRPLIB 推理路径一致，见 _mtvrp_common.py）：
  * 坐标：各向同性 min-shift / max-span 归一化（作者 normalize_coord）；
  * 需求：demand / capacity（作者 scale_demand 口径，车辆容量=1）；
  * VRPTW：时间轴 s = max_time(4.6) / depot_window_end，speed = 1/(s*scale)，
    模型内时间 = s × 真实时间（时窗/服务/回仓约束与真实问题严格等价）；
  * CVRP：TW 默认 [0,inf]、service=0、speed=1；
  * p_s_tag = [C=1, O=0, TW∈{0,1}, L=0, B=0, size/2000]（作者任务提示向量）。

解码配置（作者默认）：
  * 模型 eval + 贪心（VRPModel.greedy，作者 test 分支口径）；
  * POMO 全起点（num_starts = 客户数 = 100）；
  * StateAugmentation dihedral8 共 8 个增广（作者 trainer.test_lib 默认 num_augment=8）；
  * 800 条并行轨迹中按作者 get_reward（归一化坐标负行程，等距变换下等价于真实距离）
    取最优 → 切分为车辆路线。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线；
  不支持（任务, 规模）组合抛 NotImplementedError（登记 UNSUPPORTED）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- 远端路径常量（作者资产只读） -----------------------------------------
_HERE = Path(__file__).resolve().parent
_VENDOR = str(_HERE / "_vendor")                      # tensordict/entmax/rl4co 垫片
_ASSETS = "/home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets/neural_assets"
_CADA_ROOT = _ASSETS + "/CaDA"                        # utils/ 包与 policy.pt
_CADA_100 = _ASSETS + "/CaDA/100"                     # model.py 与 config.yaml
_POLICY_PT = _ASSETS + "/CaDA/policy.pt"

# ---- 模块级缓存（同一进程多次 solve 只加载一次权重与环境） ------------------
_CACHE = None  # (model, env)


def _setup_paths():
    """把本目录、垫片目录与作者资产目录插到 sys.path 最前（只影响本进程）。"""
    for p in (str(_HERE), _VENDOR, _CADA_ROOT, _CADA_100):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load():
    """懒加载：作者 VRPModel（100/config.yaml 架构）+ policy.pt 权重（strict）。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import torch
    import yaml

    torch.set_num_threads(1)  # 单核验证：限制线程避免空耗

    _setup_paths()
    # 作者代码（只读导入，不修改）：assets/100/model.py 与 assets/utils/
    from model import VRPModel

    # ---- 作者 args（run.py 口径：config.yaml + sqrt_embedding_dim） ----------
    cfg = yaml.safe_load(
        Path(_CADA_100 + "/config.yaml").read_text(encoding="utf-8"))
    model_params = dict(cfg["model_params"])
    model_params["sqrt_embedding_dim"] = float(model_params["embedding_dim"]) ** 0.5

    class _Args:
        pass

    args = _Args()
    args.model_params = model_params
    args.log = lambda *a, **k: None  # 仅训练采样分支使用；贪心解码不调用

    model = VRPModel(args)
    state = torch.load(_POLICY_PT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    from _mtvrp_common import MiniMTVRPEnv
    env = MiniMTVRPEnv()
    _CACHE = (model, env)
    return _CACHE


def _decode_all(model, env, td):
    """复刻作者 VRPModel.forward 的推理主循环，额外记录全部轨迹的动作序列。

    （作者 forward 只返回 reward/log_likelihood 不返回 actions；此处按同一循环逻辑
    逐行复刻并在 eval 模式下贪心选点，以便取最优轨迹切分为路线。）
    """
    import torch

    from model import PrecomputedCache, reshape_by_heads
    from _mtvrp_common import batchify

    args = model.args
    # 1) 任务提示 + 编码（batch = 8 增广）
    prompt = model.prompt_net(td)["prompt"]
    node_embed = model.encoder(td, prompt)
    # 2) POMO 多起点（作者 select_start_nodes：轮转 1..n，起点数 = 客户数）
    num_starts = env.get_num_starts(td)
    action = env.select_start_nodes(td, num_starts)
    td = batchify(td, num_starts)
    actions_list = [action]
    td.set("action", action)
    td = env.step(td)["next"]
    # 3) 解码器预计算缓存（作者 PrecomputedCache 口径）
    decoder_k = reshape_by_heads(model.decoder.Wk(node_embed),
                                 head_num=args.model_params["head_num"])
    decoder_v = reshape_by_heads(model.decoder.Wv(node_embed),
                                 head_num=args.model_params["head_num"])
    decoder_single_head_k = node_embed.transpose(1, 2)
    cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)
    # 4) 主循环：直到全部节点（含仓库）访问完
    max_steps = 2 * env.get_num_starts(td) + 128  # 防御性步数上限（正常 ~N+K 步）
    step = 0
    while not td["done"].all():
        logprobs, _mask = model.decoder(td, cache, num_starts)
        # 作者 eval 分支：VRPModel.greedy（log_softmax 上 argmax）
        select = logprobs.argmax(dim=-1)
        td.set("action", select)
        actions_list.append(select)
        td = env.step(td)["next"]
        step += 1
        if step > max_steps:
            raise RuntimeError("CaDA 解码步数异常，疑似死循环")
    # 5) 回报与动作（作者 get_reward：负归一化行程）
    actions = torch.stack(actions_list, 1)
    rewards = env.get_reward(td, actions)
    return actions, rewards


def solve(inst, objective, seed):
    """标准接口：standard_vrp 实例 → 完整覆盖路线（客户 id 1..dimension）。"""
    # ---- 支持矩阵：多任务 100 模型（作者仅提供 100 权重） -------------------
    if objective not in ("cvrp", "vrptw"):
        raise NotImplementedError(
            "CaDA 不支持 objective=%r（仅 'cvrp'/'vrptw'），登记 UNSUPPORTED" % objective)
    if getattr(inst, "dimension", None) != 100:
        raise NotImplementedError(
            "CaDA 仅支持 n=100（作者资产只有 100 检查点），收到 n=%r，登记 UNSUPPORTED"
            % getattr(inst, "dimension", None))

    import torch

    torch.manual_seed(int(seed))  # 贪心解码确定性；seed 仅固定随机源（接口统一）
    torch.set_num_threads(1)      # 单核验证

    from _mtvrp_common import (MiniMTVRPEnv, actions_to_routes, augment_td_8,
                               build_mtvrp_td)

    model, env = _load()

    # ---- 实例 → 作者训练域输入（batch=1，含 depot 行） ----------------------
    td = build_mtvrp_td(inst, objective)
    # ---- 重置（初始状态 + 掩码）→ 8 增广 -----------------------------------
    # 注意：p_s_tag 必须在 reset 之后设置（作者 trainer.test_lib 顺序：reset 只
    # 复制环境字段，p_s_tag 在 reset 后按任务写入）
    td = env.reset(td)
    has_tw = 1.0 if objective == "vrptw" else 0.0
    td["p_s_tag"] = torch.tensor(
        [[1.0, 0.0, has_tw, 0.0, 0.0, (inst.dimension + 1) / 2000.0]],
        dtype=torch.float32)
    td_aug = augment_td_8(td)

    # ---- 800 条轨迹（8 增广 × 100 POMO 起点）贪心解码，取最优 ----------------
    with torch.no_grad():
        actions, rewards = _decode_all(model, env, td_aug)
    best = int(rewards.argmax())

    return actions_to_routes(actions[best].tolist(), inst.dimension)
