# -*- coding: utf-8 -*-
"""MVMoE（Zhou et al., ICML 2024）多任务构造式求解器推理适配（P5 外部方法工作器）。

只读使用作者资产（_maskco_recent5_20261120/assets/MVMoE/，未做任何修改）；本模块是独立适配层。
运行环境：/home/hzeng/envs/MASKCO_env（torch 2.6.0+cpu），CPU 推理，torch.set_num_threads(1)。

支持矩阵：
  * CVRP / VRPTW：多任务权重 mvmoe_4e_n100/epoch-5000.pt（problem=Train_ALL）→ 两任务均 OK。

数据口径（与作者 Tester.py 基准求解路径 _solve_cvrplib/_solve_cvrptwlib 完全一致）：
  * CVRP: 坐标 / loc_scaler（loc_scaler=1000，X-n101-k25 坐标最大值 ≤1000）；
    需求 / capacity。
  * VRPTW: scaler = max(全部坐标最大值, 仓库截止时间/3)；坐标、服务时长、时窗全部 /scaler；
    仓库时窗 [0, depot_due/scaler]（C101 → scaler=412、depot_end=3.0）；speed=1。
  * 节点顺序：0=仓库、1..n=客户（与 standard_vrp 内部编号一致，输出可直接评价）。

解码配置（作者 test.py 官方默认）：
  * model_type=MOE（检查点状态键核对：decoder.multi_head_combine 为 MoE 专家层，
    Enc0..Enc5 层 feedForward 亦为 MoE 层，与 MOEModel（非 Light）一致，strict 加载校验）；
  * num_experts=4、topk=2、routing_level=node、routing_method=input_choice、
    expert_loc=['Enc0'..'Enc5','Dec']、norm=instance、norm_loc=norm_last、
    embedding_dim=128、encoder_layer_num=6、qkv_dim=16、head_num=8、logit_clipping=10；
  * eval_type='argmax'（作者默认贪心解码）；aug_factor=8（8 折实例增强）+ pomo_size=n，
    在 8×n 条轨迹中取负距离奖励最大者（作者 tester 同款选择）。
  * 确定性：eval 模式下 MoE 噪声门控关闭、argmax 解码无随机性；seed 仍按约定固定。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量/时窗在作者解码掩码内保证）。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

# ---- 远端路径常量（作者资产只读） -----------------------------------------
_ASSET = '/home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets/MVMoE'
_CKPT = _ASSET + '/pretrained/mvmoe_4e_n100/epoch-5000.pt'
_AUG_FACTOR = 8  # 作者 test.py 默认 8 折实例增强

# ---- 作者 test.py / README 的 MVMoE/4E 官方模型超参（与检查点严格对应） ----
_MODEL_PARAMS = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'decoder_layer_num': 1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'num_experts': 4,
    'eval_type': 'argmax',      # 作者默认贪心解码
    'norm': 'instance',
    'norm_loc': 'norm_last',
    'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
    'problem': None,            # 从检查点读取（Train_ALL）
    'topk': 2,
    'routing_level': 'node',
    'routing_method': 'input_choice',
    'device': torch.device('cpu'),
}

# ---- 模块级模型缓存（同一进程多次 solve 只加载一次权重） ------------------
_MODEL = None


def _get_model():
    """懒加载作者 MOEModel（mvmoe_4e_n100/epoch-5000.pt，只读作者代码）。"""
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗
        if _ASSET not in sys.path:
            sys.path.insert(0, _ASSET)
        from models import MOEModel  # 作者 models/__init__.py（仅 torch，自包含）

        checkpoint = torch.load(_CKPT, map_location='cpu', weights_only=False)
        params = dict(_MODEL_PARAMS)
        params['problem'] = checkpoint['problem']  # 'Train_ALL'
        model = MOEModel(**params)
        # strict=True：若模型类型/专家位置等与检查点不一致会立即报错（口径自校验）
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()  # eval 模式：MoE 噪声门控关闭 → 确定性
        _MODEL = model
    return _MODEL


def _coords_of(inst):
    """standard_vrp 实例 → (n+1, 2) 原始坐标数组；0=仓库、1..n=客户。"""
    n = inst.dimension
    return np.array([inst.coords[i] for i in range(n + 1)], dtype=np.float64)


def _build_cvrp_data(inst):
    """CVRPInstance → 作者 CVRP 数据元组与 loc_scaler（口径同 Tester._solve_cvrplib）。"""
    n = inst.dimension
    coords = _coords_of(inst)
    loc_scaler = 1000.0
    assert float(coords.max()) <= loc_scaler, 'loc_scaler 过小（作者口径要求坐标<=1000）'
    depot_xy = torch.tensor(coords[0:1], dtype=torch.float32)[None] / loc_scaler   # (1,1,2)
    node_xy = torch.tensor(coords[1:], dtype=torch.float32)[None] / loc_scaler     # (1,n,2)
    node_demand = torch.tensor(
        [[inst.demands[i] for i in range(1, n + 1)]], dtype=torch.float32
    ) / float(inst.capacity)                                                      # (1,n)
    return (depot_xy, node_xy, node_demand), loc_scaler


def _build_vrptw_data(inst):
    """VRPTWInstance → 作者 VRPTW 数据元组、scaler 与 depot_end（口径同 _solve_cvrptwlib）。"""
    n = inst.dimension
    coords = _coords_of(inst)
    depot_due = float(inst.depot_window[1])
    # 作者口径：scaler = max(坐标最大值, 仓库截止时间/3)，仓库时窗归一化到 [0, 3]
    scaler = max(float(coords.max()), depot_due / 3.0)
    depot_xy = torch.tensor(coords[0:1], dtype=torch.float32)[None] / scaler      # (1,1,2)
    node_xy = torch.tensor(coords[1:], dtype=torch.float32)[None] / scaler        # (1,n,2)
    node_demand = torch.tensor(
        [[inst.demands[i] for i in range(1, n + 1)]], dtype=torch.float32
    ) / float(inst.capacity)                                                      # (1,n)
    service_time = torch.tensor(
        [[inst.service[i] for i in range(1, n + 1)]], dtype=torch.float32
    ) / scaler                                                                    # (1,n)
    tw_start = torch.tensor(
        [[inst.windows[i][0] for i in range(1, n + 1)]], dtype=torch.float32
    ) / scaler                                                                    # (1,n)
    tw_end = torch.tensor(
        [[inst.windows[i][1] for i in range(1, n + 1)]], dtype=torch.float32
    ) / scaler                                                                    # (1,n)
    depot_end = depot_due / scaler  # C101 → 1236/412 = 3.0（作者默认仓库时窗 [0,3]）
    return (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end), scaler, depot_end


def _rollout(env, model):
    """复刻作者 Tester._test_one_batch 的 POMO rollout；返回负距离奖励 (batch, pomo)。"""
    with torch.no_grad():
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    return reward


def _seq_to_routes(seq, n):
    """作者输出序列 → 路线元组。0=仓库分段；1..n 即客户 id。"""
    routes = []
    cur = []
    for t in seq:
        t = int(t)
        if t == 0:
            if cur:
                routes.append(tuple(cur))
                cur = []
        elif 1 <= t <= n:
            cur.append(t)
    if cur:
        routes.append(tuple(cur))
    return routes


def _finish(seq, n):
    """末位去 0 + 分段 + 防御性覆盖补齐（作者解码必然全覆盖，补齐仅兜底）。"""
    while seq and seq[-1] == 0:
        seq.pop()
    routes = _seq_to_routes(seq, n)
    covered = set()
    for r in routes:
        covered.update(r)
    for c in range(1, n + 1):
        if c not in covered:
            routes.append((c,))
    return tuple(routes)


def solve(inst, objective, seed):
    """标准接口：standard_vrp CVRPInstance / VRPTWInstance → 完整覆盖路线。"""
    torch.manual_seed(int(seed))  # argmax 解码本身确定性；seed 仅固定随机源，接口保持一致

    n = inst.dimension
    assert n == 100, 'MVMoE 权重为 n=100 定长模型，收到 dimension=%d' % n

    model = _get_model()

    if objective == 'cvrp':
        from envs import CVRPEnv  # 作者 envs/__init__.py（仅 torch，自包含）
        data, scaler = _build_cvrp_data(inst)
        env = CVRPEnv(problem_size=n, pomo_size=n, loc_scaler=scaler,
                      device=torch.device('cpu'))
        env.load_problems(1, problems=data, aug_factor=_AUG_FACTOR)  # 8 增广 → batch=8
    elif objective == 'vrptw':
        from envs import VRPTWEnv
        data, scaler, depot_end = _build_vrptw_data(inst)
        env = VRPTWEnv(problem_size=n, pomo_size=n, loc_scaler=scaler,
                       device=torch.device('cpu'))
        # 与作者 Tester._solve_cvrptwlib 一致：env 创建后、load_problems 前设置 depot_end
        env.depot_end = depot_end
        env.load_problems(1, problems=data, aug_factor=_AUG_FACTOR)
    else:
        raise ValueError('MVMoE 仅支持 cvrp/vrptw，收到 objective=%r' % objective)

    reward = _rollout(env, model)            # (8, n) 负距离（缩放单位）
    reward = reward.reshape(_AUG_FACTOR, n)
    best = int(reward.argmax())              # 作者 tester 同款：8 增广 × n pomo 取最优
    best_aug, best_pomo = divmod(best, n)
    seq = [int(t) for t in env.selected_node_list[best_aug, best_pomo].tolist()]
    return _finish(seq, n)
