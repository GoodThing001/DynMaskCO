# -*- coding: utf-8 -*-
"""RRNCO（Son et al., ICLR 2026）RCVRPTW 推理适配（P5 外部方法工作器）。

只读使用作者资产（_maskco_recent5_20261120/assets/RRNCO/，未做任何修改）；本模块是独立适配层。
运行环境：/home/hzeng/envs/cc_compare（rl4co 0.6.0 + torchrl 0.13.3 + tensordict 0.13.0 +
lightning 2.6.5 + torch 2.11.0），CPU 推理，torch.set_num_threads(1)。

支持矩阵：
  * VRPTW: 作者 rcvrptw/epoch_199.ckpt（RCVRPTW 模型）→ OK；
  * CVRP: 作者 checkpoints/ 下仅 rcvrptw/（无 rcvrp/ 权重）→ NotImplementedError（UNSUPPORTED）。

数据口径（按作者 rrnco/envs/rmtvrp 训练管线映射标准 VRPTW 实例）：
  * locs: 仓库+全部客户整体逐轴 min-max 归一化到 [0,1]（作者真实数据管道同款；
    dihedral8 增广的 1-x/1-y 翻折只定义在 [0,1] 域）；
  * distance_matrix: 原始欧氏距离（环境内部再 min-max 归一化，奖励按 min/max 反归一化为真实距离）；
  * duration_matrix: 距离矩阵 / DUR_SCALE ∈ [0,1]（作者 duration min-max 口径）；
  * 时窗/服务时长与 duration 同单位（/DUR_SCALE）；DUR_SCALE = max(最大点距, depot_due/4.6)
    → 时窗 ≤4.6（作者训练 max_time=4.6）且 duration ≤1，两口径均落在训练范围内；
  * demand_linehaul = 需求/容量；vehicle_capacity=1；capacity_original=Q；
    demand_backhaul=0、backhaul_class=1、open_route=False、distance_limit=inf、speed=1。
  * 掩码（作者 RMTVRPEnv.get_action_mask）：时窗到达 + 闭路回仓截止 + 容量，解码内保证。

解码配置（作者 test.py 官方默认）：
  * decode_type=greedy（作者唯一支持档）；num_starts=100；dihedral8 增广（n_aug=8，只增广 locs）；
  * 在 8×100=800 条轨迹中按真实距离（env.get_reward 反归一化）取最优（作者 tester 同款选择）。
  * 确定性：greedy 无采样；DistanceExpert 的 prob 采样由固定 seed 决定；seed 按约定固定。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量/时窗在作者解码掩码内保证）。
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

# ---- 远端路径常量（作者资产只读） -----------------------------------------
_ASSET = '/home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets/RRNCO'
_CKPT_VRPTW = _ASSET + '/checkpoints/rcvrptw/epoch_199.ckpt'
_CKPT_CVRP = _ASSET + '/checkpoints/rcvrp/epoch_199.ckpt'  # 作者未提供（仅标注路径）

# ---- 模块级策略缓存（同一进程多次 solve 只加载一次权重） ------------------
_POLICY = None
_AUTHOR = None


def _import_author():
    """导入作者 rrnco 包并缓存所需对象。

    本适配器模块名即为 rrnco（按外部工作器约定），会占用 sys.modules['rrnco']，
    与作者包同名冲突。因此导入作者包时临时让位（弹出本模块名 → 导入作者包 →
    恢复本模块名）；作者包及其子模块一经导入即常驻 sys.modules，后续引用不受影响。
    """
    global _AUTHOR
    if _AUTHOR is None:
        mine = sys.modules.pop('rrnco', None)  # 本适配器临时让位
        try:
            if _ASSET not in sys.path:
                sys.path.insert(0, _ASSET)
            import rrnco as _author_pkg  # noqa: F401 —— 作者包（只读）
            from rrnco.utils import patch_torchrl_specs
            from rrnco.models import RRNet
            from rrnco.envs.rmtvrp import RMTVRPEnv
            from rrnco.models.utils.transforms import StateAugmentation
            _AUTHOR = dict(patch=patch_torchrl_specs, RRNet=RRNet,
                           RMTVRPEnv=RMTVRPEnv, StateAugmentation=StateAugmentation)
        finally:
            if mine is not None:
                sys.modules['rrnco'] = mine  # 恢复本适配器模块名
    return _AUTHOR


def _get_policy():
    """懒加载作者 RRNet 策略（rcvrptw/epoch_199.ckpt，只读作者代码）。"""
    global _POLICY
    if _POLICY is None:
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗
        author = _import_author()
        author['patch']()  # torchrl 新版本 Spec 类改名，作者官方兼容补丁
        # 作者 test.py 同款加载（Lightning 检查点；strict=False + load_baseline=False）
        model = author['RRNet'].load_from_checkpoint(
            _CKPT_VRPTW, map_location='cpu', strict=False,
            load_baseline=False, weights_only=False)
        _POLICY = model.policy.eval()
    return _POLICY


def solve(inst, objective, seed):
    """标准接口：standard_vrp VRPTWInstance → 完整覆盖路线（客户 id 1..dimension）。"""
    if objective == 'cvrp':
        # 作者资产仅含 rcvrptw/epoch_199.ckpt（checkpoints/ 无 rcvrp/），无 CVRP 权重
        raise NotImplementedError(
            'RRNCO 无 CVRP 权重（作者仅提供 rcvrptw/epoch_199.ckpt），CVRP 登记 UNSUPPORTED')
    if objective != 'vrptw':
        raise ValueError('RRNCO 仅支持 vrptw（RCVRPTW 权重），收到 objective=%r' % objective)

    # 固定随机源：greedy 解码本身确定性；DistanceExpert 的 prob 采样与 multinomial 由 seed 决定
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    n = inst.dimension
    assert n == 100, 'RRNCO 权重为 n=100 定长模型，收到 dimension=%d' % n

    # ---- 原始坐标与距离矩阵（0=仓库、1..n=客户） --------------------------
    coords = np.array([inst.coords[i] for i in range(n + 1)], dtype=np.float64)
    cmin = coords.min(axis=0, keepdims=True)
    cmax = coords.max(axis=0, keepdims=True)
    locs = (coords - cmin) / np.maximum(cmax - cmin, 1e-12)  # 逐轴 min-max → [0,1]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)  # 原始欧氏

    # ---- duration / 时窗 / 服务时长同单位缩放（双口径均落训练域） ----------
    dmax = float(dist.max())
    dur_scale = max(dmax, float(inst.depot_window[1]) / 4.6)
    duration = dist / dur_scale                                        # ∈ [0,1]
    tw = np.zeros((n + 1, 2), dtype=np.float64)
    tw[0] = [0.0, float(inst.depot_window[1]) / dur_scale]             # 仓库时窗 ≤4.6
    for i in range(1, n + 1):
        tw[i] = [float(inst.windows[i][0]) / dur_scale,
                 float(inst.windows[i][1]) / dur_scale]
    service = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        service[i] = float(inst.service[i]) / dur_scale
    demand = np.array([float(inst.demands[i]) for i in range(1, n + 1)]) / float(inst.capacity)

    # ---- 构造作者 RMTVRPEnv 输入 TensorDict --------------------------------
    from tensordict.tensordict import TensorDict
    td = TensorDict({
        'locs': torch.tensor(locs, dtype=torch.float32)[None],          # (1, n+1, 2)
        'distance_matrix': torch.tensor(dist, dtype=torch.float32)[None],
        'duration_matrix': torch.tensor(duration, dtype=torch.float32)[None],
        'demand_linehaul': torch.tensor(demand, dtype=torch.float32)[None],
        'demand_backhaul': torch.zeros(1, n, dtype=torch.float32),
        'backhaul_class': torch.ones(1, 1, dtype=torch.int32),
        'time_windows': torch.tensor(tw, dtype=torch.float32)[None],
        'service_time': torch.tensor(service, dtype=torch.float32)[None],
        'open_route': torch.zeros(1, 1, dtype=torch.bool),
        'distance_limit': torch.full((1, 1), float('inf'), dtype=torch.float32),
        'speed': torch.ones(1, 1, dtype=torch.float32),
        'vehicle_capacity': torch.ones(1, 1, dtype=torch.float32),      # 需求已 /capacity
        'capacity_original': torch.full((1, 1), float(inst.capacity), dtype=torch.float32),
    }, batch_size=[1])

    from rl4co.utils.ops import batchify, unbatchify

    author = _import_author()
    policy = _get_policy()
    env = author['RMTVRPEnv'](check_solution=False, generator_params={'num_loc': n})

    # ---- 作者 test.py 默认：dihedral8 增广（只增广 locs）+ num_starts=100 ----
    augment = author['StateAugmentation'](augment_fn='dihedral8', no_aug_coords=False)
    td_aug = augment(td)                       # batch: 1 → 8
    td_reset = env.reset(td_aug)               # 距离矩阵 min-max 归一化等（作者环境）

    with torch.inference_mode():
        out = policy(td_reset, env, return_actions=True, phase='val',
                     calc_reward=False, num_starts=100)
    actions = out['actions']                   # (800, T) 节点访问序列（0=仓库）

    # ---- 作者 tester 同款选择：按真实距离在 8 增广 × 100 起点中取最优 ------
    td_batch = batchify(td_reset, 100)         # (800, ...)
    real_r, norm_r = env.get_reward(td_batch, actions)  # 真实负距离（反归一化）
    reward = unbatchify(real_r, (8, 100))      # (8, 100)
    best = int(reward.argmax())
    seq = [int(t) for t in actions[best].tolist()]

    # ---- 序列 → 路线（0=仓库分段；末位去 0 + 防御性覆盖补齐） ---------------
    while seq and seq[-1] == 0:
        seq.pop()
    routes = []
    cur = []
    for t in seq:
        if t == 0:
            if cur:
                routes.append(tuple(cur))
                cur = []
        elif 1 <= t <= n:
            cur.append(t)
    if cur:
        routes.append(tuple(cur))
    covered = set()
    for r in routes:
        covered.update(r)
    for c in range(1, n + 1):
        if c not in covered:
            routes.append((c,))
    return tuple(routes)
