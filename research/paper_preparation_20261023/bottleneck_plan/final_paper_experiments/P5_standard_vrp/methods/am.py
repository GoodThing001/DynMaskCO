# -*- coding: utf-8 -*-
"""AttentionModel（AM，Kool et al., ICLR 2019）CVRP100 推理适配（P5 外部方法工作器）。

只读使用作者仓库（ext_repos/AttentionModel，上传后未做任何修改）；本模块是独立适配层。

数据口径（与作者官方 VRPDataset/make_instance 完全一致）：
  * 坐标：仓库+全部客户整体逐轴 min-max 归一化到 [0,1]（CVRPLIB 坐标 → 作者训练域；
    AM 的 8 个增广方向/距离计算均依赖 [0,1] 输入域）；
  * 需求：demand / capacity（作者 VEHICLE_CAPACITY 恒为 1.0，容量约束由解码 mask 保证）；
  * 车辆容量：作者解码时 mask 强制超载前回仓，输出序列中的仓库 0 即车辆分段点。

解码配置（作者官方 eval.py 的 greedy 档）：
  * model.set_decode_type('greedy') → 每步 argmax，确定性解码（seed 仅设置 torch 随机种子，
    不改变 greedy 结果）；
  * model(batch, return_pi=True) → (cost, logp, pi)，pi 为访问序列：0=仓库、1..n=客户 id；
  * 序列按作者 eval.py 口径处理：尾部仓库 0 弹出，内部 0 作为车辆分段点。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量已在作者解码时保证）。
"""
from __future__ import annotations

import sys

# ---- 远端路径常量（作者仓库只读） ---------------------------------------
_AM_ROOT = '/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/ext_repos/AttentionModel'
_CKPT = _AM_ROOT + '/pretrained/cvrp_100/epoch-99.pt'  # 作者官方 CVRP100 权重（epoch 99）

# ---- 模块级模型缓存（同一进程多次 solve 只加载一次权重） ----------------
_MODEL = None


def _get_model():
    """懒加载作者模型：读 args.json + epoch-99.pt，置 greedy 解码（作者 eval.py 默认档）。"""
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗
        # 作者仓库使用通用包名 utils/nets/problems，必须把仓库根放进 sys.path
        if _AM_ROOT not in sys.path:
            sys.path.insert(0, _AM_ROOT)
        from utils import load_model  # 作者加载器：读取同目录 args.json 与权重
        model, args = load_model(_CKPT)
        model.eval()
        model.set_decode_type('greedy')  # 作者默认 greedy（width=0，无 beam/采样）
        _MODEL = model
    return _MODEL


def _to_batch(inst):
    """standard_vrp 的 CVRPInstance → 作者输入字典（batch=1，坐标 [0,1]，需求/容量）。"""
    import numpy as np
    import torch

    n = inst.dimension
    # ---- 坐标归一化：仓库+全部客户整体逐轴 min-max → [0,1] ---------------
    xs = [inst.coords[i][0] for i in range(n + 1)]
    ys = [inst.coords[i][1] for i in range(n + 1)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-9)  # 防除零（实例坐标不可能全同）
    y_span = max(y_max - y_min, 1e-9)

    coords = np.zeros((n + 1, 2), dtype=np.float32)
    for i in range(n + 1):
        coords[i, 0] = (inst.coords[i][0] - x_min) / x_span
        coords[i, 1] = (inst.coords[i][1] - y_min) / y_span

    # ---- 作者 VRPDataset 键：loc(1,n,2) / demand(1,n) / depot(1,2) -------
    loc = coords[1:, :]                      # 客户坐标（顺序即客户 id 1..n）
    depot = coords[0, :]                     # 仓库坐标
    demand = np.array([inst.demands[i] / float(inst.capacity)
                       for i in range(1, n + 1)], dtype=np.float32)  # 需求/容量

    return {
        'loc': torch.from_numpy(loc)[None],        # (1, n, 2)
        'demand': torch.from_numpy(demand)[None],  # (1, n)
        'depot': torch.from_numpy(depot)[None],    # (1, 2)
    }


def _seq_to_routes(seq, n):
    """作者输出序列 → 路线元组。0=仓库分段；客户索引即客户 id（1..n）。"""
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


def solve(inst, objective, seed):
    """标准接口：standard_vrp CVRPInstance → 完整覆盖路线（客户 id 1..dimension）。"""
    if objective != 'cvrp':
        raise ValueError('AM 仅支持 CVRP（作者无 VRPTW 权重），收到 objective=%r' % objective)

    import torch
    torch.manual_seed(int(seed))  # greedy 解码本身确定性；seed 仅固定随机源，接口保持一致

    model = _get_model()
    batch = _to_batch(inst)

    # ---- 作者 greedy 前向 --------------------------------------------------
    with torch.no_grad():
        _cost, _logp, pi = model(batch, return_pi=True)
    seq = [int(t) for t in pi[0].tolist()]  # 0=仓库、1..n=客户 id（含中途回仓 0）

    # ---- 按作者 eval.py 口径处理：弹出尾部仓库 0，内部 0 即分段点 -----------
    while seq and seq[-1] == 0:
        seq.pop()
    routes = _seq_to_routes(seq, inst.dimension)

    # ---- 防御性覆盖补齐：作者解码必然访问全部客户，此分支仅兜底 -------------
    covered = set()
    for r in routes:
        covered.update(r)
    for c in range(1, inst.dimension + 1):
        if c not in covered:
            routes.append((c,))

    return tuple(routes)
