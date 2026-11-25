# -*- coding: utf-8 -*-
"""POMO（Kwon et al., NeurIPS 2020）CVRP100 推理适配（P5 外部方法工作器）。

只读使用作者仓库（ext_repos/POMO/NEW_py_ver，上传后未做任何修改）；本模块是独立适配层。

数据口径（与作者 CVRP/CVRProblemDef.get_random_problems 训练域一致）：
  * 坐标：仓库+全部客户整体逐轴 min-max 归一化到 [0,1]
    （作者 8 增广 augment_xy_data_by_8_fold 的 1-x/1-y 翻折只定义在 [0,1] 域）；
  * 需求：demand / capacity（训练时 n=100 用 demand_scaler=50，demand∈[0.01,0.18]；
    X-n101-k25 归一化需求 ≤0.2，与训练域同量级）；
  * 车辆容量：作者 env 的 ninf_mask 强制"剩余载重不足即回仓"，容量约束在解码内保证。

解码配置（作者 test_n100.py 官方默认做法）：
  * env = CVRPEnv(problem_size=100, pomo_size=100)；
  * eval_type='argmax'（确定性贪心），pomo_size=100 条轨迹并行；
  * aug_factor=8（作者 8 个增广坐标方向）取最优，即 8×100=800 条轨迹取最小距离；
  * 模型权重 = 作者官方 saved_CVRP100_model/checkpoint-30500.pt。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量已在作者解码时保证）。
"""
from __future__ import annotations

import os
import sys
import tempfile

# ---- 远端路径常量（作者仓库只读） ---------------------------------------
_POMO_ROOT = ('/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/'
              'ext_repos/POMO/NEW_py_ver')
_CVRP_DIR = _POMO_ROOT + '/CVRP'          # CVRProblemDef.py 所在
_POMO_DIR = _CVRP_DIR + '/POMO'           # CVRPEnv.py / CVRPModel.py 所在
_CKPT = (_POMO_DIR + '/result/saved_CVRP100_model/checkpoint-30500.pt')

# ---- 作者 test_n100.py 的官方模型超参（与 checkpoint 严格对应） -----------
_MODEL_PARAMS = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',   # 确定性贪心（作者测试档）
}

# ---- 模块级模型缓存（同一进程多次 solve 只加载一次权重） ----------------
_MODEL = None


def _get_model():
    """懒加载作者模型（CVRPModel + checkpoint-30500.pt，只读作者代码）。"""
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗
        # 作者代码导入依赖：CVRPEnv/CVRPModel 在 CVRP/POMO/，CVRProblemDef 在 CVRP/
        for d in (_POMO_DIR, _CVRP_DIR):
            if d not in sys.path:
                sys.path.insert(0, d)
        from CVRPEnv import CVRPEnv as _Env  # noqa: N813 —— 仅确认导入可用
        from CVRPModel import CVRPModel

        model = CVRPModel(**_MODEL_PARAMS)
        checkpoint = torch.load(_CKPT, map_location='cpu')  # 作者 tester 同款加载
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _MODEL = model
    return _MODEL


def _save_saved_problems(inst, seed):
    """standard_vrp CVRPInstance → 作者 use_saved_problems 的临时 .pt 文件。

    作者 env 的 saved problems 格式（见 CVRPEnv.use_saved_problems）：
    {'depot_xy': (batch,1,2), 'node_xy': (batch,problem,2), 'node_demand': (batch,problem)}，
    坐标 [0,1]、需求/容量。写临时文件完全走作者数据加载路径，不改作者代码。
    """
    import numpy as np
    import torch

    n = inst.dimension
    assert n == 100, 'POMO 权重为 CVRP100 定长模型，收到 dimension=%d' % n

    # ---- 坐标归一化：仓库+全部客户整体逐轴 min-max → [0,1] ---------------
    xs = [inst.coords[i][0] for i in range(n + 1)]
    ys = [inst.coords[i][1] for i in range(n + 1)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-9)  # 防除零
    y_span = max(y_max - y_min, 1e-9)

    coords = np.zeros((n + 1, 2), dtype=np.float32)
    for i in range(n + 1):
        coords[i, 0] = (inst.coords[i][0] - x_min) / x_span
        coords[i, 1] = (inst.coords[i][1] - y_min) / y_span

    demand = np.array([inst.demands[i] / float(inst.capacity)
                       for i in range(1, n + 1)], dtype=np.float32)  # 需求/容量

    data = {
        'depot_xy': torch.from_numpy(coords[0:1, :])[None],      # (1, 1, 2)
        'node_xy': torch.from_numpy(coords[1:, :])[None],        # (1, n, 2)
        'node_demand': torch.from_numpy(demand)[None],           # (1, n)
    }
    fd, tmp_path = tempfile.mkstemp(prefix='pomo_input_%d_' % int(seed), suffix='.pt')
    os.close(fd)
    torch.save(data, tmp_path)
    return tmp_path


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


def solve(inst, objective, seed):
    """标准接口：standard_vrp CVRPInstance → 完整覆盖路线（客户 id 1..dimension）。"""
    if objective != 'cvrp':
        raise ValueError('POMO 仅支持 CVRP（作者无 VRPTW 权重），收到 objective=%r' % objective)

    import torch
    torch.manual_seed(int(seed))  # argmax 解码本身确定性；seed 仅固定随机源，接口保持一致

    model = _get_model()
    tmp_path = _save_saved_problems(inst, seed)
    try:
        from CVRPEnv import CVRPEnv

        # ---- 作者官方配置：problem_size=100, pomo_size=100, 8 增广 ----------
        env = CVRPEnv(problem_size=100, pomo_size=100)
        env.use_saved_problems(tmp_path, torch.device('cpu'))
        env.load_problems(batch_size=1, aug_factor=8)  # 8 个增广方向 → batch=8

        # ---- 复刻作者 CVRPTester._test_one_batch 的 rollout ---------------
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

        # ---- 8 增广 × 100 pomo 中取最优轨迹（作者 tester 同款选择） --------
        aug_reward = reward.reshape(8, 1, env.pomo_size)   # (aug, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)         # (aug, batch) 每增广最优
        best_aug = int(max_pomo_reward[:, 0].argmax())     # 最优增广方向
        best_pomo = int(reward[best_aug, :].argmax())      # 该增广内最优 pomo 轨迹

        seq = [int(t) for t in env.selected_node_list[best_aug, best_pomo].tolist()]
        # 序列含 0=仓库、1..n=客户 id；最后一步必为回仓 0
        while seq and seq[-1] == 0:
            seq.pop()
        routes = _seq_to_routes(seq, inst.dimension)
    finally:
        try:
            os.remove(tmp_path)  # 清理临时输入文件
        except OSError:
            pass

    # ---- 防御性覆盖补齐：作者解码必然访问全部客户，此分支仅兜底 -------------
    covered = set()
    for r in routes:
        covered.update(r)
    for c in range(1, inst.dimension + 1):
        if c not in covered:
            routes.append((c,))

    return tuple(routes)
