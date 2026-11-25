# -*- coding: utf-8 -*-
"""CaR（Bi et al., ICLR 2026）CVRP100 推理适配（P5 外部方法工作器）。

只读使用作者仓库（CC_Compare/models/CaR，原字节未改）；本模块是独立适配层。

作者方法概要：POMO 风格构造式交叉注意力模型（SINGLEModel 统一编码器 + POMO 解码器），
使用先训练的 CaR-POMO_100 权重（checkpoint.pt）。本适配只用构造阶段
（checkpoint 为 improve_steps=5 训练，编码器输入维度含 5 维补充特征，
构造前向与作者 Trainer._val_one_batch 完全一致）。

数据口径（与作者 CVRPEnv.get_random_problems / load_dataset 训练域一致）：
  * 坐标：仓库+全部客户整体逐轴 min-max 归一化到 [0,1]
    （作者 8 增广 augment_xy_data_by_8_fold 的 1-x/1-y 翻折只定义在 [0,1] 域）；
  * 需求：demand / capacity（作者 load_dataset 同款口径；训练时 n=100 用
    demand_scaler=50、demand∈[0.02,0.18]，X-n101-k25 归一化需求 ≤0.485 同量级）；
  * 容量：作者 env 以 load=1.0 出发，ninf_mask 强制"剩余载重不足即回仓"，
    容量约束在解码内保证（load 递减为归一化需求）。

解码配置（作者 test.py 官方默认）：
  * SINGLEModel 结构参数与 checkpoint 形状严格对应：embedding_dim=128、encoder 6 层、
    improve_steps=5（编码器输入=坐标3维+补充特征5维）、supplement_feature_dim=5、
    unified_encoder=True、device=cpu；
  * eval_type='argmax'（确定性贪心，test.py 默认），pomo_start=True、
    pomo_size=100 条轨迹并行；
  * aug_factor=8（作者 _val_and_stat 默认 8 个增广坐标方向）取最优，
    即 8×100=800 条轨迹取最小距离；
  * 模型权重 = CC_Compare/checkpoints/CaR/CVRP/CaR-POMO_100/checkpoint.pt。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量已在作者解码时保证）。
"""
from __future__ import annotations

import importlib.util
import os
import sys

# ---- 远端路径常量（作者仓库只读） ---------------------------------------
_AUTHOR_DIR = '/home/hzeng/project/MASKCO-Main/CC_Compare/models/CaR'
_CKPT = ('/home/hzeng/project/MASKCO-Main/CC_Compare/checkpoints/CaR/CVRP/'
         'CaR-POMO_100/checkpoint.pt')

# ---- 作者 test.py 官方默认超参（与 checkpoint 形状严格对应） --------------
_MODEL_PARAMS = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'decoder_layer_num': 1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',          # 确定性贪心（test.py 默认）
    'norm': 'instance',
    'norm_loc': 'norm_last',
    'problem': 'CVRP',
    'use_fast_attention': True,
    'dual_decoder': False,
    'clean_cache': False,
    'gumbel': False,
    'improvement_only': False,
    'problem_size': 100,
    'improvement_method': 'kopt',
    'rm_num': 1,
    'boundary': 0.5,
    'improve_steps': 5,             # checkpoint 训练口径：编码器输入维度 3+5
    'aspect_num': 2,
    'with_infsb_feature': True,
    'supplement_feature_dim': 5,    # CVRP 默认 5
    'with_RNN': True,
    'with_explore_stat_feature': True,
    'k_max': 4,
    'impr_encoder_start_idx': 0,
    'select_top_k': 10,
    'unified_decoder': False,
    'unified_encoder': True,
    'n2s_decoder': False,
    'v_range': 6.0,
    'pip_decoder': False,
    'generate_PI_mask': False,
    'tw_normalize': True,
    'pairwise_merge': ['feature', 'attention'],
    'which_feature': 'both',
    'succ_attention_bias': 1.0,
}

_POMO_SIZE = 100   # POMO 起点数（test.py 默认 pomo_size=100）
_AUG_FACTOR = 8    # 8 增广（作者 _val_and_stat 默认 aug_factor=8）

# ---- 模块级缓存（同一进程多次 solve 只加载一次权重/模块） ----------------
_SINGLE_MODEL = None   # SINGLEModel 类
_CVRP_ENV = None       # CVRPEnv 类
_MODEL = None          # 已加载的 SINGLEModel 实例


def _load_module(name, path):
    """以自定义模块名加载作者 .py 文件（只读执行，不污染 sys.modules 泛名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_author():
    """懒加载作者 SINGLEModel 与 CVRPEnv（只读执行作者代码）。"""
    global _SINGLE_MODEL, _CVRP_ENV
    if _SINGLE_MODEL is None:
        import torch
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗

        # 作者模型/环境内部写死 `from utils import ...`（CaR 顶层 utils.py），
        # 临时把作者 utils 注册为 'utils'，exec 完成后立即恢复原状，
        # 避免污染进程内的 'utils'/'models'/'envs' 泛名（供其它方法共存）。
        prev_utils = sys.modules.get('utils')
        try:
            utils_mod = _load_module('car_author_utils',
                                     os.path.join(_AUTHOR_DIR, 'utils.py'))
            sys.modules['utils'] = utils_mod
            sing = _load_module('car_author_singlemodel',
                                os.path.join(_AUTHOR_DIR, 'models',
                                             'SINGLEModel.py'))
            env = _load_module('car_author_cvrpenv',
                               os.path.join(_AUTHOR_DIR, 'envs',
                                            'CVRPEnv.py'))
            _SINGLE_MODEL = sing.SINGLEModel
            _CVRP_ENV = env.CVRPEnv
        finally:
            if prev_utils is None:
                sys.modules.pop('utils', None)
            else:
                sys.modules['utils'] = prev_utils
    return _SINGLE_MODEL, _CVRP_ENV


def _get_model():
    """懒加载 CaR-POMO_100 模型（只读作者权重，模块级缓存）。"""
    global _MODEL
    if _MODEL is None:
        import torch
        SINGLEModel, _ = _get_author()
        params = dict(_MODEL_PARAMS)
        params['device'] = torch.device('cpu')
        model = SINGLEModel(**params)

        checkpoint = torch.load(_CKPT, map_location='cpu', weights_only=False)
        sd = (checkpoint['model_state_dict']
              if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
              else checkpoint)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            # 多卡训练保存的权重可能带 'module.' 前缀（作者 Trainer 同款兜底）
            sd2 = {k[7:] if k.startswith('module.') else k: v
                   for k, v in sd.items()}
            model.load_state_dict(sd2, strict=True)
        model.eval()
        _MODEL = model
    return _MODEL


def _normalize_coords(inst):
    """standard_vrp 坐标 → 作者 [0,1]^2 归一化坐标（仓库+客户整体逐轴 min-max）。"""
    import numpy as np
    n = inst.dimension
    coords = np.zeros((n + 1, 2), dtype=np.float64)
    for i in range(n + 1):
        coords[i, 0] = inst.coords[i][0]
        coords[i, 1] = inst.coords[i][1]
    lo = coords.min(axis=0)
    span = coords.max(axis=0) - lo
    span = np.maximum(span, 1e-12)  # 防除零
    return (coords - lo) / span


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
        raise ValueError('CaR 仅支持 CVRP（作者无 VRPTW 权重），'
                         '收到 objective=%r' % objective)

    import numpy as np
    import torch
    torch.set_num_threads(1)      # 单核验证
    torch.manual_seed(int(seed))  # argmax 解码本身确定性；seed 仅固定随机源，接口保持一致
    np.random.seed(int(seed))

    n = inst.dimension
    assert n == 100, 'CaR 权重为 CVRP100 定长模型，收到 dimension=%d' % n

    SINGLEModel, CVRPEnv = _get_author()
    model = _get_model()
    model.set_eval_type('argmax')  # test.py 默认 eval_type

    # ---- inst → 作者数据格式（坐标 [0,1]、需求/容量） ----------------------
    scaled = _normalize_coords(inst)                              # (n+1, 2)
    depot_xy = torch.tensor(scaled[0:1], dtype=torch.float32)[None]   # (1, 1, 2)
    node_xy = torch.tensor(scaled[1:], dtype=torch.float32)[None]     # (1, n, 2)
    cap = float(inst.capacity)
    dem = np.array([float(inst.demands[i]) / cap for i in range(1, n + 1)],
                   dtype=np.float32)
    node_demand = torch.from_numpy(dem)[None]                     # (1, n)

    # ---- 作者 env：problem_size=100, pomo_size=100, 8 增广 → batch=8 -------
    env = CVRPEnv(problem_size=n, pomo_size=_POMO_SIZE, k_max=4,
                  device=torch.device('cpu'))
    env.load_problems(batch_size=1, rollout_size=_POMO_SIZE,
                      problems=(depot_xy, node_xy, node_demand),
                      aug_factor=_AUG_FACTOR)

    # ---- 复刻作者 Trainer._val_one_batch 的构造 rollout --------------------
    with torch.no_grad():
        reset_state, _, _ = env.reset()
        state, reward, done = env.pre_step()
        model.pre_forward(reset_state, None)
        while not done:
            selected, prob, _ = model(state, pomo=True)
            state, reward, done, infeasible = env.step(selected)

    # ---- 8 增广 × 100 pomo 中取最短轨迹（作者验证口径：增广取最优） --------
    td = env._get_travel_distance()   # (batch*aug, pomo)：归一化坐标上的距离
    idx = int(td.reshape(-1).argmin())
    seq = env.selected_node_list.reshape(-1,
                                         env.selected_node_list.size(-1))[idx]
    seq = [int(t) for t in seq.tolist()]
    while seq and seq[-1] == 0:       # 作者序列结尾必为回仓 0，去掉
        seq.pop()
    routes = _seq_to_routes(seq, n)

    # ---- 防御性覆盖补齐：作者掩码必然覆盖全部客户，此分支仅兜底 -------------
    covered = set()
    for r in routes:
        covered.update(r)
    for c in range(1, n + 1):
        if c not in covered:
            routes.append((c,))

    return tuple(routes)
