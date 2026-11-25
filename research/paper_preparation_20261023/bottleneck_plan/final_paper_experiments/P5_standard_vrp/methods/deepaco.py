# -*- coding: utf-8 -*-
"""DeepACO（Ye et al., NeurIPS 2023）CVRP100 推理适配（P5 外部方法工作器）。

只读使用作者仓库（CC_Compare/models/DeepACO/cvrp，原字节未改）；本模块是独立适配层。

作者方法概要：多蚁群 + 图注意力启发式网络（net.py 的 EmbNet 以节点需求、边距离为
特征输出边启发式，aco.py 的 ACO 用 启发式×信息素 采样逐步构造路径，
容量掩码在解码内保证）。

数据口径（与作者 gen_instance / gen_pyg_data 训练域一致）：
  * 坐标：仓库+全部客户整体逐轴 min-max 归一化到 [0,1]
    （作者训练 gen_instance 在 [0,1]^2 均匀采样客户坐标、仓库固定 (0.5,0.5)）；
  * 距离：归一化坐标上的欧氏距离（作者 gen_distance_matrix 同款，对角线 1e-10）；
  * GNN 节点需求特征：作者训练 demand 为 1..9 均匀整数（容量 50 口径），
    CVRPLIB 需求（如 X-n101-k25 为 1..100）直接输入会远超训练域、破坏启发式
    （远端 A/B 实测：原始需求 D≈56.3k，min-max→[1,9] 后 D≈31.7k）；
    故将客户需求逐实例 min-max 映射到 [1,9] 仅作 GNN 节点特征，仓库特征=0；
  * 容量/需求掩码：ACO 仍使用实例真实整数需求与真实容量（作者默认
    CAPACITY=50 仅针对其训练分布），容量掩码保证每条路线可行。

解码配置（作者 test.py 官方默认）：
  * n_ants=20，t_aco=[1,10,20,30,40,50,100]（分段差值在同一 ACO 上累计迭代，
    等价总迭代 T=100）；
  * 模型权重 = CC_Compare/checkpoints/DeepACO/cvrp/cvrp100.pt。

接口：solve(inst, objective, seed) -> tuple(tuple(int, ...), ...)
  返回客户 id 1..inst.dimension 的完整覆盖路线（容量已在作者解码时保证）。
"""
from __future__ import annotations

import importlib.util
import os

# ---- 远端路径常量（作者仓库只读） ---------------------------------------
_AUTHOR_DIR = '/home/hzeng/project/MASKCO-Main/CC_Compare/models/DeepACO/cvrp'
_CKPT = ('/home/hzeng/project/MASKCO-Main/CC_Compare/checkpoints/DeepACO/'
         'cvrp/cvrp100.pt')

# ---- 作者 test.py 官方默认超参 -------------------------------------------
_N_ANTS = 20
_T_ACO = [1, 10, 20, 30, 40, 50, 100]   # 累计迭代节点（作者 test.py 原值）

# ---- 模块级缓存（同一进程多次 solve 只加载一次权重/模块） ----------------
_NET_MOD = None     # net.py 模块（Net）
_ACO_MOD = None     # aco.py 模块（ACO）
_UTILS_MOD = None   # utils.py 模块（gen_pyg_data / gen_distance_matrix）
_NET = None         # 已加载的 Net 实例
_T_ACO_DIFF = None  # 由 _T_ACO 派生的分段迭代数（作者 test.py 口径）


def _load_module(name, path):
    """以自定义模块名加载作者 .py 文件（只读执行，不污染 sys.modules 泛名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_author():
    """懒加载作者模块与模型（只读执行作者代码，绝不修改作者文件）。"""
    global _NET_MOD, _ACO_MOD, _UTILS_MOD, _NET, _T_ACO_DIFF
    if _NET is None:
        import torch
        torch.set_num_threads(1)  # 单核验证：限制线程避免空耗

        _NET_MOD = _load_module('deepaco_author_net',
                                os.path.join(_AUTHOR_DIR, 'net.py'))
        _ACO_MOD = _load_module('deepaco_author_aco',
                                os.path.join(_AUTHOR_DIR, 'aco.py'))
        _UTILS_MOD = _load_module('deepaco_author_utils',
                                  os.path.join(_AUTHOR_DIR, 'utils.py'))

        # 权重加载（作者 test.py 同款：Net + load_state_dict）
        net = _NET_MOD.Net()
        checkpoint = torch.load(_CKPT, map_location='cpu', weights_only=False)
        net.load_state_dict(checkpoint)
        net.eval()
        _NET = net

        # 作者 test.py：_t_aco=[0]+t_aco，相邻差值作为 aco.run() 的迭代数，
        # 在同一个 ACO 实例上累计运行（信息素/最优解跨段保留）。
        _t_aco_full = [0] + list(_T_ACO)
        _T_ACO_DIFF = [_t_aco_full[i + 1] - _t_aco_full[i]
                       for i in range(len(_t_aco_full) - 1)]
    return _NET_MOD, _ACO_MOD, _UTILS_MOD, _NET, _T_ACO_DIFF


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
        raise ValueError('DeepACO 仅支持 CVRP（作者无 VRPTW 权重），'
                         '收到 objective=%r' % objective)

    import numpy as np
    import torch
    torch.set_num_threads(1)          # 单核验证
    torch.manual_seed(int(seed))      # ACO 采样走 torch Categorical，固定 seed 保证确定性
    np.random.seed(int(seed))

    n = inst.dimension
    assert n == 100, 'DeepACO 权重为 CVRP100 定长模型，收到 dimension=%d' % n

    net_mod, aco_mod, utils_mod, net, t_aco_diff = _get_author()

    # ---- inst → 作者数据格式 ---------------------------------------------
    scaled = _normalize_coords(inst)                  # (n+1, 2) ∈ [0,1]^2
    coords_t = torch.tensor(scaled, dtype=torch.float32)
    demands = torch.zeros(n + 1, dtype=torch.float32)  # 仓库需求 0；客户真实整数需求
    for i in range(1, n + 1):
        demands[i] = float(inst.demands[i])
    # GNN 节点需求特征：作者训练 demand 为 1..9 均匀整数，逐实例 min-max 映射到
    # [1,9] 以匹配训练域（仅用于 GNN 特征；ACO 容量掩码仍用真实需求，见下）。
    d_min = float(demands[1:].min())
    d_max = float(demands[1:].max())
    gnn_demands = demands.clone()
    gnn_demands[1:] = (1.0 + 8.0 * (gnn_demands[1:] - d_min)
                       / max(d_max - d_min, 1e-12))
    # 欧氏距离矩阵（归一化坐标口径，对角线 1e-10，作者 gen_distance_matrix 同款）
    distances = utils_mod.gen_distance_matrix(coords_t)

    # ---- 注意力网络前向：预测边启发式 --------------------------------------
    with torch.no_grad():
        pyg_data = utils_mod.gen_pyg_data(gnn_demands, distances, 'cpu')
        heu_vec = net(pyg_data)
    heu_mat = heu_vec.reshape(n + 1, n + 1) + 1e-10   # 作者 test.py 同款 EPS

    # ---- ACO 构造（作者默认 n_ants=20；容量用实例真实值保证可行） ----------
    aco = aco_mod.ACO(
        distances=distances,
        demand=demands,
        n_ants=_N_ANTS,
        heuristic=heu_mat,
        device='cpu',
        capacity=float(inst.capacity),
    )
    for t in t_aco_diff:
        aco.run(t)  # 分段累计迭代到 T=100（作者 test.py 口径）

    # ---- 最短路径 → 路线：路径含 0=仓库（起止均 0），按 0 切分 ---------------
    path = [int(x) for x in aco.shortest_path.tolist()]
    routes = _seq_to_routes(path, n)

    # ---- 防御性覆盖检查：作者掩码保证每个客户恰好一次，此分支仅兜底 ---------
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, n + 1) if c not in covered]
    if missing:
        raise RuntimeError('DeepACO 路径未覆盖客户: %s' % missing[:10])

    return tuple(routes)
