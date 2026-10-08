"""DeepACO（henry-yeh/DeepACO，NeurIPS 2023）A-v1 排序 provider。

把 DeepACO-CVRP（GNN 启发式参数化 + ACO 构造/局部搜索迭代）接到通用
OrderingAcceptReplanner 的 provider 接口：order(sp) -> tuple[int,...]。

官方评测配置（cvrp/test.py）：n_ants=20，t_aco=[1,10,20,30,40,50,100]
（累计 T=100 迭代为官方主档）；模型 pretrained/cvrp/cvrp100.pt（vanilla
DeepACO，不含 cvrp_nls 的 NLS 局部搜索档）。任务指令指定先用 cvrp100.pt。
本诊断批次档位：官方主档 T=100 在 CPU 超 10s/决策预算（n=90 实测 17.5s），
按「官方参数空间内最轻可行档」取 T=30（n=90 实测 6.1s ≤7s 探针线；t_aco
官方网格内）。n=90 全网格实测见 RESULTS.md。

任务口径（诚实零样本跨任务）：
- 原生约束集 = CVRP（容量 + 单一 depot，无 TW/冷链/多温区/在途锚点）。本 provider
  只按 CVRP 求解排序，忽略 TW；真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器 +
  certify_plan 硬认证兜底。
- 输入与官方 utils.gen_instance 一致：坐标 [0,1]（A-v1 原生即 [0,1]），depot 行 0，
  demand 为原始值（官方 1..9 vs 容量 50；A-v1 为 1..3 vs 容量 50 —— 同口径不缩放）。
- 防御：任一客户 demand > capacity 时 ACO 构造会死循环（官方实现无防护）→
  直接 EDD 兜底（A-v1 demand ≤ 3 << 50，不触发）。
- 排序来源：run(T) 后的 shortest_path 删 depot 后首次出现序；未覆盖客户 EDD 追加
  （防御，不触发）。
- 性能：同一决策内 15 辆 replan 车共享同一 pool，按内容 memoize 只跑一次 ACO。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_DACO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/DeepACO
_DACO_CVRP = os.path.join(_DACO_ROOT, 'cvrp')
for _p in (_DACO_CVRP,):
    if _p not in sys.path:
        sys.path.append(_p)   # 追加到末尾，避免遮蔽

N_ANTS = 20          # 官方 cvrp/test.py
T_ITER = 30          # 官方 t_aco 网格 [1,10,20,30,40,50,100] 内最大 ≤7s 档（n=90 无竞争探针：
                     # T=100→17.5s / T=50→9.3s / T=40→7.5s / T=30→6.1s / T=20→5.9s / T=10→2.9s，
                     # 服务器负载 ~200/96 下实测中位数；正式批次由主线按探针定档）
CAPACITY = 50        # 官方 cvrp/aco.py CAPACITY


class DeepACOProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = device
        self.num_loc = int(num_loc)
        self.n_ants = N_ANTS
        self.t_iter = T_ITER
        self._net = None
        self._cache = {}
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_edd_fallback = 0

    # ------------------------------------------------------------------ #
    def _load(self):
        """懒加载模型（只加载不推理；driver 在 10s 决策预算外先调用一次）。"""
        import torch
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        from net import Net
        net = Net().to(self.device)
        net.load_state_dict(torch.load(self.checkpoint_path,
                                       map_location=self.device))
        net.eval()
        self._net = net

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_key(sp):
        pool = tuple(int(c) for c in sp.pool_customer_ids)
        nodes = [0] + list(pool)
        payload = [str(pool),
                   str(tuple(sp.coords[sp.node_index(c)] for c in nodes)),
                   str(tuple(sp.demands[sp.node_index(c)] for c in nodes)),
                   str(float(sp.capacity))]
        return hashlib.sha256('|'.join(payload).encode('utf-8')).hexdigest()

    def _solve(self, sp, pool):
        import torch
        from utils import gen_pyg_data
        from aco import ACO
        idx = {c: sp.node_index(c) for c in pool}
        n = 1 + len(pool)
        locs = torch.zeros((n, 2), dtype=torch.float32)
        locs[0] = torch.tensor([float(x) for x in sp.coords[sp.node_index(0)]])
        dems = torch.zeros(n, dtype=torch.float32)
        for k, c in enumerate(pool):
            locs[1 + k] = torch.tensor([float(x) for x in sp.coords[idx[c]]])
            dems[1 + k] = float(sp.demands[idx[c]])
        if float(dems.max()) > float(sp.capacity) + 1e-9:
            return None     # 官方 ACO 对超容量客户死循环 → 防御性放弃（A-v1 不触发）
        distances = torch.norm(locs[:, None] - locs, dim=2, p=2)
        distances[torch.arange(n), torch.arange(n)] = 1e-10
        pyg_data = gen_pyg_data(dems, distances, self.device)
        heu_vec = self._net(pyg_data)
        heu_mat = heu_vec.reshape((n, n)) + 1e-10
        aco = ACO(distances=distances, demand=dems, n_ants=self.n_ants,
                  heuristic=heu_mat, capacity=float(sp.capacity),
                  device=self.device)
        aco.run(self.t_iter)
        if aco.shortest_path is None:
            return None
        seq = [int(t) for t in aco.shortest_path.tolist()]
        order, seen = [], set()
        for t in seq:
            if t == 0 or t in seen or t > len(pool):
                continue
            seen.add(t)
            order.append(pool[t - 1])
        rest = [c for c in pool if c not in order]
        rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))
        return tuple(order + rest)

    def order(self, sp):
        """SubProblem → DeepACO CVRP 排序（真实节点 id，不含 depot/anchor）。"""
        if self._net is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if len(pool) == 0:
            return ()
        key = self._pool_key(sp)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        order = self._solve(sp, pool)
        if order is None:
            self.n_edd_fallback += 1
            order = tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = order
        return order
