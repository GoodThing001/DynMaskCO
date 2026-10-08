"""AttentionModel（wouterkool/attention-learn-to-route）A-v1 排序 provider。

把 AM（ICLR 2019，AM/encoder-decoder + CVRP100）接到通用 OrderingAcceptReplanner
的 provider 接口：order(sp) -> tuple[int,...]（sp.pool_customer_ids 全池排序）。

官方评测路径：eval.py --decode_strategy greedy（width=0，无 beam/采样），
模型 pretrained/cvrp_100/epoch-99.pt（graph_size=100 训练；encoder/decoder 对任意
规模图可前向）。档位选择：官方论文主结果是 beam search 1280（本任务 CPU 10s/决策
预算内不可行）；官方代码参数空间内同时提供 greedy 档（eval.py 'greedy'，
--eval_batch_size<=max_calc_batch_size），本臂取 greedy（RESULTS.md 记录原因）。

任务口径（诚实零样本跨任务）：
- 原生约束集 = CVRP（容量 + 单一 depot，无 TW/冷链/多温区/在途锚点）。本 provider
  只按 CVRP 求解排序（子问题 = depot + pool），忽略 TW；真实 TW/C0 可行性由下游
  冻结 dcc_rh_v4 协调器 + certify_plan 硬认证兜底。
- 输入规范化与官方 VRPDataset/make_instance 一致：坐标 [0,1]（A-v1 原生即 [0,1]，
  不改），demand/capacity。
- 全池覆盖：解码序列删 depot 后即客户序；防御性把未出现的客户按 EDD 追加
  （官方 decode 必然访问全部客户，追加分支不触发）。
- 性能：15 辆 replan 车的子问题共享同一 pool（provider 只读 pool/coords/demands/
  capacity，不读 anchor），按内容 memoize —— 同一决策只解码一次。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_AM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/AttentionModel
if _AM_ROOT not in sys.path:
    sys.path.append(_AM_ROOT)   # 追加到末尾：utils/nets/problems 为通用名，避免遮蔽


class AMProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = device
        self.num_loc = int(num_loc)
        self._model = None
        self._cache = {}
        self.n_calls = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ #
    def _load(self):
        """懒加载模型（只加载不推理；driver 在 10s 决策预算外先调用一次）。"""
        import torch
        # 小图顺序解码：限制线程数避免 96 核上的小算子线程调度开销（运行期设置，不改算法；
        # 同进程多 provider 时 interop 只允许设一次 → 失败则忽略）
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        from utils import load_model
        model, _args = load_model(self.checkpoint_path)
        model.to(self.device).eval()
        model.set_decode_type('greedy')
        self._model = model

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_key(sp):
        """pool 相关可见内容 → 缓存键（15 辆车共享同一 pool，只解码一次）。"""
        pool = tuple(int(c) for c in sp.pool_customer_ids)
        nodes = [0] + list(pool)
        payload = [str(pool),
                   str(tuple(sp.coords[sp.node_index(c)] for c in nodes)),
                   str(tuple(sp.demands[sp.node_index(c)] for c in nodes)),
                   str(float(sp.capacity))]
        return hashlib.sha256('|'.join(payload).encode('utf-8')).hexdigest()

    def _build_batch(self, sp, pool):
        """CVRP 实例（官方 VRPDataset 格式）：loc (1,n,2) [0,1]、demand (1,n)=d/cap、
        depot (1,2)。"""
        import torch
        coords = np.zeros((1 + len(pool), 2), dtype=np.float32)
        coords[0] = [float(x) for x in sp.coords[sp.node_index(0)]]
        demands = np.zeros(len(pool), dtype=np.float32)
        for k, c in enumerate(pool):
            j = sp.node_index(c)
            coords[1 + k] = [float(x) for x in sp.coords[j]]
            demands[k] = float(sp.demands[j]) / float(sp.capacity)
        return {
            'loc': torch.from_numpy(coords[1:])[None].to(self.device),
            'demand': torch.from_numpy(demands)[None].to(self.device),
            'depot': torch.from_numpy(coords[0])[None].to(self.device),
        }

    def _decode(self, sp, pool):
        import torch
        batch = self._build_batch(sp, pool)
        with torch.no_grad():
            _cost, _ll, pi = self._model(batch, return_pi=True)
        seq = [int(t) for t in pi[0].tolist()]      # 0=depot, 1..n=pool[k-1]
        order, seen = [], set()
        for t in seq:
            if t == 0 or t in seen or t > len(pool):
                continue
            seen.add(t)
            order.append(pool[t - 1])
        rest = [c for c in pool if c not in order]
        rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))   # EDD 防御性追加
        return tuple(order + rest)

    def order(self, sp):
        """SubProblem → CVRP greedy 客户排序（真实节点 id，不含 depot/anchor）。"""
        if self._model is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if len(pool) == 0:
            return ()
        key = self._pool_key(sp)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        order = self._decode(sp, pool)
        if len(self._cache) > 512:      # 防无界增长（40 天 ≈ 8000 决策）
            self._cache.clear()
        self._cache[key] = order
        return order
