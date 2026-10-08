"""Omni-VRP（mktabak/Omni-VRP，ICML 2023）A-v1 排序 provider。

把 Omni-VRP 的 POMO-CVRP 基础模型（instance-norm 零样本档）接到通用
OrderingAcceptReplanner 的 provider 接口：order(sp) -> tuple[int,...]。

官方权重与评测路径（POMO/CVRP/test.py）：
- checkpoint = pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt
  （官方 uniform 分布 CVRP100 基础模型；norm='instance'，零样本档）。
- model_params = 官方 test.py 默认（embedding 128 / encoder 6 层 / head 8 /
  qkv 16 / logit_clipping 10 / eval_type 'argmax'）+ norm='instance'。
- 官方测试配置 pomo_size = problem_size、aug_factor=8（x8 几何增强，官方主档）。
  本臂默认 aug_factor=8；若 CPU 实测超 10s/决策预算，则取官方档
  augmentation_enable=False（aug_factor=1）并记录于 RESULTS.md。

任务口径（诚实零样本跨任务）：
- 官方放出的权重只有 CVRP/TSP 档（POMO-CVRP 为 3 维输入 x,y,demand），
  **无 TW 训练档**；仓库无 TW attribute 编码可加载。本臂按 CVRP 零样本求解
  排序（忽略 TW），真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器 + certify_plan
  硬认证兜底。
- 输入规范化与官方一致：坐标 [0,1]（A-v1 原生即 [0,1]），demand/capacity。
- 排序来源：POMO 全部 rollout（aug × pomo）中取官方 reward（负路程最大者）
  的路线，删 depot 后首次出现序；未覆盖客户 EDD 追加（防御，不触发）。
- 性能：同一决策内 15 辆 replan 车共享同一 pool，按内容 memoize 只解码一次。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_OMNI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/Omni-VRP
_POMO = os.path.join(_OMNI_ROOT, 'POMO')
_POMO_CVRP = os.path.join(_POMO, 'CVRP')
for _p in (_POMO, _POMO_CVRP):
    if _p not in sys.path:
        sys.path.append(_p)

AUG_FACTOR = 8          # 官方 tester_params（augmentation_enable + aug_factor=8）
MAX_POOL_DECODE = 400   # 纯防解码超时（A-v1 pool 上限 ~90，不触发）


class OmniVRPProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = device
        self.num_loc = int(num_loc)
        self.aug_factor = AUG_FACTOR
        self._model = None
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
        from CVRPModel import CVRPModel as Model
        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
            'norm': 'instance',          # checkpoint-…-instance-norm.pt
        }
        model = Model(**model_params)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(self.device).eval()
        self._model = model

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
        from CVRPEnv import CVRPEnv as Env
        n = len(pool)
        env = Env(problem_size=n, pomo_size=n)     # 官方：pomo_size = problem_size
        idx = {c: sp.node_index(c) for c in pool}
        depot_xy = torch.tensor(
            [[[float(x) for x in sp.coords[sp.node_index(0)]]]],
            dtype=torch.float32).to(self.device)
        node_xy = torch.tensor(
            [[[float(x) for x in sp.coords[idx[c]]] for c in pool]],
            dtype=torch.float32).to(self.device)
        node_demand = (torch.tensor(
            [[float(sp.demands[idx[c]]) for c in pool]],
            dtype=torch.float32) / float(sp.capacity)).to(self.device)
        aug = self.aug_factor if self.aug_factor > 1 else 1
        env.load_problems(1, problems=(depot_xy, node_xy, node_demand),
                          aug_factor=aug)
        reset_state, _, _ = env.reset()
        with torch.no_grad():
            self._model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self._model(state)
                state, reward, done = env.step(selected)
        aug_reward = reward.reshape(aug, 1, env.pomo_size)
        # 官方选择：全部 rollout（aug × pomo）中负 reward（= -路程）最小者
        flat = aug_reward.reshape(-1)
        best = int(flat.argmin())
        ai, pi = best // env.pomo_size, best % env.pomo_size
        seq = [int(t) for t in
               env.selected_node_list[ai, pi].tolist()]      # 0=depot, 1..n=pool[k-1]
        order, seen = [], set()
        for t in seq:
            if t == 0 or t in seen or t > n:
                continue
            seen.add(t)
            order.append(pool[t - 1])
        rest = [c for c in pool if c not in order]
        rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))
        return tuple(order + rest)

    def order(self, sp):
        """SubProblem → POMO-CVRP 排序（真实节点 id，不含 depot/anchor）。"""
        if self._model is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if len(pool) == 0:
            return ()
        if len(pool) > MAX_POOL_DECODE:
            self.n_edd_fallback += 1
            return tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
        key = self._pool_key(sp)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        order = self._solve(sp, pool)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = order
        return order
