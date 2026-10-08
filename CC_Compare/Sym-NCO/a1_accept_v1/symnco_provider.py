"""Sym-NCO（ICLR 2022）CVRP100 A-v1 排序 provider（P3 批次，2026-09-29）。

官方口径（Sym-NCO-POMO/CVRP，test_symnco.py + CVRPTester._test_one_batch，
is_pomo=False → CVRPModel_ours）：
  - checkpoint = pretrained_model/Sym-NCO/checkpoint-8000.pt（CVRP100；
    embedding 128 / 6 层 / 8 头，eval_type='argmax'）；
  - 解码 = Sym-NCO 官方评测流程：首步 depot；次步按「depot 出发贪心概率」排序取
    pomo_size 个不同起点（second_beam=1）；第 3 步对每轨迹取概率 top-1 继续；
    其后 argmax 贪心。评测增广 = 官方等价 8 增广（augment_xy_data_by_N_fold
    随机旋转/翻转 N-fold，N=8；本适配固定 torch seed 保证可复现）。
  - 输出 = 最优轨迹（aug × pomo 中总距离最短者）的客户访问顺序（去 depot）。

超时档位（服务器 CPU 实测，见 RESULTS.md 适配日志）：
  - _CONFIG['num_aug'] = 8 为官方评测档；n=60 池 CPU 单决策 >10s 时取官方代码
    已存在的更轻档 num_aug=1（augmentation_enable=False 等价，非自创配置）。

零样本跨任务口径（诚实标注，写入 RESULTS.md）：
  - 子问题只取 depot + pool_customer_ids 做 CVRP 求解（坐标、需求、容量），
    忽略 TW / 锚点 / ready_time / 当前载重——原生 Sym-NCO 无 TW；真实 TW / 容量 /
    C0 能耗可行性由下游 dcc_rh_v4 协调器（certify_append）与 certify_plan
    硬认证兜底，不借换目标。
  - 每决策一次真实求解 + 内容寻址 memo：同决策内各车共享同一池，而 CVRP 求解
    不依赖 vehicle_id/锚点 → 同一内容必然同一排序，按池内容缓存（每车 order(sp)
    仍由 adapter 调用，本缓存不改变任何决策，仅省重复计算）。
  - pool 超出官方 CVRP100 域（>100 客户）→ EDD fallback 并计数（防解码超时；
    按工作包估计 A-v1 池上限 ~90，实际不应触发）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_A1 = os.path.dirname(os.path.abspath(__file__))        # CC_Compare/Sym-NCO/a1_accept_v1
_SYMNCO = os.path.dirname(_A1)                           # CC_Compare/Sym-NCO
_CVRP = os.path.join(_SYMNCO, 'Sym-NCO-POMO', 'CVRP')
if _CVRP not in sys.path:
    sys.path.insert(0, _CVRP)

_MODEL_PARAMS = {          # test_symnco.py model_params（官方）
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

# 最终运行配置（服务器 CPU 基准后锁定；8=官方评测档，1=官方更轻档）
_CONFIG = {
    'num_aug': 8,   # n=60 CPU 实测 aug=8 见 RESULTS.md；超 10s 则取官方更轻档 aug=1
    'num_threads': 16,
}


class SymNCOProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100, num_aug=None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_loc = int(num_loc)
        self.num_aug = int(num_aug) if num_aug is not None else int(_CONFIG['num_aug'])
        assert self.num_aug in (1, 8), '官方参数空间仅 aug_factor ∈ {1, 8}'
        self.model = None
        self._Env = None
        self._cache = {}
        self.n_calls = 0
        self.n_solves = 0
        self.n_edd_fallback = 0
        self.solve_time_s = 0.0

    def _load(self):
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(int(_CONFIG['num_threads']))
        from CVRPEnv import CVRPEnv
        from CVRPModel_ours import CVRPModel
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        model = CVRPModel(**_MODEL_PARAMS)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        self.model = model
        self._Env = CVRPEnv
        # 官方 N-fold 增广用 torch.rand → 固定 seed 保证跨进程/跨次运行可复现
        torch.manual_seed(0)

    def _content_key(self, sp, pool):
        nodes = [0] + [int(c) for c in pool]
        payload = {
            'pool': [int(c) for c in pool],
            'coords': [[float(x) for x in sp.coords[sp.node_index(n)]] for n in nodes],
            'demands': [float(sp.demands[sp.node_index(n)]) for n in nodes],
            'capacity': float(sp.capacity),
        }
        return json.dumps(payload, sort_keys=True)

    def _solve(self, sp, pool):
        import torch
        n = len(pool)
        cap = max(float(sp.capacity), 1e-9)
        di = sp.node_index(0)
        depot_xy = torch.tensor([[float(x) for x in sp.coords[di]]], dtype=torch.float)
        node_xy = torch.tensor([[float(x) for x in sp.coords[sp.node_index(c)]]
                                for c in pool], dtype=torch.float)[None]
        node_demand = torch.tensor([float(sp.demands[sp.node_index(c)]) / cap
                                    for c in pool], dtype=torch.float)[None]
        env = self._Env(problem_size=n, pomo_size=n)
        env.FLAG__use_saved_problems = True
        env.saved_depot_xy = depot_xy[None]      # (1, 1, 2)
        env.saved_node_xy = node_xy              # (1, n, 2)
        env.saved_node_demand = node_demand      # (1, n)
        env.saved_index = 0
        env.load_problems(1, self.num_aug)
        aug = self.num_aug
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state, is_pomo=True)
                state, reward, done = env.step(selected)
        aug_reward = reward.reshape(aug, 1, n)
        best = int(aug_reward.argmax())
        a, p = divmod(best, n)
        seq = env.selected_node_list[a, p].tolist()
        order = [int(pool[k - 1]) for k in seq if k != 0]
        if sorted(order) != sorted(int(c) for c in pool):
            raise RuntimeError('Sym-NCO 解码未覆盖全池（不应发生）')
        return tuple(order)

    def order(self, sp):
        """SubProblem → 全池客户偏好排序（真实客户 id，精确覆盖 pool）。"""
        if self.model is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if not pool:
            return ()
        if len(pool) > self.num_loc:
            # 超出官方 CVRP100 域 → EDD fallback（防解码超时；按口径不应触发）
            self.n_edd_fallback += 1
            return tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
        key = self._content_key(sp, pool)
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        order = self._solve(sp, pool)
        self.solve_time_s += time.perf_counter() - t0
        self.n_solves += 1
        self._cache[key] = order
        return order
