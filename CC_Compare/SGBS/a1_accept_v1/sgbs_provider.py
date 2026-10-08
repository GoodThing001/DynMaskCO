"""SGBS（Simulation-Guided Beam Search，ICLR 2022）CVRP100 A-v1 排序 provider。

官方口径（SGBS/CVRP/2_SGBS，test.py mode='sgbs' +
CVRPTester._test_one_batch_simulation_guided_beam_search）：
  - checkpoint = 1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
    （CVRP100，POMO 架构 + get_expand_prob 扩展）；
  - 算法 = 模型贪心 POMO 多起点 rollout（选 num_starting_points 个最优起点）→
    仿真引导束搜索（SGBS）：每步对每个 beam 取 top-gamma 展开分支做贪心 rollout
    至完成，按 rollout 回报保留 top-beta beam。官方论文 CVRP100 用 beam=1280；
    本适配在官方参数空间（--beta/--gamma/-disable_aug 均官方 test.py 已有）内
    取 10s CPU 预算能放下的最大 beam（RESULTS.md 写明选择与原因）。
  - 输出 = 最优轨迹（aug × beam 中总距离最短者）的客户访问顺序（去 depot）。

超时档位（服务器 CPU 实测，见 RESULTS.md 适配日志）：
  - 官方论文 beta=1280 在 CPU 上不可行；候选 100/50/10（任务指定），仍超限则取
    repo 自带 test.py 默认 beta=4（官方参数空间内的已存在配置）。
  - gamma=4（官方 sgbs_gamma_minus1=3）；num_starting_points=beam_width（官方
    tester 语义）；augmentation 8-fold 或 -disable_aug（均官方 flag）。

零样本跨任务口径（诚实标注，写入 RESULTS.md）：
  - 子问题只取 depot + pool_customer_ids 做 CVRP 求解（坐标、需求、容量），
    忽略 TW / 锚点 / ready_time / 当前载重——原生 SGBS 无 TW；真实 TW / 容量 /
    C0 能耗可行性由下游 dcc_rh_v4 协调器（certify_append）与 certify_plan
    硬认证兜底，不借换目标。
  - 每决策一次真实求解 + 内容寻址 memo：同决策内各车共享同一池，而 CVRP 求解
    不依赖 vehicle_id/锚点 → 同一内容必然同一排序，按池内容缓存（每车 order(sp)
    仍由 adapter 调用，本缓存不改变任何决策，仅省重复计算）。
  - pool 超出官方 CVRP100 域（>100 客户）→ EDD fallback 并计数（防解码超时；
    按工作包估计 A-v1 池上限 ~90，实际不应触发）。
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

_A1 = os.path.dirname(os.path.abspath(__file__))        # CC_Compare/SGBS/a1_accept_v1
_SGBS = os.path.dirname(_A1)                             # CC_Compare/SGBS
_CVRP = os.path.join(_SGBS, 'CVRP')
_CODE = os.path.join(_CVRP, '2_SGBS')
for _p in (_CVRP, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MODEL_PARAMS = {          # test.py model_params（官方；无 eval_type 字段）
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
}

# 最终运行配置（服务器 CPU 基准后锁定；beta 为官方 --beta 参数空间内取值）
_CONFIG = {
    'beta': 10,             # 官方论文 1280 不可行；官方增广 x8 下 n=60 实测 beta=10≈7.8s、
                            # beta=50≈10.9s → 取 10s 预算内最大 beam=10（RESULTS.md）
    'gamma': 4,             # 官方默认 sgbs_gamma_minus1=3 → gamma=4
    'num_aug': 8,           # 官方 augmentation_enable=True（x8 8-fold）
    'num_threads': 16,
}


class SGBSProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100, beta=None,
                 gamma=None, num_aug=None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_loc = int(num_loc)
        self.beta = int(beta) if beta is not None else int(_CONFIG['beta'])
        self.gamma = int(gamma) if gamma is not None else int(_CONFIG['gamma'])
        self.expansion_minus1 = self.gamma - 1
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
        from E_CVRPEnv import E_CVRPEnv
        from CVRPModel import CVRPModel
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        model = CVRPModel(**_MODEL_PARAMS)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        self.model = model
        self._Env = E_CVRPEnv
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

    def _pomo_starting_points(self, env, n):
        """官方 _get_pomo_starting_points：pomo_size=problem_size 全起点贪心 rollout，
        取回报最优的 num_starting_points 个起点（节点 id = 排序名次 + 1）。"""
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = env.step(selected)
        sorted_index = reward.sort(dim=1, descending=True).indices
        return sorted_index

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
        env.load_problems_by_index(0, 1, self.num_aug)
        aug = self.num_aug
        beta = min(self.beta, n)                 # 池小于 beam 时按池取起点（官方 beta<=n 语义不变）

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            # ---- POMO starting points（官方） ----
            sorted_index = self._pomo_starting_points(env, n)
            starting_points = sorted_index[:, :beta] + 1    # depot=0，客户 1..n

            # ---- Simulation-Guided Beam Search（官方流程逐行等价） ----
            env.modify_pomo_size(beta)
            env.reset()
            selected = torch.zeros(size=(aug, beta), dtype=torch.long)
            state, _, done = env.step(selected)
            state, _, done = env.step(starting_points)

            rollout_width = beta * self.expansion_minus1
            rollout_env = copy.deepcopy(env)
            rollout_env.modify_pomo_size(rollout_width)

            first_rollout_flag = True
            while not done:
                probs = self.model.get_expand_prob(state)
                ordered_prob, ordered_i = probs.sort(dim=2, descending=True)
                greedy_next_node = ordered_i[:, :, 0]
                if first_rollout_flag:
                    prob_selected = ordered_prob[:, :, :self.expansion_minus1]
                    idx_selected = ordered_i[:, :, :self.expansion_minus1]
                else:
                    prob_selected = ordered_prob[:, :, 1:self.expansion_minus1 + 1]
                    idx_selected = ordered_i[:, :, 1:self.expansion_minus1 + 1]
                # 小池防御：n+1 < expansion 时官方切片变短（官方 n=100 不触发）；
                # 按官方「冗余替换」机制补齐：补 prob=0 + greedy 节点（无效→被替换）
                k = self.expansion_minus1
                if prob_selected.size(2) < k:
                    pad = k - prob_selected.size(2)
                    prob_selected = torch.cat(
                        [prob_selected, torch.zeros(aug, beta, pad)], dim=2)
                    idx_selected = torch.cat(
                        [idx_selected,
                         greedy_next_node[:, :, None].expand(aug, beta, pad)], dim=2)

                next_nodes = greedy_next_node[:, :, None].repeat(
                    1, 1, self.expansion_minus1)
                is_valid = (prob_selected > 0)
                next_nodes[is_valid] = idx_selected[is_valid]

                rollout_env.reset_by_repeating_bs_env(env, repeat=self.expansion_minus1)
                rollout_env_deepcopy = copy.deepcopy(rollout_env)
                next_nodes = next_nodes.reshape(aug, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
                while not rollout_done:
                    selected_r, _ = self.model(rollout_state)
                    rollout_state, rollout_reward, rollout_done = rollout_env.step(selected_r)

                is_redundant = (~is_valid).reshape(aug, rollout_width)
                rollout_reward[is_redundant] = float('-inf')

                if not first_rollout_flag:
                    rollout_env_deepcopy.merge(env)
                    rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                    next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                first_rollout_flag = False

                sorted_reward, sorted_index2 = rollout_reward.sort(dim=1, descending=True)
                beam_reward = sorted_reward[:, :beta]
                beam_index = sorted_index2[:, :beta]
                env.reset_by_gathering_rollout_env(rollout_env_deepcopy,
                                                   gathering_index=beam_index)
                selected = next_nodes.gather(dim=1, index=beam_index)
                state, reward, done = env.step(selected)

        aug_reward = reward.reshape(aug, 1, beta)
        best = int(aug_reward.argmax())
        a, p = divmod(best, beta)
        seq = env.selected_node_list[a, p].tolist()
        order = [int(pool[k - 1]) for k in seq if k != 0]
        if sorted(order) != sorted(int(c) for c in pool):
            raise RuntimeError('SGBS 解码未覆盖全池（不应发生）')
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
