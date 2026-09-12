"""RRNCO 真实模型后端（R0.5 Stage B，B1）。

把自包含 `SubProblem` 转成 RRNCO `RMTVRPEnv` 的 TensorDict，注入 anchor/time/load，
用 native backhaul（pickup）映射 + pool-only 多起点确定性解码，返回客户 ordering。

关键设计（Stage B 冻结口径）：
  - **native backhaul 映射**（不把 pickup 强塞 linehaul）：
      demand_linehaul        = 0（仅客户维度 n-1；_reset 会 prepend depot）
      demand_backhaul        = customer_demand / capacity（仅客户维度）
      used_capacity_backhaul = current_load / capacity（注入）
      backhaul_class         = 1
      vehicle_capacity       = 1
  - **时间缩放**：s = T_MAX / t_max（t_max = 原始 depot horizon），
      TW/service/duration ← ×s，speed ← 1/s。
  - **normalize=True**：模型内部 distance_matrix 走 RMTVRPEnv 的 min-max 归一化，
      与 checkpoint 训练分布一致；anchor-aware 选 start 与最终评价仍用原始公开距离
      （sp.dist_mat / C0 evaluator），二者解耦。
  - **num_loc=100**：epoch_199.ckpt 在 num_loc=100 上训练；子问题节点数
      N = depot + 可选非 depot anchor + pool，通常 ≤ 训练规模。
  - **visible_prob_sampling_v1**：DistanceExpert.sample_size=25 且 replacement=False，
      N<25 会崩溃；加载后只在本地实例上替换 _sample_indices——同一 inverse-distance
      概率，N>=25 保持 replacement=False，N<25 用 replacement=True；每子问题抽样 seed
      由 (master seed, subproblem hash) 派生，确定性。
  - **状态注入（不进静态 encoder）**：静态 encoder 只读 locs/demand/TW/service/
      distance/duration，不读 current_node/current_time/used_capacity；注入的
      anchor/time/load 经 POMO 强制首步后进入 decoder context 与 action mask：
      reset 后覆盖 current_node=anchor、current_time=ready_time×s、
      used_capacity_backhaul=load/capacity、visited[anchor]=True，首步移动
      anchor→pool start 时用注入 time/load 计算新状态。
  - **PoolStartNodes**：多起点只含可见 pool 客户（排除 depot / anchor / future）。
  - **anchor-aware 距离选 start**：上游 `get_reward` 默认 depot 起算，不能用于
      非 depot anchor；本后端用「anchor→route→depot」纯距离选最佳 start，上游
      reward 只记作诊断。

只负责「子问题 → ordering」；车辆分配 / 路线修复 / 可行性认证仍由 coordinator 与
pickup_certificate 完成。lazy import torch/rl4co/tensordict，checkpoint 只加载一次。

注意：本文件在本地（无 torch/rl4co）不可执行，只做 import 级语法检查；真实验证在
服务器 Stage B（B2/B3）。
"""
import hashlib
import json
import os

from preference import PreferenceProvider

T_MAX = 4.6   # RRNCO RMTVRP 时间窗上界（depot horizon，训练分布）


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def compute_injection(sp, capacity):
    """注入值（纯函数，无 torch）：(time_scaled, load_norm, visited_local)。

    ready_time 已是「anchor 可出发时刻」（committed 车 = committed_finish），故
    current_time = ready_time×s，**不再加 anchor service time**；load 已是 pickup
    后载荷（committed 车 = current_load + anchor_demand 已由 common 层算好），故
    直接 current_load/capacity。
    """
    s = T_MAX / max(float(sp.depot_tw_end), 1e-6)
    time_scaled = float(sp.ready_time) * s
    load_norm = float(sp.current_load) / float(capacity)
    visited_local = [int(sp.anchor_idx)] if int(sp.anchor_idx) != 0 else []
    return time_scaled, load_norm, visited_local


def _visible_prob_sample_indices(distance, N, sample_size, seed):
    """visible_prob_sampling_v1 核心（仅 torch，无 tensordict/rl4co）。

    inverse-distance 概率，N>=sample_size replacement=False，N<sample_size
    replacement=True；用显式 generator 保证逐子问题确定。返回 (B, N, sample_size)。
    """
    import torch
    B = distance.shape[0]
    gen = torch.Generator(device=distance.device)
    gen.manual_seed(int(seed))
    indices = torch.arange(N, device=distance.device)
    processed = distance.clone()
    processed[:, indices, indices] = 1e6
    inverse_distance = 1.0 / (processed + 1e-6)
    probabilities = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True)
    probabilities = probabilities.reshape(B * N, -1)
    replacement = bool(N < sample_size)
    sampled = torch.multinomial(
        probabilities, sample_size, replacement=replacement, generator=gen)
    return sampled.reshape(B, N, sample_size)


class _PoolStartNodes:
    """多起点只返回可见 pool 客户（本地索引），排除 depot / anchor / future。

    由 backend 在每子问题解码前注入当前子问题的 pool 本地索引。
    """

    def __init__(self):
        self.pool_local_ids = None   # 由 backend 每子问题设置

    def __call__(self, td, num_starts, backup_n_starts=None, **kwargs):
        if self.pool_local_ids is None:
            raise RuntimeError('pool_local_ids not configured')
        if int(num_starts) != len(self.pool_local_ids):
            raise ValueError('num_starts does not match pool_local_ids')
        return self._select(td, num_starts)

    def _select(self, td, num_starts):
        import torch
        ids = [int(x) for x in self.pool_local_ids]
        return torch.tensor(ids, dtype=td["current_node"].dtype, device=td.device)

    def get_num_starts(self, td):
        return len(self.pool_local_ids)


class _InjectionRMTVRPEnv:
    """RMTVRPEnv 注入包装：reset 后覆盖 anchor/time/load/visited。

    不 import 上游源码（lazy）；由 backend._load 动态创建 RMTVRPEnv 子类实例，
    避免本地无 torch 时 import 失败。
    """

    def __init__(self, base_env, select_start_nodes_fn):
        self.base_env = base_env
        base_env.select_start_nodes_fn = select_start_nodes_fn
        self.injection = None   # (anchor_local, time_scaled, load_norm, visited_local)

    def set_injection(self, anchor_local, time_scaled, load_norm, visited_local):
        self.injection = (int(anchor_local), float(time_scaled), float(load_norm),
                          tuple(int(x) for x in visited_local))

    # 代理到 base_env（backend 只调用 reset + 需要的属性）
    def reset(self, td):
        td_reset = self.base_env.reset(td)
        if self.injection is not None:
            anchor, time_s, load_norm, visited = self.injection
            td_reset["current_node"] = td_reset["current_node"].new_full(
                td_reset.batch_size, anchor, dtype=td_reset["current_node"].dtype)
            td_reset["current_time"] = td_reset["current_time"].new_full(
                (*td_reset.batch_size, 1), time_s,
                dtype=td_reset["current_time"].dtype)
            td_reset["used_capacity_backhaul"] = (
                td_reset["used_capacity_backhaul"].new_full(
                    (*td_reset.batch_size, 1), load_norm,
                    dtype=td_reset["used_capacity_backhaul"].dtype))
            td_reset["used_capacity_linehaul"] = (
                td_reset["used_capacity_linehaul"].new_full(
                    (*td_reset.batch_size, 1), 0.0,
                    dtype=td_reset["used_capacity_linehaul"].dtype))
            for v in visited:
                td_reset["visited"][..., v] = True
            td_reset.set("action_mask", self.base_env.get_action_mask(td_reset))
            self.injection = None
        return td_reset

    def __getattr__(self, name):
        return getattr(self.base_env, name)


class RRNCOBackend(PreferenceProvider):
    """真实 RRNCO checkpoint 后端：SubProblem -> 客户 ordering。"""

    def __init__(self, ckpt_path, capacity=50.0, device='cuda', seed=0):
        self.ckpt_path = ckpt_path
        self.capacity = float(capacity)
        self.device = device
        self.seed = seed
        self.ckpt_sha256 = sha256_file(ckpt_path)
        self._policy = None
        self._env = None
        self._loaded = False
        self._sample_seed = 0
        self._sample_size = 25
        self._distance_expert = None

    def _load(self):
        """lazy 加载 checkpoint + 构造注入环境（只加载一次）。"""
        if self._loaded:
            return self._policy, self._env
        import torch
        from rrnco.utils import patch_torchrl_specs
        patch_torchrl_specs()
        from rrnco.models import RRNet
        torch.manual_seed(self.seed)

        model = RRNet.load_from_checkpoint(
            self.ckpt_path, map_location='cpu', strict=False,
            load_baseline=False, weights_only=False)
        policy = model.policy.to(self.device).eval()

        # 构造 RMTVRPEnv（normalize=True 与训练一致，num_loc=100 与 checkpoint 一致）
        # + PoolStartNodes
        from rrnco.envs.rmtvrp import RMTVRPEnv
        pool_starts = _PoolStartNodes()
        base_env = RMTVRPEnv(generator_params={'num_loc': 100}, normalize=True,
                             select_start_nodes_fn=pool_starts,
                             check_solution=False)
        env = _InjectionRMTVRPEnv(base_env, pool_starts)
        self._policy = policy
        self._env = env
        self._pool_starts = pool_starts
        self._install_visible_prob_sampling()
        self._loaded = True
        return policy, env

    # ------------------------------------------------------------------
    def _find_distance_expert(self):
        from rrnco.models.env_embeddings.rcvrptw import DistanceExpert
        for m in self._policy.modules():
            if isinstance(m, DistanceExpert):
                return m
        raise RuntimeError('DistanceExpert not found in loaded policy')

    def _install_visible_prob_sampling(self):
        """visible_prob_sampling_v1：N<sample_size 时 replacement=True。

        只在已加载模型实例上替换 DistanceExpert._sample_indices（不改上游源码）：
          - 同一 inverse-distance 概率（仅可见节点）；
          - N >= sample_size：replacement=False（上游原样）；
          - N <  sample_size：replacement=True（避免 multinomial 抽 >N 不重复崩溃）；
          - 每子问题用 (master seed, subproblem hash) 派生的确定 seed。
        未来纳入 compute/control 身份。
        """
        expert = self._find_distance_expert()
        sample_size = int(expert.sample_size)
        self._sample_size = sample_size
        self._distance_expert = expert
        backend = self

        def sample(distance, phase, B, N):
            return _visible_prob_sample_indices(
                distance, N, sample_size, int(backend._sample_seed))

        expert._sample_indices = sample

    def _derive_sample_seed(self, sp):
        material = f'{self.seed}:{sp.canonical_hash()}'
        return int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) % (2 ** 31)

    # ------------------------------------------------------------------
    def _subproblem_to_td(self, sp):
        """自包含 SubProblem -> RMTVRPEnv TensorDict（native backhaul + 时间缩放）。"""
        import torch
        from tensordict import TensorDict
        n = len(sp.node_ids)
        # 局部索引：depot 恒 0；anchor 与 pool 映射到本地索引
        local_idx = {nid: i for i, nid in enumerate(sp.node_ids)}
        # 时间缩放
        t_max = float(sp.depot_tw_end)
        s = T_MAX / max(t_max, 1e-6)
        speed = 1.0 / s

        coords = torch.tensor([[float(x) for x in c] for c in sp.coords],
                              dtype=torch.float32)
        tw = torch.tensor([[float(sp.tw_start[i]) * s, float(sp.tw_end[i]) * s]
                           for i in range(n)], dtype=torch.float32)
        service = torch.tensor([float(sp.service_time[i]) * s for i in range(n)],
                               dtype=torch.float32)
        dist = torch.tensor([[float(x) for x in row] for row in sp.dist_mat],
                            dtype=torch.float32)
        dur = torch.tensor([[float(x) for x in row] for row in sp.travel_mat],
                           dtype=torch.float32) * s
        # native backhaul：demand_linehaul=0，demand_backhaul=demand/capacity。
        # RMTVRPEnv._reset 会 prepend depot 0，故这里只传「仅客户」维度（n-1）。
        customer_demands = [float(sp.demands[i]) / self.capacity for i in range(1, n)]
        demand_backhaul = torch.tensor(customer_demands, dtype=torch.float32)

        td = TensorDict({
            'locs': coords.unsqueeze(0),
            'demand_linehaul': torch.zeros((1, n - 1), dtype=torch.float32),
            'demand_backhaul': demand_backhaul.unsqueeze(0),
            'backhaul_class': torch.full((1, 1), 1, dtype=torch.int32),
            'distance_limit': torch.full((1, 1), float('inf'), dtype=torch.float32),
            'time_windows': tw.unsqueeze(0),
            'service_time': service.unsqueeze(0),
            'vehicle_capacity': torch.ones((1, 1), dtype=torch.float32),
            'capacity_original': torch.full((1, 1), self.capacity, dtype=torch.float32),
            'open_route': torch.zeros((1, 1), dtype=torch.bool),
            'speed': torch.full((1, 1), speed, dtype=torch.float32),
            'distance_matrix': dist.unsqueeze(0),
            'duration_matrix': dur.unsqueeze(0),
        }, batch_size=[1], device=self.device)
        assert td["locs"].shape[-2] == n
        assert td["demand_linehaul"].shape[-1] == n - 1
        assert td["demand_backhaul"].shape[-1] == n - 1
        return td

    def _pool_local_ids(self, sp):
        return [sp.node_index(c) for c in sp.pool_customer_ids]

    def _anchor_aware_distance(self, sp, actions, start_local):
        """anchor→route→depot 纯距离（上游 reward 只作诊断，不能用于非 depot anchor）。"""
        d = 0.0
        cur = sp.anchor_idx
        for a in actions:
            a = int(a)
            if a == 0:      # depot 分隔（多车时出现），重置
                d += float(sp.dist_mat[cur][0])
                cur = 0
                continue
            d += float(sp.dist_mat[cur][a])
            cur = a
        d += float(sp.dist_mat[cur][0])   # 返仓
        return d

    def order(self, sp):
        """返回客户真实 id 的偏好排序（最偏好在前）。"""
        policy, env = self._load()
        import torch

        n = len(sp.node_ids)
        td = self._subproblem_to_td(sp)
        pool_local = self._pool_local_ids(sp)
        if not pool_local:
            return tuple()

        # 逐子问题确定性抽样 seed（visible_prob_sampling_v1）
        self._sample_seed = self._derive_sample_seed(sp)

        # 注入 anchor/time/load（ready_time 已是可出发时刻，不再加 anchor 服务）
        time_scaled, load_norm, visited_local = compute_injection(sp, self.capacity)
        env.set_injection(sp.anchor_idx, time_scaled, load_norm, visited_local)
        self._pool_starts.pool_local_ids = pool_local

        with torch.inference_mode():
            td_reset = env.reset(td)
            assert td_reset["locs"].shape[-2] == n
            assert td_reset["demand_linehaul"].shape[-1] == n
            assert td_reset["demand_backhaul"].shape[-1] == n
            assert td_reset["visited"].shape[-1] == n
            assert td_reset["action_mask"].shape[-1] == n
            out = policy(td_reset, env, return_actions=True, phase='val',
                         calc_reward=False, num_starts=len(pool_local))

        actions = out['actions'].cpu().numpy()   # (num_starts, seq_len)
        if actions.ndim == 1:
            actions = actions[None, :]

        # anchor-aware 选最佳 start
        best = None
        best_dist = float('inf')
        best_actions = None
        for k in range(actions.shape[0]):
            seq = actions[k]
            dd = self._anchor_aware_distance(sp, seq, k)
            if dd < best_dist:
                best_dist = dd
                best = k
                best_actions = seq

        # 首次出现顺序（删除 depot/anchor），映射回真实客户 id
        seen = set()
        order = []
        for a in best_actions:
            a = int(a)
            if a == 0 or a == sp.anchor_idx:
                continue
            nid = sp.node_ids[a]
            if nid == 0 or nid in seen:
                continue
            seen.add(nid)
            order.append(nid)

        # audit hash（供确定性/扰动对账）
        self.last_audit = {
            'checkpoint_sha256': self.ckpt_sha256,
            'subproblem_hash': sp.canonical_hash(),
            'pool_local_ids': list(pool_local),
            'model_distance_normalized': True,
            'start_selection_distance': 'raw_public_distance',
            'evaluation_distance': 'raw_public_distance',
            'distance_sample_size': int(self._sample_size),
            'subproblem_node_count': int(n),
            'sampling_replacement': bool(n < self._sample_size),
            'sampling_seed': int(self._sample_seed),
            'sampling_policy': 'visible_prob_sampling_v1',
            'raw_actions': [[int(x) for x in actions[k]] for k in range(actions.shape[0])],
            'best_start': int(best),
            'best_anchor_aware_distance': float(best_dist),
            'ordering': [int(c) for c in order],
        }
        self.last_audit['audit_hash'] = _json_hash(self.last_audit)
        return tuple(order)
