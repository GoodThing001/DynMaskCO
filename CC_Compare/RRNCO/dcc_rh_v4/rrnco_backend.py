"""RRNCO 真实模型后端（R0.5 Stage B）。

把自包含 `SubProblem` 转成 RRNCO `RMTVRPEnv` 的 TensorDict，注入 anchor/time/load，
用 native backhaul（pickup）映射 + pool-only 多起点确定性解码，返回客户 ordering。

公开接口（对齐 Stage B 计划文档）：
  - `BackendConfig` — 后端配置（checkpoint/device/seed/T_MAX/num_loc/normalize/sampling）
  - `BackendAudit`  — 后端审计（tensor/injected-state/ordering/reward/device/runtime/抽样口径）
  - `subproblem_to_tensordict(subproblem, device)` — 子问题 → TensorDict（native backhaul）
  - `PoolStartNodes`       — 多起点只含可见 pool 客户（排除 depot/anchor/future）
  - `InjectedRMTVRPEnv`    — reset 后覆盖 anchor/time/load/visited 并重算 action_mask
  - `RRNCOPreferenceProvider.order(subproblem) -> tuple[int, ...]` — 真实 checkpoint 后端

关键设计（Stage B 冻结口径）：
  - **native backhaul 映射**（不把 pickup 强塞 linehaul）：
      demand_linehaul        = 0（仅客户维度 n-1；_reset 会 prepend depot）
      demand_backhaul        = customer_demand / capacity（仅客户维度）
      used_capacity_backhaul = current_load / capacity（注入）
      backhaul_class         = 1
      vehicle_capacity       = 1
  - **时间缩放**：s = T_MAX / depot_tw_end，TW/service/duration ← ×s，speed ← 1/s。
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
      anchor/time/load 经 POMO 强制首步后进入 decoder context 与 action mask。
  - **PoolStartNodes**：多起点只含可见 pool 客户（排除 depot / anchor / future）。
  - **anchor-aware 距离选 start**：上游 `get_reward` 默认 depot 起算，不能用于
      非 depot anchor；本后端用「anchor→route→depot」纯距离选最佳 start，上游
      reward 只记作诊断。

只负责「子问题 → ordering」；车辆分配 / 路线修复 / 可行性认证仍由 coordinator 与
pickup_certificate 完成。lazy import torch/rl4co/tensordict，checkpoint 只加载一次。

注意：本文件在本地（无 torch/rl4co）不可执行，只做 import 级语法检查；真实验证在
服务器 Stage B。
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass

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


# ---------------------------------------------------------------------------
# 配置与审计（计划文档接口）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendConfig:
    """后端配置（冻结）。capacity 不从配置取——每个 SubProblem 自带 capacity。"""
    checkpoint_path: str
    device: str = 'cuda'
    seed: int = 0
    t_max: float = T_MAX
    num_loc: int = 100
    normalize: bool = True
    sampling_policy: str = 'visible_prob_sampling_v1'


@dataclass
class BackendAudit:
    """一次 order() 调用的完整审计（确定性字段，供 B3/B6 对账）。"""
    checkpoint_sha256: str
    subproblem_hash: str
    tensor_hash: str
    injected_state_hash: str
    pool_local_ids: tuple
    starts: tuple
    selected_start: int
    raw_actions: list
    ordering_hash: str
    ordering: tuple
    reward: list                       # 上游 reward（诊断，不用于选 start / 评价）
    device: str
    inference_time_s: float
    # 距离/抽样口径
    model_distance_normalized: bool
    start_selection_distance: str
    evaluation_distance: str
    distance_sample_size: int
    subproblem_node_count: int
    sampling_replacement: bool
    sampling_seed: int
    sampling_policy: str

    def to_dict(self) -> dict:
        return asdict(self)

    def audit_hash(self) -> str:
        return _json_hash(self.to_dict())


# ---------------------------------------------------------------------------
# 纯函数（无 torch，本地可测）
# ---------------------------------------------------------------------------

def compute_injection(sp, capacity):
    """注入值（纯函数）：(time_scaled, load_norm, visited_local)。

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


def _tensordict_hash(td):
    """TensorDict 内容 canonical hash（逐顶层字段 numpy bytes，确定性）。"""
    parts = []
    for k in sorted(td.keys()):
        parts.append(k)
        parts.append(td[k].detach().cpu().numpy().tobytes().hex())
    return hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# TensorDict 转换 + 多起点 + 注入环境
# ---------------------------------------------------------------------------

def subproblem_to_tensordict(sp, device):
    """自包含 SubProblem -> RMTVRPEnv TensorDict（native backhaul + 时间缩放）。"""
    import torch
    from tensordict import TensorDict
    n = len(sp.node_ids)
    capacity = float(sp.capacity)
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
    customer_demands = [float(sp.demands[i]) / capacity for i in range(1, n)]
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
        'capacity_original': torch.full((1, 1), capacity, dtype=torch.float32),
        'open_route': torch.zeros((1, 1), dtype=torch.bool),
        'speed': torch.full((1, 1), speed, dtype=torch.float32),
        'distance_matrix': dist.unsqueeze(0),
        'duration_matrix': dur.unsqueeze(0),
    }, batch_size=[1], device=device)
    if td["locs"].shape[-2] != n:
        raise ValueError(f'locs 维度 {td["locs"].shape[-2]} != n')
    if td["demand_linehaul"].shape[-1] != n - 1 or \
            td["demand_backhaul"].shape[-1] != n - 1:
        raise ValueError('demand 维度 != n-1')
    return td


class PoolStartNodes:
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


class InjectedRMTVRPEnv:
    """RMTVRPEnv 注入包装：reset 后覆盖 anchor/time/load/visited。

    不 import 上游源码（lazy）；由 backend._load 动态创建 RMTVRPEnv 实例，
    避免本地无 torch 时 import 失败。
    """

    def __init__(self, base_env, select_start_nodes_fn):
        self.base_env = base_env
        base_env.select_start_nodes_fn = select_start_nodes_fn
        self.injection = None   # (anchor_local, time_scaled, load_norm, visited_local)

    def set_injection(self, anchor_local, time_scaled, load_norm, visited_local):
        self.injection = (int(anchor_local), float(time_scaled), float(load_norm),
                          tuple(int(x) for x in visited_local))

    def set_tolerance(self, s, capacity):
        """把 base_env.get_action_mask 替换为容差版本（对齐 pickup_certificate 的 1e-6 容差）。

        RRNCO 上游用严格 `<`（env.py:358/364），边界（arrive==tw_end / ret==depot_tw_end）
        会被拒绝，而 pickup_certificate 用 `<= + 1e-6` 接受。仅本地 monkey-patch，
        不改上游源码。
        """
        self._tw_eps = 1e-6 * float(s)
        self._cap_eps = 1e-6 / max(float(capacity), 1e-6)
        holder = self
        self.base_env.get_action_mask = lambda td: holder._tolerant_action_mask(td)

    def _tolerant_action_mask(self, td):
        """RRNCO get_action_mask 的容差版（< → <= + eps，其余逻辑逐行一致）。"""
        import torch
        from rl4co.utils.ops import gather_by_index
        curr_node = td["current_node"]
        b_idx = torch.arange(td.batch_size[0], device=td.device)
        dist_ij = td["distance_matrix"][b_idx, curr_node, :]
        dist_j0 = td["distance_matrix"][:, :, 0]
        dur_ij = td["duration_matrix"][b_idx, curr_node, :]
        dur_j0 = td["duration_matrix"][:, :, 0]
        early_tw, late_tw = td["time_windows"][..., 0], td["time_windows"][..., 1]
        arrival_time = td["current_time"] + dur_ij
        can_reach_customer = arrival_time <= late_tw + self._tw_eps
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + dur_j0
        ) * ~td["open_route"] <= late_tw[..., 0:1] + self._tw_eps
        exceeds_dist_limit = (
            td["current_route_length"] + dist_ij + (dist_j0 * ~td["open_route"])
            > td["distance_limit"]
        )
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"]
            > td["vehicle_capacity"] + self._cap_eps
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"]
            > td["vehicle_capacity"] + self._cap_eps
        )
        linehauls_missing = (
            (td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[..., None]
        is_carrying_backhaul = gather_by_index(
            src=td["demand_backhaul"], idx=td["current_node"], dim=1, squeeze=False
        ) > 0
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing & ~exceeds_cap_linehaul & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))
        cannot_serve_linehaul = (
            td["demand_linehaul"]
            > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)
        can_visit = (
            can_reach_customer & can_reach_depot & meets_demand_constraint
            & ~exceeds_dist_limit & ~td["visited"]
        )
        can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

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


# ---------------------------------------------------------------------------
# 后端 provider
# ---------------------------------------------------------------------------

class RRNCOPreferenceProvider(PreferenceProvider):
    """真实 RRNCO checkpoint 后端：SubProblem -> 客户 ordering。"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.ckpt_sha256 = sha256_file(config.checkpoint_path)
        self._policy = None
        self._env = None
        self._pool_starts = None
        self._loaded = False
        self._sample_seed = 0
        self._sample_size = 25
        self._distance_expert = None
        self.last_audit = None   # BackendAudit

    # ------------------------------------------------------------------
    def _load(self):
        """lazy 加载 checkpoint + 构造注入环境（只加载一次）。"""
        if self._loaded:
            return self._policy, self._env
        import torch
        from rrnco.utils import patch_torchrl_specs
        patch_torchrl_specs()
        from rrnco.models import RRNet
        torch.manual_seed(self.config.seed)

        model = RRNet.load_from_checkpoint(
            self.config.checkpoint_path, map_location='cpu', strict=False,
            load_baseline=False, weights_only=False)
        policy = model.policy.to(self.config.device).eval()

        from rrnco.envs.rmtvrp import RMTVRPEnv
        pool_starts = PoolStartNodes()
        base_env = RMTVRPEnv(generator_params={'num_loc': self.config.num_loc},
                             normalize=self.config.normalize,
                             select_start_nodes_fn=pool_starts,
                             check_solution=False)
        env = InjectedRMTVRPEnv(base_env, pool_starts)
        self._policy = policy
        self._env = env
        self._pool_starts = pool_starts
        self._install_visible_prob_sampling()
        self._loaded = True
        return policy, env

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
        material = f'{self.config.seed}:{sp.canonical_hash()}'
        return int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) % (2 ** 31)

    def _pool_local_ids(self, sp):
        return [sp.node_index(c) for c in sp.pool_customer_ids]

    def _anchor_aware_distance(self, sp, actions):
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

    # ------------------------------------------------------------------
    def order(self, sp):
        """返回客户真实 id 的偏好排序（最偏好在前）。"""
        policy, env = self._load()
        import torch

        n = len(sp.node_ids)
        capacity = float(sp.capacity)
        td = subproblem_to_tensordict(sp, self.config.device)
        td_hash = _tensordict_hash(td)
        pool_local = self._pool_local_ids(sp)
        if not pool_local:
            return tuple()

        # 逐子问题确定性抽样 seed（visible_prob_sampling_v1）
        self._sample_seed = self._derive_sample_seed(sp)

        # 注入 anchor/time/load（ready_time 已是可出发时刻，不再加 anchor 服务）
        # 先安装容差 mask（对齐 pickup_certificate 的 1e-6 边界容差）
        s = T_MAX / max(float(sp.depot_tw_end), 1e-6)
        env.set_tolerance(s, capacity)
        time_scaled, load_norm, visited_local = compute_injection(sp, capacity)
        env.set_injection(sp.anchor_idx, time_scaled, load_norm, visited_local)
        self._pool_starts.pool_local_ids = pool_local

        t0 = time.perf_counter()
        with torch.inference_mode():
            td_reset = env.reset(td)
            if (td_reset["locs"].shape[-2] != n
                    or td_reset["demand_linehaul"].shape[-1] != n
                    or td_reset["demand_backhaul"].shape[-1] != n
                    or td_reset["visited"].shape[-1] != n
                    or td_reset["action_mask"].shape[-1] != n):
                raise ValueError('post-reset 维度不符')
            out = policy(td_reset, env, return_actions=True, phase='val',
                         calc_reward=True, num_starts=len(pool_local))
        inference_time_s = time.perf_counter() - t0

        actions = out['actions'].cpu().numpy()   # (num_starts, seq_len)
        if actions.ndim == 1:
            actions = actions[None, :]
        reward = [float(r) for r in out['reward'].detach().cpu().numpy().reshape(-1)]

        # anchor-aware 选最佳 start（不用上游 reward）
        best = None
        best_dist = float('inf')
        best_actions = None
        for k in range(actions.shape[0]):
            dd = self._anchor_aware_distance(sp, actions[k])
            if dd < best_dist:
                best_dist = dd
                best = k
                best_actions = actions[k]

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

        self.last_audit = BackendAudit(
            checkpoint_sha256=self.ckpt_sha256,
            subproblem_hash=sp.canonical_hash(),
            tensor_hash=td_hash,
            injected_state_hash=_json_hash({'anchor': int(sp.anchor_idx),
                                            'time_scaled': float(time_scaled),
                                            'load_norm': float(load_norm),
                                            'visited': [int(v) for v in visited_local]}),
            pool_local_ids=tuple(pool_local),
            starts=tuple(pool_local),
            selected_start=int(best),
            raw_actions=[[int(x) for x in actions[k]] for k in range(actions.shape[0])],
            ordering_hash=_json_hash([int(c) for c in order]),
            ordering=tuple(int(c) for c in order),
            reward=reward,
            device=self.config.device,
            inference_time_s=float(inference_time_s),
            model_distance_normalized=self.config.normalize,
            start_selection_distance='raw_public_distance',
            evaluation_distance='raw_public_distance',
            distance_sample_size=int(self._sample_size),
            subproblem_node_count=int(n),
            sampling_replacement=bool(n < self._sample_size),
            sampling_seed=int(self._sample_seed),
            sampling_policy=self.config.sampling_policy,
        )
        return tuple(order)
