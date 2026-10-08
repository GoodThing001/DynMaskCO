"""CaDA A-v1 排序 provider（2026-09-30，P2 批次对比臂）。

把 CaDA（CIAM-Group/CaDA，NeurIPS 2024；官方 CVRP100 checkpoint
`100/result/2024-1121-1355/checkpoint-300.pt`）接到通用 OrderingAcceptReplanner
的 provider 接口：order(sp) -> tuple[int,...]（可见池客户真实 id 的偏好排序，
不含 depot/anchor）。

零样本跨任务口径（诚实声明）：
  CaDA 原生只支持 TSP/CVRP 等无时间窗问题；A-v1 子问题是带时间窗/温度区/C0 能耗
  的冷链接单子问题。本 provider 按 **CVRP 求解排序，忽略 TW**：
    - tw_start/tw_end/service_time 不进入模型输入与 action mask（官方 CVRP 约定
      time_windows=[0, inf]、service=0）；
    - 温度区/C0 能耗不在 CaDA 输入空间中，不参与排序。
  真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器（pickup_certificate 逐车 TW/容量
  认证）+ OrderingAcceptReplanner.certify_plan（真实 C0 硬预算）兜底——CaDA 排序
  不满足真实约束时该单被拒（deferred / certify_budget），决策口径与其余外部臂一致。

官方评测配置（复刻 CaDA 100/run.py --test 的贪心解码）：
  - 解码 = 官方 VRPModel.forward 贪心 multi-start（每可见池客户一个起点；官方
    a8gap 8× 增广是外包 best-of-8，单次解码即官方基础档），无采样；
  - CVRP 数据约定照官方：demand/capacity 归一（vehicle_capacity=1）、depot 行 0、
    coords∈[0,1]^2、distance_limit=+inf、time_windows=[0,inf]（encoder nan_to_num
    后特征=0）、service=0、open_route=0（闭路）、speed=1、
    p_s_tag=[1,0,0,0,0,(n-1)/2000]（官方 dataset()/generator 两套约定下 CVRP
    任务码同为 [1,0,0,0,0]）；
  - best-start 选择 = 官方 get_reward 口径（depot→tour→depot 总长最小）；
  - 变长规模：CaDA 为 attention 结构，n=1+anchor?+pool 直接前向，不补 phantom
    （官方 select_start_nodes 按 n 自适应）。
  与官方解码的两处忠实差异（均非自创配置）：
    (a) multi-start 起点 = 可见池客户（与冻结 dcc_rh_v4 PoolStartNodes 口径一致；
        官方起点数为全部客户，此处池外 anchor 已预访问、不作起点）；
    (b) anchor 车辆注入：anchor 行预标记 visited、初始 used_capacity =
        min(sp.current_load, capacity)/capacity（回 depot 清零=卸载，与官方
        env._step 语义一致；RRNCO 后端同款注入）。

解码引擎：torchrl 无关——官方 MTVRPEnv._step / get_action_mask 的 CVRP 子集数学
  在 provider 内逐行复刻（envs/env.py 语义逐式对应），避免 torchrl 0.13 新旧 API
  兼容问题；encoder/decoder 走官方 model.py 原码。解码步数硬上限 6(n+1)+32，
  超限抛 RuntimeError（adapter 记 fallback → 拒单，不静默降级）。

环境：cc_compare env（torch 2.11 + tensordict 0.13 + einops + entmax——entmax 为
  CaDA 官方 requirements.txt 依赖，config use_sparse='topk' 下不被调用，仅 import
  需要）。CPU 推理：_load() 里 torch.set_num_threads(16)（6 worker × 16 = 96 核，
  与 run 脚本 OMP_NUM_THREADS=16 一致，避免小 batch 线程抖动；运行时设置，非模型
  配置）。模型 _load() 由 driver 在 10s 决策预算外先调用一次；order() 内无内部
  超时（决策总时限由 driver 计 timeouts）。
"""
from __future__ import annotations

import os
import sys

_CADA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/CaDA/
for _p in (_CADA_ROOT, os.path.join(_CADA_ROOT, '100')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class CaDAProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_loc = int(num_loc)
        self._model = None
        self._args = None
        self.n_calls = 0
        self.n_decode_fail = 0
        self.n_singleton = 0
        self.total_runtime_s = 0.0
        self.max_pool_seen = 0

    # ------------------------------------------------------------------ #
    # 懒加载（driver 在 10s 决策预算外先调用一次；只加载不推理）
    # ------------------------------------------------------------------ #
    def _load(self):
        if self._model is not None:
            return
        import torch
        import yaml
        # torch>=2.6 默认 weights_only=True；CaDA checkpoint 内含 pickled env/rng
        # 状态（本地可信官方资产）——进程内临时放宽（RouteFinder provider 同款）。
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        # checkpoint 训练于 torchrl 0.1：env_state_dict 里 pickle 了旧 spec 类名，
        # 现代 torchrl 已改名——进程内别名桥接（RouteFinder provider 同款，幂等）。
        try:
            import torchrl.data.tensor_specs as _ts
            _ALIASES = {'CompositeSpec': 'Composite', 'BoundedTensorSpec': 'Bounded',
                        'UnboundedTensorSpec': 'Unbounded',
                        'UnboundedContinuousTensorSpec': 'Unbounded',
                        'BoundedContinuousTensorSpec': 'Bounded',
                        'DiscreteTensorSpec': 'Discrete',
                        'UnboundedDiscreteTensorSpec': 'UnboundedDiscrete',
                        'OneHotDiscreteTensorSpec': 'OneHot',
                        'BinaryDiscreteTensorSpec': 'Binary',
                        'MultiDiscreteTensorSpec': 'MultiDiscrete',
                        'MultiOneHotDiscreteTensorSpec': 'MultiOneHot',
                        'NonTensorSpec': 'NonTensor'}
            for _old, _new in _ALIASES.items():
                if not hasattr(_ts, _old) and hasattr(_ts, _new):
                    setattr(_ts, _old, getattr(_ts, _new))
        except Exception:  # noqa: BLE001
            pass
        try:
            from model import VRPModel  # CaDA 100/model.py（官方原码）
            cfg = yaml.safe_load(
                open(os.path.join(_CADA_ROOT, '100', 'config.yaml'), encoding='utf-8'))
            args = argparse_compat_namespace(cfg, self.num_loc)
            torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
            model = VRPModel(args)
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            model.to(self.device)
            model.eval()
        finally:
            torch.load = _orig_load
        self._model = model
        self._args = args

    # ------------------------------------------------------------------ #
    # SubProblem → 官方 CVRP TensorDict（batch=1）
    # ------------------------------------------------------------------ #
    def _build_td(self, sp, pool):
        """按官方数据约定把 sp 构造成 CVRP 实例（忽略 TW）。

        行序 = sp.node_ids（[0, anchor?, *pool]）；demand 按 sp.capacity 归一；
        anchor 行 demand=0（已预访问、不可调度）；其余字段官方 CVRP 默认。
        """
        import torch
        from tensordict import TensorDict
        n_ids = [int(x) for x in sp.node_ids]
        idx_of = {int(x): i for i, x in enumerate(n_ids)}
        n = len(n_ids)
        anchor_idx = int(sp.anchor_idx)
        cap = max(float(sp.capacity), 1e-6)
        dev = self.device
        locs = torch.zeros((1, n, 2), dtype=torch.float32, device=dev)
        dlh = torch.zeros((1, n), dtype=torch.float32, device=dev)
        for i, nid in enumerate(n_ids):
            j = int(sp.node_index(nid))
            locs[0, i, 0] = float(sp.coords[j][0])
            locs[0, i, 1] = float(sp.coords[j][1])
            if i != 0 and i != anchor_idx and nid in idx_of and nid != 0:
                dlh[0, i] = float(sp.demands[j]) / cap
        td = TensorDict(
            {
                'locs': locs,
                'demand_linehaul': dlh,                     # (1, n) 平铺, depot=0
                'demand_backhaul': torch.zeros((1, n), dtype=torch.float32, device=dev),
                'distance_limit': torch.full((1, 1), float('inf'), dtype=torch.float32,
                                             device=dev),
                'time_windows': torch.zeros((1, n, 2), dtype=torch.float32, device=dev),
                'service_time': torch.zeros((1, n), dtype=torch.float32, device=dev),
                'vehicle_capacity': torch.ones((1, 1), dtype=torch.float32, device=dev),
                'capacity_original': torch.full((1, 1), cap, dtype=torch.float32,
                                                device=dev),
                'open_route': torch.zeros((1, 1), dtype=torch.bool, device=dev),
                'speed': torch.ones((1, 1), dtype=torch.float32, device=dev),
                'p_s_tag': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, (n - 1) / 2000.0]],
                                        dtype=torch.float32, device=dev),
                'current_node': torch.zeros((1,), dtype=torch.long, device=dev),
                'current_time': torch.zeros((1, 1), dtype=torch.float32, device=dev),
                'current_route_length': torch.zeros((1, 1), dtype=torch.float32,
                                                    device=dev),
                'used_capacity_linehaul': torch.zeros((1, 1), dtype=torch.float32,
                                                      device=dev),
                'used_capacity_backhaul': torch.zeros((1, 1), dtype=torch.float32,
                                                      device=dev),
                'visited': torch.zeros((1, n), dtype=torch.bool, device=dev),
            },
            batch_size=[1],
        )
        # 官方 CVRP 时间窗默认 [0, inf]（encoder nan_to_num 后特征=0；mask 需要 inf）
        td['time_windows'][..., 1] = float('inf')
        return td, anchor_idx, idx_of

    # ------------------------------------------------------------------ #
    # 官方解码路径：VRPModel.forward 贪心 multi-start（torchrl-free CVRP step）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_start_nodes(td, starts):
        """官方 select_start_nodes 的池版：起点 = 可见池客户（anchor 不作起点）。"""
        return len(starts), starts

    @staticmethod
    def _gather(src, idx, dim=1, squeeze=True):
        expanded_shape = list(src.shape)
        expanded_shape[dim] = -1
        idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
        squeeze = idx.size(dim) == 1 and squeeze
        return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)

    @staticmethod
    def _distance(x, y):
        return (x - y).norm(p=2, dim=-1)

    def _step(self, td, prev_node, curr_node):
        """MTVRPEnv._step 语义（envs/env.py 逐式复刻，torchrl-free）。"""
        import torch
        prev_loc = self._gather(td['locs'], prev_node)
        curr_loc = self._gather(td['locs'], curr_node)
        distance = self._distance(prev_loc, curr_loc)[..., None]
        service_time = self._gather(src=td['service_time'], idx=curr_node, dim=1,
                                    squeeze=False)
        start_times = self._gather(src=td['time_windows'], idx=curr_node, dim=1,
                                   squeeze=False)[..., 0]
        curr_time = (curr_node[:, None] != 0) * (
            torch.max(td['current_time'] + distance / td['speed'], start_times)
            + service_time)
        curr_route_length = (curr_node[:, None] != 0) * (
            td['current_route_length'] + distance)
        selected_demand_lh = self._gather(td['demand_linehaul'], curr_node, dim=1,
                                          squeeze=False)
        selected_demand_bh = self._gather(td['demand_backhaul'], curr_node, dim=1,
                                          squeeze=False)
        used_cap_lh = (curr_node[:, None] != 0) * (
            td['used_capacity_linehaul'] + selected_demand_lh)
        used_cap_bh = (curr_node[:, None] != 0) * (
            td['used_capacity_backhaul'] + selected_demand_bh)
        visited = td['visited'].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        td.update({
            'current_node': curr_node,
            'current_route_length': curr_route_length,
            'current_time': curr_time,
            'used_capacity_linehaul': used_cap_lh,
            'used_capacity_backhaul': used_cap_bh,
            'visited': visited,
            'done': done,
        })
        td.set('action_mask', self._get_action_mask(td))
        return td

    @staticmethod
    def _get_action_mask(td):
        """MTVRPEnv.get_action_mask 语义（envs/env.py 逐式复刻，torchrl-free）。"""
        import torch
        curr_node = td['current_node']
        locs = td['locs']
        d_ij = CaDAProvider._distance(
            CaDAProvider._gather(locs, curr_node)[..., None, :], locs)
        d_j0 = CaDAProvider._distance(locs, locs[..., 0:1, :])
        early_tw, late_tw = td['time_windows'][..., 0], td['time_windows'][..., 1]
        arrival_time = td['current_time'] + (d_ij / td['speed'])
        can_reach_customer = arrival_time < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td['service_time'] + (d_j0 / td['speed'])
        ) * ~td['open_route'] < late_tw[..., 0:1]
        exceeds_dist_limit = (
            td['current_route_length'] + d_ij + (d_j0 * ~td['open_route'])
            > td['distance_limit'])
        exceeds_cap_linehaul = (
            td['demand_linehaul'] + td['used_capacity_linehaul']
            > td['vehicle_capacity'])
        exceeds_cap_backhaul = (
            td['demand_backhaul'] + td['used_capacity_backhaul']
            > td['vehicle_capacity'])
        linehauls_missing = ((td['demand_linehaul'] * ~td['visited']).sum(-1) > 0)[..., None]
        is_carrying_backhaul = (
            CaDAProvider._gather(src=td['demand_backhaul'], idx=curr_node, dim=1,
                                 squeeze=False) > 0)
        meets_demand_constraint = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td['demand_linehaul'] > 0)
        ) | (~exceeds_cap_backhaul & (td['demand_backhaul'] > 0))
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td['visited']
        )
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    @staticmethod
    def _get_reward(td, actions):
        """官方 MTVRPEnv.get_reward（闭路：depot→tour→depot 总长，reward 取负）。"""
        import torch
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = CaDAProvider._gather(td['locs'], go_from)
        loc_to = CaDAProvider._gather(td['locs'], go_to)
        distances = CaDAProvider._distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td['open_route'])).sum(-1)
        return -tour_length

    def _decode(self, td, anchor_idx, pool_rows, current_load_norm):
        """官方 VRPModel.forward 贪心 multi-start 解码（CVRP step，torchrl-free）。

        返回 (actions (B, T) 长整型, reward (B,))；B = len(pool_rows)。
        超步数上限抛 RuntimeError（adapter 记 fallback → 拒单）。
        """
        import torch
        from model import VRPModel, reshape_by_heads, PrecomputedCache

        model, args = self._model, self._args
        B = len(pool_rows)
        n = td['locs'].shape[-2]
        cap = float(td['capacity_original'][0, 0].item())
        load = min(max(float(current_load_norm), 0.0), cap) / cap

        prompt = model.prompt_net(td)['prompt']                      # (1,1,128)
        node_embed = model.encoder(td, prompt)                       # (1,n,128)

        # ---- batchify 到 num_starts 行（官方 batchify(td, num_starts) 等价）----
        locs_b = td['locs'].expand(B, n, 2).contiguous()
        dlh_b = td['demand_linehaul'].expand(B, n).contiguous()
        dbh_b = td['demand_backhaul'].expand(B, n).contiguous()
        tw_b = td['time_windows'].expand(B, n, 2).contiguous()
        svc_b = td['service_time'].expand(B, n).contiguous()
        dlim_b = td['distance_limit'].expand(B, 1).contiguous()
        vcap_b = td['vehicle_capacity'].expand(B, 1).contiguous()
        open_b = td['open_route'].expand(B, 1).contiguous()
        speed_b = td['speed'].expand(B, 1).contiguous()
        td_b = td.__class__(
            {
                'locs': locs_b,
                'demand_linehaul': dlh_b,
                'demand_backhaul': dbh_b,
                'distance_limit': dlim_b,
                'time_windows': tw_b,
                'service_time': svc_b,
                'vehicle_capacity': vcap_b,
                'open_route': open_b,
                'speed': speed_b,
                'current_node': torch.zeros((B,), dtype=torch.long, device=self.device),
                'current_time': torch.zeros((B, 1), dtype=torch.float32,
                                            device=self.device),
                'current_route_length': torch.zeros((B, 1), dtype=torch.float32,
                                                    device=self.device),
                'used_capacity_linehaul': torch.full((B, 1), load, dtype=torch.float32,
                                                     device=self.device),
                'used_capacity_backhaul': torch.zeros((B, 1), dtype=torch.float32,
                                                      device=self.device),
                'visited': torch.zeros((B, n), dtype=torch.bool, device=self.device),
                'done': torch.zeros((B,), dtype=torch.bool, device=self.device),
            },
            batch_size=[B],
        )
        if anchor_idx != 0:
            td_b['visited'][:, anchor_idx] = True
        td_b.set('action_mask', self._get_action_mask(td_b))

        # ---- 起点：每个可见池客户一个 start（官方 POMO 式）----
        action = torch.tensor(pool_rows, dtype=torch.long, device=self.device)  # (B,)
        prev_node = td_b['current_node'].clone()
        td_b.set('action', action)
        td_b = self._step(td_b, prev_node, action)
        actions_list = [action]

        decoder_k = reshape_by_heads(model.decoder.Wk(node_embed),
                                     head_num=args.model_params['head_num'])
        decoder_v = reshape_by_heads(model.decoder.Wv(node_embed),
                                     head_num=args.model_params['head_num'])
        decoder_single_head_k = node_embed.transpose(1, 2)
        cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)

        step = 0
        step_cap = 6 * (n + 1) + 32
        while not td_b['done'].all():
            logprobs, mask = model.decoder(td_b, cache, B)
            select = VRPModel.greedy(logprobs, mask)
            prev_node = td_b['current_node'].clone()
            td_b.set('action', select)
            actions_list.append(select)
            td_b = self._step(td_b, prev_node, select)
            step += 1
            if step > step_cap:
                raise RuntimeError('CaDA decode step cap exceeded (n=%d)' % n)
        actions = torch.stack(actions_list, 1)                       # (B, T)
        reward = self._get_reward(td_b, actions)
        return actions, reward

    # ------------------------------------------------------------------ #
    # order(sp)
    # ------------------------------------------------------------------ #
    def order(self, sp):
        """SubProblem → 池客户真实 id 偏好排序（最偏好在前；不含 depot/anchor）。"""
        import time
        import torch
        if self._model is None:
            self._load()
        self.n_calls += 1
        t0 = time.perf_counter()
        pool = sorted(int(c) for c in sp.pool_customer_ids)
        self.max_pool_seen = max(self.max_pool_seen, len(pool))
        if not pool:
            self.total_runtime_s += time.perf_counter() - t0
            return tuple()
        # 单客户池快捷路径：官方解码在 num_starts=1 时形状崩溃（unbatchify 把
        # current_node 压成 2-D，与 3-D state_embedding cat 失败——官方评测
        # num_loc≥50 永不触发）；单客户排序被强制为 (c,)，无需解码，直接返回。
        if len(pool) == 1:
            self.n_singleton += 1
            self.total_runtime_s += time.perf_counter() - t0
            return (pool[0],)

        td, anchor_idx, idx_of = self._build_td(sp, pool)
        pool_rows = [idx_of[c] for c in pool]
        current_load = float(getattr(sp, 'current_load', 0.0))
        with torch.inference_mode():
            actions, reward = self._decode(td, anchor_idx, pool_rows,
                                           current_load / max(float(sp.capacity), 1e-6))

        best = int(reward.argmax().item())          # 官方口径：总长最小
        best_actions = actions[best].tolist()
        order, seen = [], set()
        for a in best_actions:
            a = int(a)
            if a == 0 or a == anchor_idx or a in seen:
                continue
            nid = int(sp.node_ids[a])
            if nid == 0 or nid in seen:
                continue
            seen.add(nid)
            order.append(nid)
        # 覆盖保障（解码 done=全节点已访问时必为空；仅防御）
        rest = [c for c in pool if c not in order]
        if rest:
            rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))
            order.extend(rest)
        self.total_runtime_s += time.perf_counter() - t0
        return tuple(order)


def argparse_compat_namespace(cfg, num_loc):
    """镜像 CaDA 100/run.py 的 args 组装（VRPModel 只需要 model_params/env）。"""
    import argparse
    args = argparse.Namespace()
    for k, v in cfg.items():
        setattr(args, k, v)
    args.model_params = dict(cfg['model_params'])
    args.model_params['sqrt_embedding_dim'] = args.model_params['embedding_dim'] ** 0.5
    args.env = dict(cfg['env'])
    args.env['generator_params'] = dict(args.env.get('generator_params', {}))
    args.env['generator_params']['num_loc'] = num_loc
    args.log = print
    return args
