"""RouteFinder A-v1 排序 provider（P2 批次，2026-09-29）。

把 RouteFinder（TMLR 2025，any-to-any 48 变体基础模型）接到通用
OrderingAcceptReplanner 的 provider 接口：order(sp) -> tuple[int,...]。

API 已按官方 quickstart/test.py 钉死：
  from routefinder.envs import MTVRPEnv
  from routefinder.models import RouteFinderBase
  model = RouteFinderBase.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
  env = MTVRPEnv(); policy = model.policy.to(device).eval()
  td = env.reset(batch); out = policy(td, env, phase="test", decode_type="greedy",
                                      return_actions=True); actions = out["actions"]

规模策略：不补 phantom 行。每次调用用 MTVRPGenerator(num_loc=n_real-1,
variant_preset='vrptw', scale_demand=False) 生成**恰好等于子问题规模**的 TD 并逐行
覆盖（depot+pool，无任何虚节点）。原因（服务器烟测定位）：
  (a) mtvrp 解码终止条件 done=visited.sum==n+1，要求访问全部节点 → phantom 行
      必须可访问，不能靠 demand=0 “忽略”；
  (b) phantom 的 TW 是相对原生 depot 生成的，depot 被移动后部分 phantom 永远
      不可达（arrival>tw_end）→ 解码器永久回 depot → 死循环。
checkpoint 的 encoder/context/init 嵌入全部 shape-agnostic（attention + 按 td 字段
读特征），因此小 n 也能前向；解码在 A-v1 TW/距离下必然可终止（任意客户从 depot
出发 arrival=d/1.5<tw_end，can_reach_depot 在 depot_late=22 下宽松成立）。
pool > 400 时返回 EDD 排序（纯防解码超时；A-v1 pool 上限 ~90，不会触发）。
"""
from __future__ import annotations

import sys
import os

import numpy as np

_RF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/RouteFinder/
if _RF not in sys.path:
    sys.path.insert(0, _RF)


class RouteFinderProvider:
    def __init__(self, checkpoint_path, device='cuda', num_loc=100):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_loc = int(num_loc)
        self._policy = None
        self._env = None
        self.n_calls = 0
        self.n_edd_fallback = 0

    def _load(self):
        import torch
        from routefinder.envs import MTVRPEnv
        from routefinder.models import RouteFinderBase
        # torch>=2.6 默认 weights_only=True；该 checkpoint 内 pickle 了多个可信配置/数据类
        # （MTVRPEnv/TensorDictDataset/omegaconf...）——逐类白名单不可行，进程内临时把
        # torch.load 默认改回 weights_only=False（本地可信资产；只影响本进程）。
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        # checkpoint 训练于旧 torchrl：tensor_specs 旧类名已批量更名 —— 进程内别名桥接
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
            model = RouteFinderBase.load_from_checkpoint(
                self.checkpoint_path, map_location="cpu", strict=False)
        finally:
            torch.load = _orig_load
        self._env = MTVRPEnv()
        self._policy = model.policy.to(self.device).eval()

    def _build_td(self, sp, pool):
        """用 vrptw 生成器生成**恰好 n_real=depot+pool 规模**的 TD 并逐行覆盖全部内容；
        无 phantom 行（解码 done 要求访问所有节点，phantom 会死锁，见模块 docstring）。"""
        import torch
        from routefinder.envs.mtvrp.generator import MTVRPGenerator
        n_real = 1 + len(pool)
        gen = MTVRPGenerator(num_loc=n_real - 1, variant_preset='vrptw',
                             scale_demand=False)
        td = gen(1)
        idx = {c: 1 + k for k, c in enumerate(pool)}
        coords = np.zeros((n_real, 2), dtype=np.float32)
        coords[0] = [float(x) for x in sp.coords[sp.node_index(0)]]
        demand = np.zeros(n_real - 1, dtype=np.float32)
        tw = np.zeros((n_real, 2), dtype=np.float32)
        tw[0] = [0.0, float(sp.depot_tw_end)]
        service = np.zeros(n_real, dtype=np.float32)
        for c in pool:
            i = idx[c]
            j = sp.node_index(c)
            coords[i] = [float(x) for x in sp.coords[j]]
            demand[i - 1] = float(sp.demands[j])
            tw[i] = [float(sp.tw_start[j]), float(sp.tw_end[j])]
            service[i] = float(sp.service_time[j])
        td['locs'][0] = torch.tensor(coords)
        td['demand_linehaul'][0] = torch.tensor(demand)
        td['demand_backhaul'][0] = 0.0
        td['time_windows'][0] = torch.tensor(tw)
        td['service_time'][0] = torch.tensor(service)
        td['vehicle_capacity'][0, 0] = float(sp.capacity)
        td['capacity_original'][0, 0] = float(sp.capacity)
        td['speed'][0, 0] = 1.5    # A-v1 tw_speed = SPEED/KM_PER_UNIT
        td = self._env.reset(td).to(self.device)
        return td

    def order(self, sp):
        """SubProblem → vrptw 访问排序（真实节点 id，不含 depot/anchor）。"""
        import torch
        if self._policy is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if len(pool) > 400 or len(pool) == 0:
            # 超大规模（纯防解码超时；A-v1 不会触发）/空池 → EDD 排序，带原因
            self.n_edd_fallback += 1
            return tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
        td = self._build_td(sp, pool)
        with torch.inference_mode():
            out = self._policy(td, self._env, phase="test",
                               decode_type="greedy", return_actions=True,
                               num_starts=1)   # 单起点（多起点在 CPU 上过慢且无增益证明）
        actions = out["actions"][0].cpu().tolist()   # (seq_len,)
        order = []
        seen = set()
        for a in actions:
            node = int(a)
            if node == 0 or node in seen or node > len(pool):
                continue
            seen.add(node)
            order.append(pool[node - 1])    # td 行 1..n_real-1 → pool 第 k=node-1 位客户
        # 补充未被解码访问的 pool 客户（按 EDD）→ 保证全池排序
        rest = [c for c in pool if c not in order]
        rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))
        return tuple(order + rest)
