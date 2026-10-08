"""MVMoE A-v1 排序 provider（P2 批次，2026-09-30）。

把 Routing-MVMoE（ICML 2024）的 POMO 系 VRPTW 单任务模型
（pretrained/pomo_vrptw_n100/epoch-5000.pt）接到通用 OrderingAcceptReplanner
的 provider 接口：order(sp) -> tuple[int, ...]（sp.pool_customer_ids 全池客户
真实 id 排序，最偏好在前；空池返回 ()）。

官方加载与测试 API（已按 Tester.py / test.py / VRPTWEnv.py 钉死）：
  - model_type=SINGLE，checkpoint['problem'] 决定 attr 维度（VRPTW → load+current_time）；
  - 官方模型超参（test.py 默认）：embedding_dim=128, encoder_layer_num=6,
    decoder_layer_num=1, qkv_dim=16, head_num=8, logit_clipping=10,
    ff_hidden_dim=512, eval_type='argmax', norm='instance', norm_loc='norm_last'；
  - 实例构造照官方 _solve_cvrptwlib 缩放约定：
      scaler = max(max_coord, depot_tw_end / 3)；
      坐标 / TW / 服务时间同除 scaler；需求 = raw / capacity；speed = 1；
      depot TW = [0, 3]（= depot_tw_end/scaler）；
  - 解码照官方 POMO rollout：load_problems → reset → pre_forward → pre_step →
    while not done: selected, _ = model(state); env.step(selected)。
    首两步官方强制（depot → START_NODE）；eval_type='argmax' 确定性贪心。

规模策略：变长直接喂子问题规模（depot + pool，无 phantom 行）。依据：
  (a) SINGLE 模型是纯 Transformer（attention + 按输入形状读特征），shape-agnostic；
      官方 CVRPLIB 路径本身就按实例规模 problem_size=n 构造；
  (b) 官方 VRPTWEnv 的 finished 判定要求访问全部节点 → phantom 行必须可访问，
      而 phantom 的 TW 是相对原生 depot 生成的，depot 移动后部分 phantom 可能
      永远不可达 → 解码死循环（与 RouteFinder 臂服务器烟测定位结论一致）。
  解码加步数上限（3×(m+2)，官方全访问解码只需 m+2 步）+ 未访问客户按 EDD 补尾
  （RouteFinder 臂同款防御，preference.validate_ordering 要求精确覆盖全池）→
  保证终止与全池覆盖。A-v1 pool 上限 ~90，EDD 直返档位 120 不会触发。

anchor / ready_time / current_load 不注入（已知局限，写入 RESULTS.md）：
  POMO 解码器无原生起点状态注入（官方 VRPTWEnv 恒 depot 起、满载、t=0）；排序
  契约只要求全池偏好序，真实 anchor 可行性由 dcc_rh_v4 协调器 + certify_suffix +
  C0 certify_plan 兜底（同 OR-Tools/PyVRP 臂的 myopic 语义）。跨车同池缓存
  （同一次决策多车子问题池内容相同）→ 每决策一次解码，只耗一次推理预算。

超参（官方评测空间内；最终档见 RESULTS.md 适配日志）：
  - aug_factor ∈ {1, 8}（官方 CLI 档位），默认 1；
  - pomo multistart（官方 VRPTW 评测口径 pomo_size = problem_size）；
    pomo_size=1 是官方 CLI 允许的轻档（help: "should <= problem size"）。
  CPU 推理：torch.set_num_threads(8)（6 worker × 8 = 48 核，与 96 核共享机无争抢）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

_MVMOE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MVMOE_ROOT not in sys.path:
    sys.path.insert(0, _MVMOE_ROOT)

# 官方模型超参（test.py 默认；SINGLE/VRPTW）
_MODEL_PARAMS = dict(
    embedding_dim=128,
    sqrt_embedding_dim=128 ** 0.5,
    encoder_layer_num=6,
    decoder_layer_num=1,
    qkv_dim=16,
    head_num=8,
    logit_clipping=10,
    ff_hidden_dim=512,
    num_experts=4,
    eval_type='argmax',
    norm='instance',
    norm_loc='norm_last',
    expert_loc=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
    topk=2,
    routing_level='node',
    routing_method='input_choice',
)

_EDD_FALLBACK_POOL = 120   # 纯防解码超时；A-v1 pool 上限 ~90，不触发
_MAX_STEP_FACTOR = 3       # 解码步数上限 = _MAX_STEP_FACTOR * (m + 2)
_CACHE_CAP = 64            # 跨车同池缓存条目上限
_N_THREADS = 8


class MVMoEProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100,
                 aug_factor=1, multistart=True):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_loc = int(num_loc)
        self.aug_factor = int(aug_factor)
        self.multistart = bool(multistart)
        self._model = None
        self._loaded = False
        self._cache = {}
        self._ckpt_problem = None
        self._ckpt_epoch = None
        # 诊断计数（RESULTS.md 引用）
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_edd_fallback = 0
        self.total_decode_s = 0.0

    # ------------------------------------------------------------------ #
    def _load(self):
        """懒加载 checkpoint + 构造模型（只加载不推理；driver 在 10s 决策预算外
        先调用一次，同 JIT 预热排除先例）。幂等。"""
        if self._loaded:
            return
        import torch
        torch.set_num_threads(_N_THREADS)
        ckpt = None
        for weights_only in (True, False):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location='cpu',
                                  weights_only=weights_only)
                break
            except Exception:  # noqa: BLE001
                if not weights_only:
                    raise
        problem = ckpt.get('problem', 'VRPTW')
        params = dict(_MODEL_PARAMS)
        params['problem'] = problem
        params['device'] = torch.device(self.device)
        from models import SINGLEModel
        model = SINGLEModel(**params)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.to(self.device).eval()
        self._model = model
        self._ckpt_problem = problem
        self._ckpt_epoch = ckpt.get('epoch')
        self._loaded = True

    # ------------------------------------------------------------------ #
    def _pool_key(self, sp):
        """跨车同池缓存键：只含池相关字段（anchor/ready_time/load 排除）。"""
        idx = {n: i for i, n in enumerate(sp.node_ids)}
        pool = [int(c) for c in sp.pool_customer_ids]
        payload = {
            'pool': pool,
            'depot': [float(x) for x in sp.coords[sp.node_index(0)]],
            'coords': [[float(x) for x in sp.coords[sp.node_index(c)]]
                       for c in pool],
            'demands': [float(sp.demands[sp.node_index(c)]) for c in pool],
            'tw_start': [float(sp.tw_start[sp.node_index(c)]) for c in pool],
            'tw_end': [float(sp.tw_end[sp.node_index(c)]) for c in pool],
            'service': [float(sp.service_time[sp.node_index(c)]) for c in pool],
            'capacity': float(sp.capacity),
            'depot_tw_end': float(sp.depot_tw_end),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    def _build_data(self, sp, pool):
        """官方 _solve_cvrptwlib 缩放约定 → (depot_xy, node_xy, node_demand,
        service, tw_start, tw_end)，各 (1, ...) 张量（构造在 self.device 上，
        device='cuda' 时同样成立——不依赖官方 test.py 的默认张量类型切换）。"""
        import torch
        i0 = sp.node_index(0)
        depot_c = [float(x) for x in sp.coords[i0]]
        idx = [sp.node_index(c) for c in pool]
        coords = [list(sp.coords[j]) for j in idx]
        max_coord = max([max(depot_c)] + [max(c) for c in coords])
        scaler = max(max_coord, float(sp.depot_tw_end) / 3.0)
        if scaler <= 0.0:
            scaler = 1.0
        cap = float(sp.capacity)
        dev = self.device
        depot_xy = torch.tensor([[depot_c]], dtype=torch.float32,
                                device=dev) / scaler
        node_xy = torch.tensor([coords], dtype=torch.float32,
                               device=dev) / scaler
        node_demand = torch.tensor(
            [[float(sp.demands[j]) / cap for j in idx]], dtype=torch.float32,
            device=dev)
        service = torch.tensor(
            [[float(sp.service_time[j]) / scaler for j in idx]],
            dtype=torch.float32, device=dev)
        tw_start = torch.tensor(
            [[float(sp.tw_start[j]) / scaler for j in idx]],
            dtype=torch.float32, device=dev)
        tw_end = torch.tensor(
            [[float(sp.tw_end[j]) / scaler for j in idx]],
            dtype=torch.float32, device=dev)
        return depot_xy, node_xy, node_demand, service, tw_start, tw_end

    def _decode(self, data):
        """官方 POMO rollout（aug_factor + multistart），返回最佳轨迹的
        客户首次出现序（本地索引 1..m）。终止性：官方 done 或步数上限。

        选轨迹口径：优先全池覆盖，其次最小缩放旅行距离（官方 = 最小距离选最优，
        加覆盖优先是「部分客户在官方 mask 下永久不可达」时的安全扩展）。
        """
        import torch
        from envs import VRPTWEnv
        depot_xy, node_xy, node_demand, service, tw_start, tw_end = data
        m = node_xy.size(1)
        pomo = m if self.multistart else 1
        env = VRPTWEnv(problem_size=m, pomo_size=pomo, loc_scaler=None,
                       device=self.device)
        model = self._model
        model.eval()
        with torch.no_grad():
            env.load_problems(1, problems=(depot_xy, node_xy, node_demand,
                                           service, tw_start, tw_end),
                              aug_factor=self.aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, _, done = env.pre_step()
            max_steps = _MAX_STEP_FACTOR * (m + 2)
            steps = 0
            while not done and steps < max_steps:
                selected, _ = model(state)
                state, _, done = env.step(selected)
                steps += 1
        seqs = env.selected_node_list       # (batch=aug, pomo, L)
        all_xy = env.depot_node_xy          # (batch, m+1, 2)
        best = None
        for b in range(seqs.size(0)):
            for t in range(seqs.size(1)):
                seq = [int(x) for x in seqs[b, t]]
                seen = set()
                order = []
                for x in seq:
                    if x == 0 or x in seen:
                        continue
                    seen.add(x)
                    order.append(x)
                dist = 0.0
                prev = 0
                for x in seq:
                    d = (all_xy[b, x] - all_xy[b, prev]).pow(2).sum().sqrt()
                    dist += float(d)
                    prev = x
                dist += float(
                    (all_xy[b, 0] - all_xy[b, prev]).pow(2).sum().sqrt())
                if best is None or (len(order), -dist) > best[1]:
                    best = (order, (len(order), -dist))
        return best[0]

    # ------------------------------------------------------------------ #
    def order(self, sp):
        """SubProblem → 全池客户真实 id 排序（覆盖精确、无外部/重复；空池 ()）。"""
        if self._model is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if not pool:
            return tuple()
        if len(pool) > _EDD_FALLBACK_POOL:
            self.n_edd_fallback += 1
            return tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
        key = self._pool_key(sp)
        cached = self._cache.get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached
        t0 = time.perf_counter()
        data = self._build_data(sp, pool)
        local_order = self._decode(data)
        self.total_decode_s += time.perf_counter() - t0
        covered_ids = set(pool[x - 1] for x in local_order)
        rest = [c for c in pool if c not in covered_ids]
        rest.sort(key=lambda c: float(sp.tw_end[sp.node_index(c)]))
        ordering = tuple(pool[x - 1] for x in local_order) + tuple(rest)
        if sorted(ordering) != sorted(pool):
            raise ValueError('ordering 未精确覆盖全池（provider bug）')
        if len(self._cache) >= _CACHE_CAP:
            self._cache.clear()
        self._cache[key] = ordering
        return ordering
