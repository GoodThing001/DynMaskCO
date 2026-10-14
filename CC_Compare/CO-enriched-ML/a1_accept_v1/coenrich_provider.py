"""CO-enriched-ML（ML-CO）→ A-v1 排序 provider。

方法（arXiv:2304.00789, Baty et al., Transportation Science 2024）
----------------------------------------------------------------
ML-CO = 特征工程 → 学习策略预测「本 epoch 该派哪些请求」（每请求一个 profit 预测）
→ PC-HGS（profit-collecting HGS）在给定 profit 下决定派车集合 + 路由 → 下一 epoch。

A-v1 桥接（本文件）
------------------
接口 = 项目侧统一 provider 约定 `order(sp) -> tuple[int, ...]`（可见池优先级排序），
由 `OrderingAcceptReplanner`（`scripts/evaluation/rrnco_accept_replanner.py`）消费：
排序 → 冻结 dcc_rh_v4 协调器打包多车 suffix → 共享 `certify_plan` 硬认证
（TW / 容量 / C0 能耗预算）→ 决定该揭示事件接单或拒单。

    A-v1 每次揭示 = 原文的一个「epoch」
    pool（可见未服务客户）= 原文 epoch_instance 的请求集合
    学习策略输出 profit(请求)  →  排序优先级（高 profit 优先）
    **CO 路由层**：
      - `--pchgs <exe>`：调用官方 PC-HGS（原方法的 CO 层），取它「决定服务的请求集合
        + 路线顺序」作为排序前缀；这是原机制的真实现调用。
      - 不提供 exe / 调用失败：退化为纯学习排序（fallback 计数如实上报）。
    多车分配、TW/容量/C0 预算一律交给共享协调器 + `certify_plan`（不复刻原文多趟回仓语义）。

与原文的差异（如实标注）
----------------------
1. 原文无 accept/reject（未派请求顺延到后续 epoch）；A-v1 揭示即承诺。本 provider 只产出
   **排序**，接/拒由共享认证决定——即「波次派车」被替换为「逐揭示排序 + 硬可行性认证」。
2. 原文路由是多趟回 depot 的静态 VRPTW；A-v1 是单车 pickup-to-depot + 在途接单，语义不同。
   因此 PC-HGS 只用作**优先级生成器**（哪些请求值得派 + 访问顺序偏好），真实可行性/能耗由
   共享层判定。
3. 特征口径见 `coenrich_features.py`（可见静态实例、小时制时间单位）。
4. 排序只依赖 `sp`（自包含可见子问题），不含任何未来/隐藏信息。

用法（服务器，cc_compare env，CPU）：
    python .../smsoke_coenrich.py            # 烟测
    run_a1_external_accept.py --solver ordering --provider coenrich \
        --provider-ckpt CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts ...
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coenrich_features import compute_features, load_feature_computer  # noqa: E402
from coenrich_model import NumpyMLP  # noqa: E402

_CE_ROOT = os.path.dirname(_HERE)                      # CC_Compare/CO-enriched-ML
_DEFAULT_ARTIFACTS = os.path.join(_HERE, 'artifacts')


def _pick(d, names):
    """artifacts 目录内按候选名取第一个存在的文件（官方权重 / A-v1 重训权重同名异构）。"""
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return os.path.join(d, names[0])


def _load_official_tools():
    """按文件路径加载官方 `evaluation/tools.py`（只读；只用其 JSON 打包函数）。"""
    path = os.path.join(_CE_ROOT, 'evaluation', 'tools.py')
    spec = importlib.util.spec_from_file_location('coenrich_official_tools', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve_pchgs(exe, inst, seconds, seed=1):
    """调用官方 PC-HGS CLI（参数与官方 `tools.solve_pchgs` 完全一致），差别只在
    可执行文件用**绝对路径**：官方实现是 `"./" + executable`，绝对路径会被拼成
    `".//home/..."` 这种非法相对路径；这里显式用绝对路径 + 临时 cwd。"""
    import json
    payload = _TOOLS.json.dumps(inst, cls=_TOOLS.NumpyJSONEncoder)
    cmd = [exe, '-', '-t', str(int(max(seconds, 1))), '-assumeJSONInput', '1',
           '-seed', str(int(seed)), '-veh', '-1', '-quiet', '1',
           '-useWallClockTime', '1']
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.Popen(cmd, cwd=td, stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        out, err = p.communicate(payload, timeout=max(seconds * 4, 10))
    if p.returncode != 0:
        raise RuntimeError('PC-HGS 退出码 %s: %s' % (p.returncode, err[-400:]))
    if not out.strip():
        raise RuntimeError('PC-HGS 无输出: %s' % err[-400:])
    return json.loads(out)


_TOOLS = _load_official_tools()


class COEnrichedMLProvider:
    """ML-CO 预测层 → 可见池优先级排序（A-v1 provider 约定）。"""

    def __init__(self, ckpt, device='cpu', feature_set='dynamicstatic',
                 npz=None, struct=None, pchgs_exe=None, pchgs_seconds=1,
                 pchgs_seed=1, clock=None, edd_fallback_max_pool=400):
        art = str(ckpt) if ckpt else _DEFAULT_ARTIFACTS
        if os.path.isdir(art):
            npz = npz or _pick(art, ('official_weights.npz', 'a1_weights.npz'))
            struct = struct or _pick(art, ('official_model_struct.json',
                                           'a1_model_struct.json'))
        else:                                   # 允许直接给 npz 路径
            npz = npz or art
        if struct is None and npz is not None:
            d = os.path.dirname(os.path.abspath(npz))
            struct = _pick(d, ('official_model_struct.json', 'a1_model_struct.json'))
        self.artifacts = os.path.dirname(os.path.abspath(npz))
        self.device = device                    # 兼容 registry 签名（本实现纯 CPU/numpy）
        self.feature_set = feature_set
        self.pchgs_exe = pchgs_exe or os.environ.get('COENRICH_PCHGS_EXE') or None
        self.pchgs_seconds = int(pchgs_seconds)
        self.pchgs_seed = int(pchgs_seed)
        self.clock = clock
        self.edd_fallback_max_pool = int(edd_fallback_max_pool)
        self._FC = load_feature_computer()
        self.model = NumpyMLP(npz, struct)
        # 计数器（如实上报）
        self.n_calls = 0
        self.n_cache_hit = 0
        self.n_pchgs_calls = 0
        self.n_pchgs_error = 0
        self.n_edd_fallback = 0
        self.n_bad_dim = 0
        self._cache = {}

    # ------------------------------------------------------------------ #
    def _feature_hash(self, sp):
        """可见子问题内容的确定性摘要（缓存键；只用 sp 内字段）。"""
        h = hashlib.sha256()
        h.update(repr(tuple(int(c) for c in sp.pool_customer_ids)).encode())
        h.update(np.asarray(sp.coords, dtype=np.float64).tobytes())
        h.update(np.asarray(sp.demands, dtype=np.float64).tobytes())
        h.update(np.asarray(sp.tw_start, dtype=np.float64).tobytes())
        h.update(np.asarray(sp.tw_end, dtype=np.float64).tobytes())
        h.update(np.asarray(sp.service_time, dtype=np.float64).tobytes())
        h.update(np.asarray(sp.dist_mat, dtype=np.float64).tobytes())
        h.update(np.asarray([float(sp.capacity), float(sp.depot_tw_end)]).tobytes())
        return h.hexdigest()

    def profits(self, sp):
        """可见 pool 每个请求的 ML-CO profit 预测（float32 数组，与 pool 同序）。"""
        X, names = compute_features(sp, clock=self.clock,
                                    feature_set=self.feature_set,
                                    Predictor=self._FC)
        if X.shape[1] != self.model.input_dim:
            self.n_bad_dim += 1
            raise ValueError('特征维度 %d ≠ 官方模型输入 %d（%s）'
                             % (X.shape[1], self.model.input_dim, names))
        pred = self.model.predict(X)            # (1+m,)
        pred = pred.astype(np.float64).copy()
        pred[0] = 0.0                            # depot profit = 0（官方 predict_profits）
        return pred[1:]

    # ------------------------------------------------------------------ #
    def order(self, sp):
        """SubProblem → 可见池排序（真实节点 id；高 profit 优先）。"""
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if not pool:
            return ()
        key = self._feature_hash(sp)
        hit = self._cache.get(key)
        if hit is not None:
            self.n_cache_hit += 1
            return hit

        # 超大池防御（A-v1 池 ≤ ~90，正常不触发）：EDD，带原因计数
        if len(pool) > self.edd_fallback_max_pool:
            self.n_edd_fallback += 1
            ordr = tuple(sorted(pool, key=lambda c: float(
                sp.tw_end[sp.node_index(c)])))
            self._cache[key] = ordr
            return ordr

        prof = self.profits(sp)
        # 纯学习排序（严格降序 profit；并列按 id 保证确定性）
        order_scores = sorted(range(len(pool)),
                              key=lambda k: (-float(prof[k]), pool[k]))
        order = [pool[k] for k in order_scores]

        # 原方法的 CO 层：PC-HGS 在预测 profit 下决定「服务哪些请求 + 访问顺序」
        if self.pchgs_exe and os.path.isfile(self.pchgs_exe):
            try:
                order = self._pchgs_order(sp, prof, order)
                self.n_pchgs_calls += 1
            except Exception as exc:            # noqa: BLE001
                self.n_pchgs_error += 1
                sys.stderr.write('[coenrich] PC-HGS 调用失败，退化为纯学习排序: %s\n'
                                 % exc)

        # 覆盖保证（协调器要求排序 = 池的一个排列）
        assert sorted(order) == sorted(pool), '排序未覆盖全池'
        ordr = tuple(int(c) for c in order)
        self._cache[key] = ordr
        return ordr

    # ------------------------------------------------------------------ #
    def _pchgs_order(self, sp, prof, order_fallback):
        """official `format_pchgs_instance_as_json` 打包 → PC-HGS → 排序。"""
        from coenrich_features import _node_arrays
        pool, nodes, coords, tw, svc, dem, dm = _node_arrays(sp, None)
        inst_dict = {
            'coords': coords,
            'demands': dem,
            'time_windows': tw,
            'service_times': svc,
            'duration_matrix': dm,
            'capacity': float(sp.capacity),
            'request_idx': np.array(nodes, dtype=np.int64),
            'release_times': np.zeros(len(nodes), dtype=np.float64),
        }
        profits = np.zeros(len(nodes), dtype=np.int64)
        profits[1:] = np.rint(prof).astype(np.int64)
        payload = _TOOLS.format_pchgs_instance_as_json(
            None, instance=inst_dict, profits=profits)
        sol = _solve_pchgs(self.pchgs_exe, payload, self.pchgs_seconds,
                           self.pchgs_seed)
        routes = sol.get('routes') or []
        served = []
        for route in routes:
            for r in route.get('requests', []):
                r = int(r)
                if r in pool and r not in served:
                    served.append(r)
        rest = [c for c in order_fallback if c not in served]
        return served + rest

    def stats(self):
        return {
            'n_calls': self.n_calls,
            'n_cache_hit': self.n_cache_hit,
            'n_pchgs_calls': self.n_pchgs_calls,
            'n_pchgs_error': self.n_pchgs_error,
            'n_edd_fallback': self.n_edd_fallback,
            'n_bad_dim': self.n_bad_dim,
            'feature_set': self.feature_set,
            'artifacts': self.artifacts,
            'pchgs_exe': self.pchgs_exe,
        }
