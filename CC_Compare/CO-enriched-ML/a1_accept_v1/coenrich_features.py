"""A-v1 可见子问题 → ML-CO 特征矩阵（**复用官方 FeatureComputer，不重写特征公式**）。

角色
----
CO-enriched-ML（arXiv:2304.00789）的预测层输入是 `features/FeatureComputer.py` 算出的
节点特征（featureset-dynamicstatic：11 个 dynamic + 9 个 static = 20 维）。本模块把
A-v1 每辆车的**自包含可见子问题** `SubProblem`（`CC_Compare/RRNCO/dcc_rh_v4/subproblem.py`）
打包成 FeatureComputer 期望的 `(args, observation, instance, static_info)`，然后**逐行调用
官方特征方法**（顺序与官方 `run_feature_computer` 完全一致）。

与原文的差异（必须如实标注）
--------------------------
1. **可见静态实例**：官方 `static` 分位数特征在「全天静态实例」（含未揭示订单）上计算；
   A-v1 是严格因果协议，未揭示订单不可见，因此这里把「静态实例」定义为
   **depot + 当前可见 pool**。同一公式、可见口径 → 无未来信息泄漏（结构上：输入只有 sp）。
2. **时间单位**：官方 `EPOCH_DURATION=3600`（秒/epoch）、`MARGIN_DISPATCH=3600`；A-v1 的
   TW/服务时间/行驶时间单位是**小时**（depot tw_end=22.0，booking horizon=16.0）。因此
   构造后覆盖 `EPOCH_DURATION=1.0`、`MARGIN_DISPATCH=0.0`，`static_info['end_epoch']`
   = depot 时间窗终点（= 天数视界终点）。归一化特征的**定义**不变，只是单位换到 A-v1。
3. **planning_starttime**：SubProblem 无全局时钟字段，取 `sp.ready_time`（该车在当前
   时刻的就绪时间，空转车即当前时钟）作为 planning_starttime 的代理。
4. `must_dispatch`：A-v1 无「必须派车」概念（接/拒由决策层决定），全部置 0。

不改第三方文件：`FeatureComputer.py` 只读导入（importlib 按文件路径加载，避免
`features` / `evaluation` 这类通用包名污染 sys.path）。
"""
from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import numpy as np

_CE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CC_Compare/CO-enriched-ML

# A-v1 时间单位 = 小时；官方 EPOCH_DURATION=3600 s、MARGIN_DISPATCH=3600 s。
A1_EPOCH_DURATION = 1.0
A1_MARGIN_DISPATCH = 0.0

_DYNAMIC_ORDER = ('coords', 'demands', 'serviceTimes', 'timeWindows',
                  'timeDepotRequest',
                  'ratio_timeDepotLocation_timeWindowEndMinusServiceTime',
                  'normalizedTimeWindows', 'mustDispatch')
_STATIC_ORDER = ('quantileTimeToRequests', 'quantileTimeRemainingTimeWindows')


def load_feature_computer():
    """按文件路径加载官方 `features/FeatureComputer.py`（不污染 sys.path）。"""
    path = os.path.join(_CE_ROOT, 'features', 'FeatureComputer.py')
    if not os.path.isfile(path):
        raise FileNotFoundError('官方 FeatureComputer 缺失: %s' % path)
    spec = importlib.util.spec_from_file_location('coenrich_official_FeatureComputer',
                                                 path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FeatureComputer


def _node_arrays(sp, order=None):
    """从 sp 抽取 depot + pool 的可见数组（节点顺序 = [0, *pool]）。"""
    pool = [int(c) for c in sp.pool_customer_ids]
    idx = {int(n): i for i, n in enumerate(sp.node_ids)}
    nodes = [0] + pool
    if 0 not in idx:
        raise ValueError('子问题缺少 depot(0) 节点')
    coords = np.array([list(map(float, sp.coords[idx[n]])) for n in nodes],
                      dtype=np.float64)
    tw = np.array([[float(sp.tw_start[idx[n]]), float(sp.tw_end[idx[n]])]
                   for n in nodes], dtype=np.float64)
    svc = np.array([float(sp.service_time[idx[n]]) for n in nodes], dtype=np.float64)
    dem = np.array([float(sp.demands[idx[n]]) for n in nodes], dtype=np.float64)
    dm = np.array([[float(sp.travel_mat[idx[a]][idx[b]]) for b in nodes]
                   for a in nodes], dtype=np.float64)
    return pool, nodes, coords, tw, svc, dem, dm


def build_mlco_inputs(sp, clock=None, end_epoch=None):
    """sp → (args, observation, instance, static_info)。"""
    pool, nodes, coords, tw, svc, dem, dm = _node_arrays(
        sp, [int(c) for c in sp.pool_customer_ids])
    start = float(sp.ready_time if clock is None else clock)
    horizon = float(sp.depot_tw_end if end_epoch is None else end_epoch)
    if horizon <= start:                      # 视界已过：给最小正剩余时间，避免 0 除
        horizon = start + 1e-6
    is_depot = np.array([True] + [False] * len(pool))
    # 官方 read_vrplib 口径：demands 不含 depot（长度 m），其余含 depot（长度 m+1）
    instance = {
        'is_depot': is_depot,
        'coords': coords,
        'demands': dem[1:].copy(),
        'capacity': float(sp.capacity),
        'time_windows': tw,
        'service_times': svc,
        'duration_matrix': dm,
    }
    epoch_instance = {
        'is_depot': is_depot,
        'customer_idx': np.arange(len(nodes), dtype=np.int64),
        'request_idx': np.array(nodes, dtype=np.int64),
        'coords': coords,
        'demands': dem.copy(),
        'time_windows': tw,
        'service_times': svc,
        'must_dispatch': np.zeros(len(nodes), dtype=bool),
        'duration_matrix': dm,
        'release_times': np.zeros(len(nodes), dtype=np.float64),
        'capacity': float(sp.capacity),
    }
    observation = {
        'current_epoch': 0,
        'current_time': start,
        'planning_starttime': start,
        'epoch_instance': epoch_instance,
    }
    args = SimpleNamespace(predictor='NeuralNetwork', feature_set='dynamicstatic')
    static_info = {'end_epoch': horizon}
    return args, observation, instance, static_info


def compute_features(sp, clock=None, feature_set='dynamicstatic', Predictor=None):
    """返回 (X, feature_names)：X 形状 (1+len(pool), n_feat)，第 0 行 = depot。"""
    args, observation, instance, static_info = build_mlco_inputs(sp, clock=clock)
    args.feature_set = feature_set
    FC = Predictor if Predictor is not None else load_feature_computer()
    fc = FC(args, observation, instance, static_info)
    # 时间单位适配（见模块 docstring 第 2 条）
    fc.EPOCH_DURATION = A1_EPOCH_DURATION
    fc.MARGIN_DISPATCH = A1_MARGIN_DISPATCH
    if 'dynamic' in feature_set:
        for name in _DYNAMIC_ORDER:
            getattr(fc, 'get_feature_' + name)()
    if 'static' in feature_set:
        for name in _STATIC_ORDER:
            getattr(fc, 'get_feature_' + name)()
    df = fc.get_features()
    X = np.asarray(df.values, dtype=np.float64)
    return X, list(df.columns)
