# -*- coding: utf-8 -*-
"""P5-V2 远端执行器（单端点）：五种受控版本 + 六种正式内部组合，统一 60 秒单核。

版本语义（EXPERIMENT_PROTOCOL_V2 §1，开发面板用）：
  STD_OLD    —— 冻结 P5-V1 代码路径（复现 V1：v1 特征 + V1 搜索）
  STD_BUGFIX —— V2 模块、仅缺陷修复、V1 简化搜索（variant="bugfix"，features mode="v1"）
  STD_FIX    —— V2 全机制（variant="full"），冻结编码（features mode="v2_fix"）
  STD_NORM   —— V2 全机制，训练语义对齐编码（features mode="v2_norm"）
  STD_DIST   —— V2 全机制，无神经引导（heatmap_fn=None）

正式复验面板的六种内部组合（与 P5-V1 名称一一对应，全部走 V2 修复内核）：
  ORDER_SPLIT   神经 + Split            ORDER_SPLIT_D 距离 + Split
  MASK          神经（无 Split）         SLOT          神经 + 插位保护（无 Split）
  DIST          距离（无 Split）         DIST_SLOT     距离 + 插位保护

冻结网络：f0_r0（cross_dataset compare chain，JAX CPU）；每 epoch 一次前向。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

P5 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
P5V2 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2")
HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/hzeng/project/MASKCO-Main")
V1_ROOT = P5  # 冻结的 P5-V1 包根（standard_vrp）
sys.path.insert(0, str(V1_ROOT))   # 供 V1 模块内部 import
sys.path.insert(0, str(P5V2))      # V2 包优先解析 standard_vrp

VERSIONS = ("STD_OLD", "STD_BUGFIX", "STD_FIX", "STD_NORM", "STD_DIST")

SPEC = {
    "STD_BUGFIX": dict(mode="v1", heat=True, split=False, slot=False, variant="bugfix"),
    "STD_FIX": dict(mode="v2_fix", heat=True, split=True, slot=False, variant="full"),
    "STD_NORM": dict(mode="v2_norm", heat=True, split=True, slot=False, variant="full"),
    "STD_DIST": dict(mode="v2_fix", heat=False, split=True, slot=False, variant="full"),
    "ORDER_SPLIT": dict(mode="v2_fix", heat=True, split=True, slot=False, variant="full"),
    "ORDER_SPLIT_D": dict(mode="v2_fix", heat=False, split=True, slot=False, variant="full"),
    "MASK": dict(mode="v2_fix", heat=True, split=False, slot=False, variant="full"),
    "SLOT": dict(mode="v2_fix", heat=True, split=False, slot=True, variant="full"),
    "DIST": dict(mode="v2_fix", heat=False, split=False, slot=False, variant="full"),
    "DIST_SLOT": dict(mode="v2_fix", heat=False, split=False, slot=True, variant="full"),
}
INTERNAL = ("ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT")
ALL_METHODS = ("STD_OLD",) + VERSIONS[1:] + INTERNAL


def load_instance(path, objective):
    from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance
    if objective == "cvrp":
        return CVRPInstance.parse_vrp(Path(path).read_text(encoding="utf-8"))
    return RV(path)


def RV(path):
    from standard_vrp.core.standard_evaluator import VRPTWInstance
    return VRPTWInstance.parse_solomon(Path(path))


def _jsonable(obj):
    """递归把 numpy 标量/数组转为 Python 原生类型（JSON 兼容）。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def check(row, version, block, affinity, initial=None):
    if row["case"] != block["case"] or row["seed"] != block["seed"] or row["version"] != version:
        raise ValueError("P5-V2 cell identity mismatch")
    allowed = ("AUDITED", "TRUNCATED_OK")
    if version == "STD_OLD":
        # 冻结 V1 代码在部分实例上崩溃或产出不可行解：如实登记，不掩盖、不修复 V1
        allowed += ("V1_CRASH", "V1_INFEASIBLE")
    if row["status"] not in allowed:
        raise ValueError("P5-V2 infeasible endpoint: " + str(row["status"]))
    if len(affinity) != 1:
        raise ValueError("single core required")
    if row["status"] not in ("V1_CRASH",) and not 0 <= row["online_s"] <= 60:
        raise ValueError("online budget exceeded: %.2f" % row["online_s"])
    return True


def _violation(inst, routes):
    """VRPTW 不可行度（分钟）：最大时窗/仓库窗超限；CVRP 返回容量超限。"""
    try:
        from standard_vrp.core.standard_evaluator import CVRPInstance
        if isinstance(inst, CVRPInstance):
            worst = 0.0
            for r in routes:
                load = sum(inst.demands[i] for i in r)
                worst = max(worst, load - inst.capacity)
            return dict(kind="capacity", worst=round(worst, 6))
        worst = 0.0
        for r in routes:
            t = inst.depot_window[0]
            prev = 0
            for i in r:
                t += inst.dist[prev][i]
                if t < inst.windows[i][0]:
                    t = inst.windows[i][0]
                worst = max(worst, t - inst.windows[i][1])
                t += inst.service[i]
                prev = i
            t += inst.dist[prev][0]
            worst = max(worst, t - inst.depot_window[1])
        return dict(kind="window", worst=round(worst, 6))
    except Exception:  # noqa: BLE001
        return dict(kind="unknown", worst=None)


def make_predictor():
    """加载冻结 f0_r0（一次）。返回 (predict, scope_cm, access)。"""
    origin = PROJECT / "_maskco_cross_dataset_20261119"
    sys.path.insert(0, str(origin))
    import compare
    access, core, source, locked = compare.setup("SLOT")
    policy, scope = compare.load("SLOT", access, core, source, locked, "c205")
    from profit_neural import predict

    def forward(batch):
        with access.restricted_scope(scope):
            logits = predict(policy, batch)
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    return forward, access, scope


def make_heatmap_v2(forward, fb, wfb, inst):
    """V2 特征构造 + 窗口映射；返回每 epoch 热图函数。"""
    n = inst.dimension

    def heatmap(inst2, routes, removed, _neutral=None):
        removed = set(removed)
        if n <= 100:
            batch = fb.build(routes, removed)
            return forward(batch)
        window, batch, pos_of = wfb.build_windowed(routes, removed)
        h = forward(batch)  # (1, w, w) log-softmax rows
        full = np.full((n + 1, n + 1), -np.inf, np.float32)
        for i in removed:
            p = pos_of[i]
            for j, q in pos_of.items():
                full[i][j] = h[p][q]
        return full

    return heatmap


def make_heatmap_v1(forward, inst, objective, mod):
    """STD_OLD：冻结 P5-V1 特征构造器（v1 编码，显式模块）。"""
    fb = mod.StandardFeatureBuilder(inst, objective)
    wfb = mod.WindowedFeatureBuilder(inst, objective)
    n = inst.dimension

    def heatmap(inst2, routes, removed, _neutral=None):
        removed = set(removed)
        if n <= 100:
            return forward(fb.build(routes, removed))
        window, batch, pos_of = wfb.build_windowed(routes, removed)
        h = forward(batch)
        full = np.full((n + 1, n + 1), -np.inf, np.float32)
        for i in removed:
            p = pos_of[i]
            for j, q in pos_of.items():
                full[i][j] = h[p][q]
        return full

    return heatmap


def main(idx, version):
    cfg = json.loads((HERE / "phase.json").read_text(encoding="utf-8"))
    block = cfg["blocks"][int(idx)]
    if version not in ALL_METHODS or version not in block["methods"]:
        raise ValueError("unregistered method")
    if cfg["seconds"] != 60:
        raise ValueError("wall time must be 60")
    case, seed, objective = block["case"], block["seed"], block["objective"]
    dest = HERE / "out" / f"{idx}_{version}.json"
    stash = HERE.parent / "dev01_stash" / f"{idx}_{version}.json"
    if not dest.exists() and stash.exists():
        # 断点续跑：该端点在上一轮（同一代码路径）已产出，直接搬入本轮 out/
        try:
            done = json.loads(stash.read_text(encoding="utf-8"))
        except Exception:
            done = None
        if done and done.get("case") == case and done.get("seed") == seed:
            (HERE / "out").mkdir(exist_ok=True)
            dest.write_bytes(stash.read_bytes())
            return
    if dest.exists():
        try:
            done = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:
            dest.unlink()
        else:
            if done.get("case") == case and done.get("seed") == seed:
                return
    inst = load_instance(block["path"], objective)

    if version == "STD_OLD":
        forward, access, scope = make_predictor()
        from standard_vrp.adapters.objective import adapter_for
        import importlib.util as iu
        v1feat = iu.spec_from_file_location(
            "v1_standard_features", str(V1_ROOT / "standard_vrp/core/standard_features.py"))
        v1fb_mod = iu.module_from_spec(v1feat)
        v1feat.loader.exec_module(v1fb_mod)
        v1search = iu.spec_from_file_location(
            "v1_standard_pool_search", str(V1_ROOT / "standard_vrp/solver/standard_pool_search.py"))
        v1 = iu.module_from_spec(v1search)
        v1search.loader.exec_module(v1)
        heat = make_heatmap_v1(forward, inst, objective, v1fb_mod)
        adapter = adapter_for(inst)
        from standard_vrp.solver.pool_search import nn_witness
        init = [tuple(r) for r in nn_witness(inst)]
        t0 = time.perf_counter()
        try:
            out = v1.standard_search(inst, seconds=60, seed=seed, objective=objective,
                                     heatmap_fn=heat, split_enabled=True, slot=False)
        except Exception as exc:  # noqa: BLE001 —— 冻结 V1 代码的缺陷如实登记
            import traceback
            online_s = time.perf_counter() - t0
            row = dict(case=case, seed=seed, version=version, objective=objective,
                       status="V1_CRASH", distance=None, vehicles=None, routes=[],
                       online_s=online_s, stats=dict(epochs=None),
                       error="%s: %s" % (type(exc).__name__, exc),
                       traceback=traceback.format_exc()[-2000:])
            ok0, val0, k0 = adapter.evaluate(init)
            row.update(initial_routes=init, initial_U=dict(distance=val0, vehicles=k0))
            check(row, version, block, sorted(os.sched_getaffinity(0)))
            (HERE / "out").mkdir(exist_ok=True)
            with (HERE / "out" / f"{idx}_{version}.json").open("x") as f:
                json.dump(_jsonable(row), f, allow_nan=False)
            return
        online_s = time.perf_counter() - t0
        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "V1_INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s, stats=dict(epochs=out.get("epochs")))
        if not ok:
            row["violation"] = _violation(inst, routes)
    else:
        spec = SPEC[version]
        forward, access, scope = (make_predictor() if spec["heat"] else (None, None, None))
        from standard_vrp.adapters.objective import adapter_for
        from standard_vrp.solver.pool_search import standard_search, nn_witness
        from standard_vrp.core.standard_features import (StandardFeatureBuilder,
                                                         WindowedFeatureBuilder)
        heat = None
        if spec["heat"]:
            fb = StandardFeatureBuilder(inst, objective, mode=spec["mode"])
            wfb = WindowedFeatureBuilder(inst, objective, mode=spec["mode"])
            heat = make_heatmap_v2(forward, fb, wfb, inst)
        adapter = adapter_for(inst)
        t0 = time.perf_counter()
        out = standard_search(adapter, seconds=60, seed=seed, heatmap_fn=heat,
                              split_enabled=spec["split"], slot=spec["slot"],
                              variant=spec["variant"])
        online_s = time.perf_counter() - t0
        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s,
                   stats={kk: out[kk] for kk in ("epochs", "iterations", "improvements",
                                                 "split_blocks", "split_orders", "split_offers",
                                                 "ranking_counts", "regret_epochs",
                                                 "mask_mode_counts", "fallbacks", "predictor",
                                                 "history")})
        init = [tuple(r) for r in nn_witness(inst)]
    ok0, val0, k0 = adapter_for(inst).evaluate(init)
    row.update(initial_routes=init, initial_U=dict(distance=val0, vehicles=k0))
    check(row, version, block, sorted(os.sched_getaffinity(0)))
    (HERE / "out").mkdir(exist_ok=True)
    with (HERE / "out" / f"{idx}_{version}.json").open("x") as f:
        json.dump(_jsonable(row), f, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
