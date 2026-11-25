# -*- coding: utf-8 -*-
"""远端 P5 工作器（单端点）：冻结 f0_r0 热图 + 标准掩码重构搜索。

方法语义（18 法名单内本文族 + 标准经典基线）：
  ORDER_SPLIT   神经热图 + 六组合试选 + 每3轮跨车 Split
  ORDER_SPLIT_D 纯距离 + 六组合试选 + 每3轮跨车 Split
  MASK          神经热图 + 掩码重构（无 Split）
  SLOT          神经热图 + 最短可行绕行插位保护（无 Split）
  DIST          纯距离 + 掩码重构（无 Split）
  DIST_SLOT     纯距离 + 插位保护（无 Split；全位置评价下保护自动满足）
  ALNS          标准 ALNS（Ropke–Pisinger，标准任务独立实现，共同初解）
  VNS           标准 VNS（relocate/swap/2-opt + 震荡，共同初解）
校准热图在标准任务上退化为原热图（无标准任务残差校准，见 P5_预登记）。
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

P5 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(P5))
PROJECT = Path("/home/hzeng/project/MASKCO-Main")

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw)
from standard_vrp.core.standard_features import StandardFeatureBuilder, WindowedFeatureBuilder
from standard_vrp.solver.standard_pool_search import standard_search
from standard_vrp.solver.standard_classical import alns_search, vns_search

INTERNAL = ("ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT")
CLASSICAL = ("ALNS", "VNS")


def load_instance(name, objective):
    if objective == "cvrp":
        return CVRPInstance.parse_vrp((P5 / "cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))
    return VRPTWInstance.parse_solomon(P5 / "solomon/instances" / (name + ".vrp"))


def check(row, method, block, affinity, initial=None):
    if row["case"] != block["case"] or row["seed"] != block["seed"] or row["method"] != method:
        raise ValueError("P5 cell identity mismatch")
    if row["status"] != "AUDITED":
        raise ValueError("P5 infeasible/unaudited endpoint")
    if len(affinity) != 1:
        raise ValueError("single core required")
    if not 0 <= row["online_s"] <= 60:
        raise ValueError("online budget exceeded: %.2f" % row["online_s"])
    return True


def main(idx, method):
    cfg = json.loads((HERE / "phase.json").read_text(encoding="utf-8"))
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or method not in INTERNAL + CLASSICAL or cfg["seconds"] != 60:
        raise ValueError("Unregistered P5 method or wall time")
    name, seed, objective = block["case"], block["seed"], block["objective"]
    inst = load_instance(name, objective)
    # 冻结模型（f0_r0；c205 仅作作用域案例）——仅神经方法加载
    neural = method in ("ORDER_SPLIT", "MASK", "SLOT")
    if neural:
        origin = PROJECT / "_maskco_cross_dataset_20261119"
        sys.path.insert(0, str(origin))
        import compare
        access, core, source, locked = compare.setup("SLOT")
        policy, scope = compare.load("SLOT", access, core, source, locked, "c205")
        from profit_neural import predict
    fb = StandardFeatureBuilder(inst, objective)
    wfb = WindowedFeatureBuilder(inst, objective)

    def heatmap_fn(inst2, partial, removed, neutral=None):
        if inst2.dimension <= 100:
            batch = fb.build(partial, removed)
            with access.restricted_scope(scope):
                logits = predict(policy, batch)
            shifted = logits - logits.max(axis=1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        window, batch, pos_of = wfb.build_windowed(partial, removed)
        with access.restricted_scope(scope):
            logits = predict(policy, batch)
        shifted = logits - logits.max(axis=1, keepdims=True)
        h = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        full = np.full((inst2.dimension + 1, inst2.dimension + 1), -np.inf, np.float32)
        for i in removed:
            p = pos_of[i]
            for j, q in pos_of.items():
                full[i][j] = h[p][q]
        return full

    t0 = time.perf_counter()
    from standard_vrp.solver.standard_pool_search import nn_witness
    initial = nn_witness(inst)
    if method in CLASSICAL:
        out = (alns_search if method == "ALNS" else vns_search)(
            inst, seconds=60, seed=seed, objective=objective)
        out["epochs"] = out.get("iterations", 0)
    else:
        out = standard_search(inst, seconds=60, seed=seed, objective=objective,
                              heatmap_fn=heatmap_fn if neural else None,
                              split_enabled=method in ("ORDER_SPLIT", "ORDER_SPLIT_D"),
                              slot=method in ("SLOT", "DIST_SLOT"))
    online_s = time.perf_counter() - t0
    if objective == "cvrp":
        ok, d, loads, k = evaluate_cvrp(inst, out["routes"], rounded=True)
        ok0, d0, _, _ = evaluate_cvrp(inst, initial, rounded=True)
        row = dict(case=name, seed=seed, method=method, objective="cvrp",
                   status="AUDITED" if ok else "INFEASIBLE", vehicles=k, distance=d,
                   routes=out["routes"], initial_routes=initial, initial_U=d0,
                   online_s=online_s, epochs=out["epochs"])
    else:
        ok, k, d, w, loads = evaluate_vrptw(inst, out["routes"])
        ok0, k0, d0, _, _ = evaluate_vrptw(inst, initial)
        row = dict(case=name, seed=seed, method=method, objective="vrptw",
                   status="AUDITED" if ok else "INFEASIBLE", vehicles=k, distance=d, wait=w,
                   routes=out["routes"], initial_routes=initial,
                   initial_U=dict(vehicles=k0, distance=d0),
                   online_s=online_s, epochs=out["epochs"])
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    (HERE / "out").mkdir(exist_ok=True)
    with (HERE / "out" / f"{idx}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
