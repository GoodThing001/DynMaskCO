# -*- coding: utf-8 -*-
"""远端 P5 外部方法工作器（单端点）：PyVRP 原生求解 + 作者权重模型批次。

方法语义（P5_预登记 §9 支持矩阵）：
  PyVRP      pyvrp.Model 原生求解（CVRP 距离矩阵取整口径 / VRPTW 时间矩阵缩放口径），
             MaxRuntime(58) 硬截止 + 种子；在线墙钟 ≤60s；
  AttentionModel / POMO / DeepACO / CaR / MVMoE / CaDA / RouteFinder / RRNCO
             作者权重构造式推理（模块 external01/methods/<name>.py，作者仓库只读），
             不接受共同初解（构造式方法，initial 记为 None），原生构造结果由独立评价器复核。
所有终点均用 standard_vrp 独立评价器重算（1e-6）。
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

P5 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(P5))

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw)

AUTHOR_MODULES = {
    "AttentionModel": "am.py",
    "POMO": "pomo.py",
    "DeepACO": "deepaco.py",
    "CaR": "car.py",
    "MVMoE": "mvmoe.py",
    "CaDA": "cada.py",
    "RouteFinder": "routefinder.py",
    "RRNCO": "rrnco.py",
}
CONSTRUCTORS = frozenset(AUTHOR_MODULES)

# 预登记 §6 已登记的 8 个"1 位截断"实例：官方 BKS 在双精度下边界超窗 ≤0.1，
# 整数缩放求解器（PyVRP）可能给出仅截断口径可行的解 → 允许 TRUNCATED_OK 登记。
TRUNCATED_CASES = frozenset({"R102", "R105", "R107", "R108", "R112", "R211",
                             "RC101", "RC105"})
TRUNCATION_TOL = 0.11


def _vrptw_max_violation(inst, routes):
    """最大时窗/仓库窗超限量（双精度分钟）。"""
    worst = 0.0
    for r in routes:
        t = inst.depot_window[0]
        for idx, i in enumerate(r):
            t += inst.dist[0 if idx == 0 else r[idx - 1]][i]
            if t < inst.windows[i][0]:
                t = inst.windows[i][0]
            worst = max(worst, t - inst.windows[i][1])
            t += inst.service[i]
        t += inst.dist[r[-1]][0]
        worst = max(worst, t - inst.depot_window[1])
    return worst


def load_instance(name, objective):
    if objective == "cvrp":
        return CVRPInstance.parse_vrp((P5 / "cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))
    return VRPTWInstance.parse_solomon(P5 / "solomon/instances" / (name + ".vrp"))


def check(row, method, block, affinity, initial=None):
    if row["case"] != block["case"] or row["seed"] != block["seed"] or row["method"] != method:
        raise ValueError("P5 cell identity mismatch")
    allowed = ("AUDITED", "TRUNCATED_OK")
    if method in CONSTRUCTORS:
        allowed = allowed + ("INFEASIBLE",)  # 构造式方法原生成绩如实登记（预登记 §6）
    if row["status"] not in allowed:
        raise ValueError("P5 infeasible/unaudited endpoint")
    if row["status"] == "TRUNCATED_OK" and not (
            row["case"] in TRUNCATED_CASES and row.get("max_violation", 1e9) <= TRUNCATION_TOL):
        raise ValueError("P5 truncated-status not allowed for this cell")
    if len(affinity) != 1:
        raise ValueError("single core required")
    # 搜索族：60s 在线预算；构造式方法（作者默认推理）按预登记 §3 记录原生成绩，
    # 预算档放宽到 1200s 并以 online_s 如实记录（RRNCO 作者默认 num_starts=100 单核 ~490s）
    cap = 1200 if method in CONSTRUCTORS else 60
    if not 0 <= row["online_s"] <= cap:
        raise ValueError("online budget exceeded: %.2f" % row["online_s"])
    return True


def solve_pyvrp(inst, objective, seed):
    """PyVRP 0.11.3 原生求解（显式 Depot/Client/VehicleType/ProblemData 构造，与 P4 同版）。

    CVRP：EUC_2D 逐边取整距离矩阵（numpy 向量化构建，避免 n=1000 时 Python 循环计入在线预算）；
    VRPTW：距离/时间矩阵 ×100 缩放取整（搜索导向），终点距离由 standard_vrp 独立评价器
    （双精度）重算为准。MaxRuntime(56) + 提取/评价余量 → online_s ≤ 60s。
    """
    import numpy as np
    import pyvrp
    from pyvrp.stop import MaxRuntime
    n = inst.dimension
    coords = np.array([[inst.coords[i][0], inst.coords[i][1]] for i in range(0, n + 1)], dtype=float)
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_f = np.hypot(dx, dy)
    np.fill_diagonal(dist_f, 0.0)
    # 车队上限（注册勘误 §2）：CVRP 自由车队（num_available=n，K 报告）；VRPTW 官方 25
    if objective == "cvrp":
        num_vehicles = n
    else:
        num_vehicles = inst.max_vehicles or n
    if objective == "cvrp":
        mat = np.floor(dist_f + 0.5).astype(np.int64)  # CVRPLIB EUC_2D 逐边取整
        matrix = mat.tolist()
        depot = pyvrp.Depot(x=int(round(inst.coords[0][0] * 1000)),
                            y=int(round(inst.coords[0][1] * 1000)))
        clients = [pyvrp.Client(x=int(round(inst.coords[i][0] * 1000)),
                                y=int(round(inst.coords[i][1] * 1000)),
                                delivery=[int(round(inst.demands[i]))])
                   for i in range(1, n + 1)]
        vehicle = pyvrp.VehicleType(num_available=num_vehicles,
                                    capacity=[int(round(inst.capacity))])
        data = pyvrp.ProblemData(clients, [depot], [vehicle], [matrix], [matrix])
    else:
        scale = 100
        mat = np.floor(dist_f * scale + 0.5).astype(np.int64)
        matrix = mat.tolist()
        depot = pyvrp.Depot(x=int(round(inst.coords[0][0] * 1000)),
                            y=int(round(inst.coords[0][1] * 1000)),
                            tw_early=int(round(inst.depot_window[0] * scale)),
                            tw_late=int(round(inst.depot_window[1] * scale)))
        clients = [pyvrp.Client(x=int(round(inst.coords[i][0] * 1000)),
                                y=int(round(inst.coords[i][1] * 1000)),
                                delivery=[int(round(inst.demands[i]))],
                                tw_early=int(round(inst.windows[i][0] * scale)),
                                tw_late=int(round(inst.windows[i][1] * scale)),
                                service_duration=int(round(inst.service[i] * scale)))
                   for i in range(1, n + 1)]
        vehicle = pyvrp.VehicleType(num_available=num_vehicles,
                                    capacity=[int(round(inst.capacity))],
                                    tw_early=int(round(inst.depot_window[0] * scale)),
                                    tw_late=int(round(inst.depot_window[1] * scale)))
        data = pyvrp.ProblemData(clients, [depot], [vehicle], [matrix], [matrix])
    model = pyvrp.Model.from_data(data)
    # 按规模分级收缩搜索时长：HGS 单迭代在 n 大时可达数秒，须给迭代超调留余量（online_s ≤ 60）
    if n >= 800:
        budget = 44
    elif n >= 500:
        budget = 50
    elif n >= 300:
        budget = 53
    else:
        budget = 56
    res = model.solve(stop=MaxRuntime(budget), seed=seed, display=False)
    # visits() 返回位置序号（0=仓库，1..n=客户），与本模块客户 id 口径一致
    return tuple(tuple(int(v) for v in route.visits()) for route in res.best.routes())


def main(idx, method):
    cfg = json.loads((HERE / "phase.json").read_text(encoding="utf-8"))
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or cfg["seconds"] != 60:
        raise ValueError("Unregistered P5 external method or wall time")
    name, seed, objective = block["case"], block["seed"], block["objective"]
    inst = load_instance(name, objective)
    if method == "PyVRP":
        from standard_vrp.solver.standard_pool_search import nn_witness
        initial = nn_witness(inst)
        t0 = time.perf_counter()
        routes = solve_pyvrp(inst, objective, seed)
        online_s = time.perf_counter() - t0
        epochs = None
    elif method in AUTHOR_MODULES:
        # 构造式方法：不接受共同初解；作者默认推理
        initial, initial_U = None, None
        spec = importlib.util.spec_from_file_location(
            method.lower(), str(P5 / "external01/methods" / AUTHOR_MODULES[method]))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(P5 / "ext_repos"))  # AM 等仓库通用包名 utils/nets/problems
        sys.path.insert(0, str(P5 / "external01/methods"))  # 兄弟模块（_mtvrp_common/_vendor）
        t0 = time.perf_counter()
        spec.loader.exec_module(mod)
        routes = mod.solve(inst, objective, seed)
        online_s = time.perf_counter() - t0
        epochs = None
    else:
        raise NotImplementedError("external method not yet integrated: " + method)
    if objective == "cvrp":
        ok, d, loads, k = evaluate_cvrp(inst, routes, rounded=True)
        if initial is not None:
            ok0, d0, _, _ = evaluate_cvrp(inst, initial, rounded=True)
        else:
            d0 = None
        row = dict(case=name, seed=seed, method=method, objective="cvrp",
                   status="AUDITED" if ok else "INFEASIBLE", vehicles=k, distance=d,
                   routes=routes, initial_routes=initial, initial_U=d0,
                   online_s=online_s, epochs=epochs)
    else:
        ok, k, d, w, loads = evaluate_vrptw(inst, routes)
        status = "AUDITED" if ok else "INFEASIBLE"
        max_viol = 0.0
        if not ok:
            max_viol = _vrptw_max_violation(inst, [tuple(r) for r in routes])
            if name in TRUNCATED_CASES and max_viol <= TRUNCATION_TOL:
                status = "TRUNCATED_OK"  # 截断口径可行（预登记 §6 已登记实例）
        if initial is not None:
            ok0, k0, d0, _, _ = evaluate_vrptw(inst, initial)
            initial_U = dict(vehicles=k0, distance=d0)
        else:
            initial_U = None
        row = dict(case=name, seed=seed, method=method, objective="vrptw",
                   status=status, vehicles=k, distance=d, wait=w,
                   routes=routes, initial_routes=initial, initial_U=initial_U,
                   online_s=online_s, epochs=epochs, max_violation=max_viol)
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    (HERE / "out").mkdir(exist_ok=True)
    with (HERE / "out" / f"{idx}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
