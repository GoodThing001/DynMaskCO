# -*- coding: utf-8 -*-
"""校验 24 个开发实例：解析 .vrp、构造解可行性、规模与 BKS 无关性登记。"""
import json
import sys
from pathlib import Path

V2 = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2")
sys.path.insert(0, str(V2))
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                 evaluate_cvrp, evaluate_vrptw)

DEV = V2 / "development"
manifest = json.loads((DEV / "manifest.json").read_text(encoding="utf-8"))
ok_all = True
for row in manifest["instances"]:
    p = DEV / "instances" / row["file"]
    if row["kind"] == "cvrp":
        inst = CVRPInstance.parse_vrp(p.read_text(encoding="utf-8"))
        # 用最近邻见证检验可行（构造解不存在 CVRP）
        ok, d, k = None, None, None
        from standard_vrp.solver.pool_search import nn_witness
        w = nn_witness(inst)
        ok, d, loads, k = evaluate_cvrp(inst, w, rounded=True)
        status = "OK" if ok else "FAIL"
        print("%-26s cvrp n=%4d cap=%5d witness_feasible=%s D=%.1f K=%d"
              % (p.stem, inst.dimension, int(inst.capacity), status, d, k))
    else:
        inst = VRPTWInstance.parse_solomon(p)
        routes = [tuple(r) for r in row["constructed_routes"]]
        ok, k, d, w, loads = evaluate_vrptw(inst, routes)
        status = "OK" if ok else "FAIL"
        ok2 = False
        if not ok:
            print("  constructed routes infeasible; retry with nn witness")
        print("%-26s vrptw n=%3d constructed_feasible=%s (K=%d,D=%.2f) max_vh=%s"
              % (p.stem, inst.dimension, status, k, d, inst.max_vehicles))
        ok_all = ok_all and ok
    if row["kind"] == "cvrp":
        ok_all = ok_all and ok
print("ALL DEV INSTANCES VALID:" , ok_all)
