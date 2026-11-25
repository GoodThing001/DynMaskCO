# -*- coding: utf-8 -*-
"""P5-B 官方 BKS 复核（9.2）：本地评价器复算 CVRPLIB Solomon BKS 路线。

同时比对本地 SINTEF 原件与 CVRPLIB 副本的字节一致性。
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "P5_standard_vrp" / "data" / "solomon"
SINTEF = HERE.parent.parent.parent / "C-VRP_Cold-chainVehicleRoutingProblem" / "data" / \
    "classic_static_sources_20261020" / "solomon_100" / "instances" / "In"
sys.path.insert(0, str(HERE.parent / "P5_standard_vrp"))
from standard_vrp.core.standard_evaluator import VRPTWInstance, evaluate_vrptw  # noqa: E402


def parse_bks(text):
    routes = []
    for line in text.splitlines():
        if line.startswith("Route #"):
            routes.append(tuple(int(x) for x in line.split(":")[1].split()))
    m = re.search(r"Cost\s+([0-9.]+)", text)
    return routes, float(m.group(1)) if m else None


def main():
    inst_dir = DATA / "instances"
    bks_dir = DATA / "bks"
    sols = sorted(bks_dir.glob("*.sol"))
    print("bks files:", len(sols))
    if not sols:
        print("Solomon BKS not downloaded yet")
        return
    rows = []
    for sol in sols:
        name = sol.stem
        inst_path = inst_dir / (name + ".vrp")
        if not inst_path.is_file():
            rows.append(dict(instance=name, status="NO_INSTANCE"))
            continue
        inst = VRPTWInstance.parse_solomon(inst_path)
        routes, cost = parse_bks(sol.read_text(encoding="utf-8"))
        ok, k, d, w, loads = evaluate_vrptw(inst, routes)
        rows.append(dict(instance=name, n=inst.dimension, capacity=inst.capacity,
                         header_vehicles=inst.max_vehicles, k=len(routes), official_cost=cost,
                         evaluated_d=round(d, 4) if ok else None, feasible=ok,
                         match=None if cost is None else abs(d - cost) < 0.05))
    n_ok = sum(1 for r in rows if r["feasible"])
    n_match = sum(1 for r in rows if r.get("match") is True)
    print("feasible:", n_ok, "/", len(rows), "cost match:", n_match)
    for r in rows:
        if not r["feasible"] or r.get("match") is not True:
            print("CHECK:", r)
    # 本地 SINTEF 与 CVRPLIB 副本字节比对
    diffs = []
    for p in sorted(SINTEF.glob("*.txt")):
        alt = inst_dir / (p.stem.upper() + ".vrp")
        if alt.is_file():
            a = hashlib.sha256(p.read_bytes()).hexdigest()
            b = hashlib.sha256(alt.read_bytes()).hexdigest()
            if a != b:
                diffs.append((p.name, "content differs"))
        else:
            diffs.append((p.name, "no cvrplib copy"))
    print("sintef vs cvrplib byte diff:", len(diffs), diffs[:8])
    with (DATA / "bks_verification.json").open("w", encoding="utf-8") as f:
        json.dump(dict(feasible=n_ok, total=len(rows), cost_match=n_match, rows=rows, byte_diffs=diffs),
                  f, indent=2, ensure_ascii=False)
    print("bks_verification.json written")


if __name__ == "__main__":
    main()
