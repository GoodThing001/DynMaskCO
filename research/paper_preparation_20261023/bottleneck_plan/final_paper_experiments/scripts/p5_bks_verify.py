# -*- coding: utf-8 -*-
"""P5-A 官方 BKS 复核（9.2）：本地评价器复算 CVRPLIB X 集 100 份 BKS 路线。

对照两种距离口径：双精度 vs 逐边四舍五入，锁定与官方 Cost 一致的口径。
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "P5_standard_vrp" / "data" / "cvrplib"
sys.path.insert(0, str(HERE.parent / "P5_standard_vrp"))
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp  # noqa: E402


def parse_bks_routes(text):
    routes = []
    for line in text.splitlines():
        if line.startswith("Route #"):
            ids = [int(x) for x in line.split(":")[1].split()]
            routes.append(tuple(ids))
    m = re.search(r"Cost\s+(\d+)", text)
    cost = int(m.group(1)) if m else None
    return routes, cost


def main():
    manifest = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
    dl = manifest["individual_download"]
    matches = dict(double=0, rounded=0, neither=0, both=0)
    rows = []
    for name in sorted(dl["instances"]):
        vrp = DATA / "instances" / (name + ".vrp")
        sol = DATA / "bks" / (name + ".sol")
        if not sol.is_file():
            rows.append(dict(instance=name, status="NO_BKS"))
            continue
        inst = CVRPInstance.parse_vrp(vrp.read_text(encoding="utf-8"))
        routes, official = parse_bks_routes(sol.read_text(encoding="utf-8"))
        ok1, d1, _, k1 = evaluate_cvrp(inst, routes, rounded=False)
        ok2, d2, _, k2 = evaluate_cvrp(inst, routes, rounded=True)
        d1r, d2r = round(d1), round(d2)
        hit1 = official is not None and abs(d1 - official) < 0.5001  # 官方为整数Cost
        hit2 = official is not None and abs(d2 - official) < 0.5001
        key = "both" if (hit1 and hit2) else ("double" if hit1 else ("rounded" if hit2 else "neither"))
        matches[key] += 1
        rows.append(dict(instance=name, n=inst.dimension, capacity=inst.capacity,
                         k=len(routes), official_cost=official,
                         double=d1 if ok1 else None, double_rounded=d2 if ok2 else None,
                         feasible=ok1 and ok2, match=key))
    print("match summary:", matches)
    bad = [r for r in rows if r["match"] == "neither" or not r["feasible"]]
    print("unmatched/infeasible:", len(bad))
    for r in bad[:10]:
        print(r)
    with (DATA / "bks_verification.json").open("w", encoding="utf-8") as f:
        json.dump(dict(matches=matches, rows=rows), f, indent=2, ensure_ascii=False)
    print("bks_verification.json written")


if __name__ == "__main__":
    main()
