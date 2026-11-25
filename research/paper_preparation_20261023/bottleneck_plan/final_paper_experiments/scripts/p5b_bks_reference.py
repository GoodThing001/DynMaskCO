# -*- coding: utf-8 -*-
"""P5-B BKS 参考表构建：官方解路线在双精度评价下的 (K, D)，5 例边界按截断口径处理。

规则（P5_预登记 §6）：
  - 双精度可行 → (K, D) 直接采用；
  - 双精度不可行但在官方 1 位截断行驶时间口径可行 → 采用其 (K, D)，标注 truncated_time=True；
  - 经典文献值抽查（C101 828.94 / C201 591.56）已在此前验证。
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "P5_standard_vrp" / "data" / "solomon"
sys.path.insert(0, str(HERE.parent / "P5_standard_vrp"))
from standard_vrp.core.standard_evaluator import VRPTWInstance, evaluate_vrptw  # noqa: E402


def truncated_feasible(inst, routes):
    """官方 1 位截断行驶时间口径的可行性。"""
    for r in routes:
        load = sum(inst.demands[i] for i in r)
        if load > inst.capacity + 1e-9:
            return False
        t = inst.depot_window[0]
        prev = 0
        for i in r:
            t += math.floor(10 * inst.dist[prev][i]) / 10
            if t > inst.windows[i][1] + 1e-9:
                return False
            if t < inst.windows[i][0]:
                t = inst.windows[i][0]
            t += inst.service[i]
            prev = i
        t += math.floor(10 * inst.dist[prev][0]) / 10
        if t > inst.depot_window[1] + 1e-9:
            return False
    return True


def parse_bks(text):
    routes = []
    for line in text.splitlines():
        if line.startswith("Route #"):
            routes.append(tuple(int(x) for x in line.split(":")[1].split()))
    return routes


def main():
    rows = []
    for sol in sorted((DATA / "bks").glob("*.sol")):
        name = sol.stem
        inst = VRPTWInstance.parse_solomon(DATA / "instances" / (name + ".vrp"))
        routes = parse_bks(sol.read_text(encoding="utf-8"))
        ok, k, d, w, loads = evaluate_vrptw(inst, routes)
        if ok:
            rows.append(dict(instance=name, family=name[:2].upper(), vehicles=k,
                             distance=round(d, 4), truncated_time=False))
        elif truncated_feasible(inst, routes):
            # 截断口径下的距离仍用双精度（SINTEF 展示口径）；K 同
            rows.append(dict(instance=name, family=name[:2].upper(), vehicles=k,
                             distance=round(d, 4), truncated_time=True))
        else:
            rows.append(dict(instance=name, family=name[:2].upper(), vehicles=k,
                             distance=round(d, 4), truncated_time=None))
    assert len(rows) == 56, len(rows)
    truncated = [r for r in rows if r["truncated_time"]]
    print("truncated-time cases:", [r["instance"] for r in truncated])
    # 经典文献抽查
    c101 = next(r for r in rows if r["instance"] == "C101")
    c201 = next(r for r in rows if r["instance"] == "C201")
    print("C101:", c101, "(literature 10/828.94)")
    print("C201:", c201, "(literature 3/591.56)")
    assert c101["vehicles"] == 10 and abs(c101["distance"] - 828.94) < 0.01
    assert c201["vehicles"] == 3 and abs(c201["distance"] - 591.56) < 0.01
    with (DATA / "bks_reference.json").open("w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, note="CVRPLIB官方BKS路线；双精度评价；5例为官方截断时间口径可行",
                       truncated_cases=[r["instance"] for r in truncated]), f, indent=2, ensure_ascii=False)
    print("bks_reference.json written: 56 instances")


if __name__ == "__main__":
    main()
