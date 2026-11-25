# -*- coding: utf-8 -*-
"""定位 CVRPLIB Solomon BKS 的距离口径：逐边规则实验。"""
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "P5_standard_vrp" / "data" / "solomon"
sys.path.insert(0, str(HERE.parent / "P5_standard_vrp"))
from standard_vrp.core.standard_evaluator import VRPTWInstance  # noqa: E402


def parse_bks(text):
    routes = []
    for line in text.splitlines():
        if line.startswith("Route #"):
            routes.append(tuple(int(x) for x in line.split(":")[1].split()))
    m = re.search(r"Cost\s+([0-9.]+)", text)
    return routes, float(m.group(1)) if m else None


def main():
    for name in ("C101", "R101", "C201"):
        inst = VRPTWInstance.parse_solomon(DATA / "instances" / (name + ".vrp"))
        routes, cost = parse_bks((DATA / "bks" / (name + ".sol")).read_text(encoding="utf-8"))
        legs = []
        for r in routes:
            prev = 0
            for i in r:
                legs.append((prev, i))
                prev = i
            legs.append((prev, 0))
        raw = sum(math.hypot(inst.coords[a][0] - inst.coords[b][0],
                             inst.coords[a][1] - inst.coords[b][1]) for a, b in legs)
        floor10 = sum(math.floor(10 * math.hypot(inst.coords[a][0] - inst.coords[b][0],
                                                 inst.coords[a][1] - inst.coords[b][1])) / 10 for a, b in legs)
        round10 = sum(round(10 * math.hypot(inst.coords[a][0] - inst.coords[b][0],
                                            inst.coords[a][1] - inst.coords[b][1])) / 10 for a, b in legs)
        ceil10 = sum(math.ceil(10 * math.hypot(inst.coords[a][0] - inst.coords[b][0],
                                              inst.coords[a][1] - inst.coords[b][1])) / 10 for a, b in legs)
        int_round = sum(int(math.hypot(inst.coords[a][0] - inst.coords[b][0],
                                       inst.coords[a][1] - inst.coords[b][1]) + 0.5) for a, b in legs)
        print(name, "official:", cost,
              "raw: %.2f" % raw, "floor1d: %.2f" % floor10, "round1d: %.2f" % round10,
              "ceil1d: %.2f" % ceil10, "intround: %d" % int_round)
        # 时间口径：服务时长并入？官方 C101=828.94（SINTEF）vs 827.3（CVRPLIB）
        # 旅行时间=距离（速度1）？等待+服务？
        total_service = sum(inst.service[i] for i in range(1, inst.dimension + 1))
        print("   service sum:", total_service, "customers:", inst.dimension)


if __name__ == "__main__":
    main()
