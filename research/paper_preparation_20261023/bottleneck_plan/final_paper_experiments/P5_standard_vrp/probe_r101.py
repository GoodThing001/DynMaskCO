# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import (VRPTWInstance, evaluate_vrptw, route_schedule)
from standard_vrp.solver.standard_pool_search import nn_witness

inst = VRPTWInstance.parse_solomon(HERE / "data/solomon/instances/R101.vrp")
w = nn_witness(inst)
ok, k, d, _, _ = evaluate_vrptw(inst, w)
print("R101 witness: ok=%s K=%d D=%.2f" % (ok, k, d))
seen = set()
for r in w:
    seen.update(r)
missing = sorted(set(range(1, inst.dimension + 1)) - seen)
print("missing customers:", missing)
print("their windows:", [(i, inst.windows[i]) for i in missing])
# 尝试把这些客户插入 25 条路线的最佳可行位置
for cust in missing:
    best = None
    for ri, _r in enumerate(w):
        r = list(_r)
        for pos in range(len(r) + 1):
            nr = tuple(r[:pos] + [cust] + r[pos:])
            if route_schedule(inst, nr) is None:
                continue
            delta = (inst.edge_distance(([0] + r + [0])[pos], cust) +
                     inst.edge_distance(cust, ([0] + r + [0])[pos + 1]) -
                     inst.edge_distance(([0] + r + [0])[pos], ([0] + r + [0])[pos + 1]))
            if best is None or delta < best[0]:
                best = (delta, ri, pos)
    print("cust", cust, "window", inst.windows[cust], "direct insert:", best)
