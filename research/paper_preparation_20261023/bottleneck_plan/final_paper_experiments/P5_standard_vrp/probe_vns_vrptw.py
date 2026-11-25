# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import VRPTWInstance, evaluate_vrptw, route_schedule
from standard_vrp.solver.standard_pool_search import nn_witness
from standard_vrp.solver.standard_classical import vns_search, _relocate, _swap, _two_opt, _shake, _score

inst = VRPTWInstance.parse_solomon(HERE / "data/solomon/instances/C101.vrp")
w = nn_witness(inst)
ok0, k0, d0, _, _ = evaluate_vrptw(inst, w)
print("witness:", ok0, k0, d0)

t0 = time.perf_counter()
out = vns_search(inst, seconds=10, seed=69001, objective="vrptw")
print("vns elapsed %.1fs" % (time.perf_counter() - t0))
ok, k, d, w_, loads = evaluate_vrptw(inst, out["routes"])
print("final eval:", ok, k, d, "reported:", out["vehicles"], out["distance"])
if not ok:
    for ri, r in enumerate(out["routes"]):
        sched = route_schedule(inst, tuple(r))
        if sched is None:
            cap = sum(inst.demands[i] for i in r)
            print("route", ri, "len", len(r), "load", cap, "> cap?", cap > inst.capacity + 1e-9)
            # 检查时窗
            t = inst.depot_window[0]
            for idx, i in enumerate(r):
                travel = inst.dist[0 if idx == 0 else r[idx - 1]][i]
                t += travel
                if t < inst.windows[i][0]:
                    t = inst.windows[i][0]
                if t > inst.windows[i][1] + 1e-9:
                    print("  window violated at customer", i, "t=%.2f" % t, "win", inst.windows[i])
                    break
                t += inst.service[i]
            t += inst.dist[r[-1]][0]
            if t > inst.depot_window[1] + 1e-9:
                print("  depot window violated, end=%.2f" % t)
