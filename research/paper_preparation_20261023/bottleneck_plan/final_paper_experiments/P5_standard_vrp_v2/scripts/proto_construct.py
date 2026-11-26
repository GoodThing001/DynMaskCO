import sys, json, statistics as st
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from standard_vrp.core.standard_evaluator import VRPTWInstance, route_schedule
man = json.loads(Path("development/manifest.json").read_text(encoding="utf-8"))

def construct(inst, key):
    order = sorted(range(1, inst.dimension + 1), key=key)
    routes = []
    for cust in order:
        placed = False
        if routes:
            r = routes[-1]
            for pos in range(len(r) + 1):
                trial = r[:pos] + [cust] + r[pos:]
                if route_schedule(inst, tuple(trial)) is not None:
                    routes[-1] = trial
                    placed = True
                    break
        if not placed:
            routes.append([cust])
    return routes

def eval_routes(inst, routes):
    tot = 0.0
    for r in routes:
        ok = route_schedule(inst, tuple(r)) is not None
        if not ok:
            return None, False, 0
        prev = 0
        for c in r:
            tot += inst.dist[prev][c]; prev = c
        tot += inst.dist[prev][0]
    return tot, True, len(routes)

keys = {
    "due": lambda i: (inst.windows[i][1], inst.windows[i][0], i),
    "ready": lambda i: (inst.windows[i][0], inst.windows[i][1], i),
}
for row in man["instances"]:
    if row["kind"] != "vrptw":
        continue
    inst = VRPTWInstance.parse_solomon(Path("development/instances") / row["file"])
    line = "%-22s k_manifest=%2d " % (row["name"], row["k"])
    for name, kf in keys.items():
        routes = construct(inst, kf)
        tot, ok, K = eval_routes(inst, routes)
        line += " %s:K=%2d ok=%-5s" % (name, K, ok)
    print(line)
