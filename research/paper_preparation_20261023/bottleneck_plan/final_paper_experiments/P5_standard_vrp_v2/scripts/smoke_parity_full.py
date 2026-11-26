import sys, random, time, json
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance
from standard_vrp.adapters.objective import adapter_for
from standard_vrp.solver.pool_search import standard_search
import math

def mk(n, seed, cap):
    rng = random.Random(seed)
    coords = {0: (50.0, 50.0)}; dem = {0: 0.0}
    for i in range(1, n + 1):
        coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100)); dem[i] = float(rng.randint(1, 9))
    return CVRPInstance("t", n, cap, coords, dem)

for n, cap in ((60, 30.0), (100, 40.0)):
    inst = mk(n, n, cap)
    ad = adapter_for(inst)
    for variant in ("full",):
        t0 = time.perf_counter()
        out = standard_search(ad, seconds=5, seed=7, heatmap_fn=None, split_enabled=True,
                              slot=False, variant=variant)
        el = time.perf_counter() - t0
        print(n, variant, "D=%.0f K=%d ep=%d it=%d imp=%d hist=%s el=%.2f" % (
            out["distance"], out["vehicles"], out["epochs"], out["iterations"],
            out["improvements"], list(out["history"]), el))
# 有热图（合成热图）验证 related 模式与校准
inst = mk(80, 9, 36.0)
ad = adapter_for(inst)
nn = len(inst.dist)
def heatmap(inst2, routes, removed, _n=None):
    import numpy as np
    rng = np.random.default_rng(0)
    h = rng.random((nn, nn)).astype(float)
    for i in removed: h[i] = rng.random(nn)
    return h
t0 = time.perf_counter()
out = standard_search(ad, seconds=6, seed=11, heatmap_fn=heatmap, split_enabled=True,
                      slot=False, variant="full")
print("heat", "D=%.0f K=%d ep=%d it=%d rc=%s mm=%s el=%.2f" % (
    out["distance"], out["vehicles"], out["epochs"], out["iterations"],
    out["ranking_counts"], out["mask_mode_counts"], time.perf_counter() - t0))
