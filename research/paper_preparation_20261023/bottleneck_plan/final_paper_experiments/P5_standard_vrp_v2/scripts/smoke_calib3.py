import sys, random, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from standard_vrp.core.standard_evaluator import CVRPInstance
from standard_vrp.adapters.objective import adapter_for
from standard_vrp.solver.pool_search import standard_search
rng = random.Random(5)
coords = {0: (50.0, 50.0)}; dem = {0: 0.0}
for i in range(1, 81):
    coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100)); dem[i] = float(rng.randint(1, 9))
inst = CVRPInstance("t", 80, 36.0, coords, dem); ad = adapter_for(inst)
n = inst.dimension
def heatmap(inst2, routes, removed, _n=None):
    r = np.random.default_rng(len(removed))
    h = r.random((n + 1, n + 1))
    for i in removed:
        h[i] = r.random(n + 1)
    return h
for secs in (6.0,):
    out = standard_search(ad, seconds=secs, seed=3, heatmap_fn=heatmap, split_enabled=True, slot=False)
    print("D=%.0f K=%d ep=%d rc=%s mm=%s fb=%s" % (out["distance"], out["vehicles"], out["epochs"],
          out["ranking_counts"], out["mask_mode_counts"], out["fallbacks"]))
