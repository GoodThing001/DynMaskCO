import sys, random
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import importlib.util as iu
spec = iu.spec_from_file_location("fx", "scripts/p5v2_fixed_candidates.py")
m = iu.module_from_spec(spec); spec.loader.exec_module(m)
from standard_vrp.core.standard_evaluator import CVRPInstance
from standard_vrp.adapters.objective import adapter_for
from standard_vrp.solver.pool_search import nn_witness
rng = random.Random(3)
coords = {0: (50.0, 50.0)}; dem = {0: 0.0}
for i in range(1, 31):
    coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100)); dem[i] = float(rng.randint(1, 9))
inst = CVRPInstance("t", 30, 40.0, coords, dem)
ad = adapter_for(inst)
init = [tuple(r) for r in nn_witness(inst)]
print("init", ad.evaluate(init))
mask = m.select_mask(inst, init, 7)
print("mask", len(mask), sorted(mask)[:5])
mset = set(mask)
kept = [tuple(c for c in r if c not in mset) for r in init]
kept = [r for r in kept if r]
print("kept", ad.evaluate(kept), len(kept))
cands = m.candidates_for(ad, kept, mask[0])
print("cands", len(cands), cands[0][0], cands[-1][0], [round(m.scalar(c[3]),2) for c in cands[:3]])
pick = m.pick_distance(ad, cands)
print("oracle rank of oracle", m.rank_of(ad, cands, pick))
print("second", m.rank_of(ad, cands, sorted(cands, key=lambda c: -m.scalar(c[3]))[1]))
# 插入执行
routes = [tuple(x) for x in kept]
routes = ([tuple(x) for x in routes] + [tuple(pick[4])] if pick[0]=="new" else [tuple(pick[4]) if i==pick[1] else r for i,r in enumerate(routes)])
print("after one insert", ad.evaluate(routes))
