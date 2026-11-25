# -*- coding: utf-8 -*-
import sys
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp
from standard_vrp.solver.standard_pool_search import masked_repair, nn_witness, rank_customers

vrp = HERE / "data/cvrplib/instances/X-n101-k25.vrp"
inst = CVRPInstance.parse_vrp(vrp.read_text(encoding="utf-8"))
best = nn_witness(inst)
ok0, d0, _, k0 = evaluate_cvrp(inst, best, rounded=True)
print("witness: K=%d D=%.2f" % (k0, d0))

rng = random.Random(68001)
nrng = __import__("numpy").random.default_rng(68001)
improved_count = 0
best_overall = d0
for epoch in range(12):
    mask = nrng.choice(__import__("numpy").arange(1, inst.dimension + 1), size=min(40, inst.dimension), replace=False).tolist()
    removed = set(mask)
    partial = tuple(tuple(i for i in r if i not in removed) for r in best)
    partial = tuple(r for r in partial if r)
    _, p0, _, _ = evaluate_cvrp(inst, partial, rounded=True)
    t0 = time.perf_counter()
    order = rank_customers(inst, partial, removed, "distance", None, rng)
    new, improved = masked_repair(inst, partial, removed, order, "greedy", rng, "cvrp", heat=None, slot=True)
    dt = time.perf_counter() - t0
    _, d, _, k = evaluate_cvrp(inst, new, rounded=True)
    same = tuple(new) == tuple(best)
    print("epoch %2d: partial D=%.2f -> repaired D=%.2f K=%d improved=%s identical_to_best=%s (%.2fs)"
          % (epoch, p0, d, k, improved, same, dt))
    if improved:
        improved_count += 1
        if d < best_overall:
            best_overall = d
print("improved epochs: %d/12; best_overall %.2f vs witness %.2f" % (improved_count, best_overall, d0))
