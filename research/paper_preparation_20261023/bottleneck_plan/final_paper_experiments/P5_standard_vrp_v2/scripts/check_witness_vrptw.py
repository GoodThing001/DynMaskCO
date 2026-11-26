import sys, json
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from standard_vrp.core.standard_evaluator import VRPTWInstance
from standard_vrp.adapters.objective import adapter_for
from standard_vrp.solver.pool_search import nn_witness
man = json.loads(Path("development/manifest.json").read_text(encoding="utf-8"))
for row in man["instances"]:
    if row["kind"] != "vrptw":
        continue
    inst = VRPTWInstance.parse_solomon(Path("development/instances") / row["file"])
    ad = adapter_for(inst)
    w = [tuple(r) for r in nn_witness(inst)]
    ok, d, k = ad.evaluate(w)
    print("%-24s n=%d cap=%.0f K=%d ok=%s D=%.2f" % (row["name"], inst.dimension, inst.capacity, k, ok, d))
