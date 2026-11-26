import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from standard_vrp.core.standard_evaluator import CVRPInstance
from standard_vrp.solver.pool_search import nn_witness

man = json.loads(Path("development/manifest.json").read_text(encoding="utf-8"))
stash = Path("development_results/dev01_prefix_D6/out")
rows = {}
for f in stash.glob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    rows.setdefault(r["case"], r["initial_routes"])
checked = 0
same = 0
for row in man["instances"]:
    if row["name"] not in rows or row["kind"] != "cvrp":
        continue
    inst = CVRPInstance.parse_vrp(Path("development/instances", row["file"]).read_text(encoding="utf-8"))
    w = [list(r) for r in nn_witness(inst)]
    eq = w == [list(r) for r in rows[row["name"]]]
    checked += 1
    same += int(eq)
    print("%-22s initial identical: %s" % (row["name"], eq))
print("identical %d/%d" % (same, checked))
