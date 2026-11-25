# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01
JAX_PLATFORMS=cpu /home/hzeng/envs/MASKCO_env/bin/python -B - <<'EOF'
import sys
sys.path.insert(0, "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp
import unified_worker as w

inst = CVRPInstance.parse_vrp(open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/cvrplib/instances/X-n101-k25.vrp").read())
routes = w.solve_pyvrp(inst, "cvrp", 68001)
print("n routes:", len(routes))
print("customers:", sum(len(r) for r in routes), "/", inst.dimension)
print("first routes:", routes[:3])
ok, d, loads, k = evaluate_cvrp(inst, routes, rounded=True)
print("eval:", ok, d, k)
if not ok:
    seen = set()
    for r in routes:
        for i in r:
            if i in seen:
                print("DUP customer", i)
            seen.add(i)
    missing = set(range(1, inst.dimension + 1)) - seen
    print("missing:", sorted(missing)[:20])
    bad = [(l, inst.capacity) for l in loads if l > inst.capacity + 1e-9]
    print("over capacity:", bad[:5])
EOF
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
