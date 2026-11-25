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
from standard_vrp.core.standard_evaluator import CVRPInstance
import pyvrp
from pyvrp.stop import MaxRuntime

inst = CVRPInstance.parse_vrp(open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/cvrplib/instances/X-n101-k25.vrp").read())
print("capacity:", inst.capacity, "dimension:", inst.dimension)
print("demands 1..5:", [inst.demands[i] for i in range(1, 6)])
print("demand sum:", sum(inst.demands.values()), "max:", max(inst.demands.values()))
n = inst.dimension
dist = [[0] * (n + 1) for _ in range(n + 1)]
for i in range(0, n + 1):
    for j in range(0, n + 1):
        if i != j:
            dist[i][j] = int(inst.edge_distance(i, j, rounded=True))
depot = pyvrp.Depot(x=int(round(inst.coords[0][0] * 1000)), y=int(round(inst.coords[0][1] * 1000)))
clients = [pyvrp.Client(x=int(round(inst.coords[i][0] * 1000)), y=int(round(inst.coords[i][1] * 1000)),
                        delivery=[int(round(inst.demands[i]))]) for i in range(1, n + 1)]
vehicle = pyvrp.VehicleType(num_available=n, capacity=[int(round(inst.capacity))])
data = pyvrp.ProblemData(clients, [depot], [vehicle], [dist], [dist])
print("data num_clients:", data.num_clients, "num_depots:", data.num_depots, "num_vehicles:", data.num_vehicles)
model = pyvrp.Model.from_data(data)
res = model.solve(stop=MaxRuntime(10), seed=1, display=False)
for route in res.best.routes():
    v = [int(x) + 1 for x in route.visits()]
    load = sum(inst.demands[c] for c in v)
    print("route:", v[:8], "load:", load, "dist-ours:", sum(int(inst.edge_distance(0 if i == 0 else v[i-1], v[i], rounded=True)) for i in range(len(v))) if False else "?")
print("res cost:", res.cost(), "feasible:", res.is_feasible())
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
