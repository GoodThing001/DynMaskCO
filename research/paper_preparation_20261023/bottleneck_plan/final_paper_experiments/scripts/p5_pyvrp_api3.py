# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
/home/hzeng/envs/MASKCO_env/bin/python -B - <<'EOF'
import inspect
import pyvrp
print("pyvrp file:", pyvrp.__file__)
print("ProblemData:", inspect.signature(pyvrp.ProblemData.__init__))
print("has read:", hasattr(pyvrp, "read"))
print("Route attrs:", [n for n in dir(pyvrp.Route) if not n.startswith("_")])
print("Result attrs:", [n for n in dir(pyvrp.Result) if not n.startswith("_")])
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
