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
print("version:", getattr(pyvrp, "__version__", "?"))
print(inspect.signature(pyvrp.Model.from_data))
print([n for n in dir(pyvrp) if n in ("Model", "SolveParams", "Result", "Route")])
import pyvrp.stop as st
print("stop:", [n for n in dir(st) if not n.startswith("_")])
print(inspect.signature(pyvrp.Model.solve))
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
