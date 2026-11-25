# -*- coding: utf-8 -*-
"""定位 run_dev72_clock.py 及 compare 链路径。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import json
from pathlib import Path
cands = [
    Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run_dev72_clock.py"),
    Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run10_six_methods_dev72/recovery_preparation/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run_dev72_clock.py"),
    Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run10_six_methods_dev72/recovery_preparation/workspace/C-VRP_Cold-chainVehicleRoutingProblem/tools/run_dev72_clock.py"),
    Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/run_dev72_clock.py"),
]
for p in cands:
    print(p.is_file(), p)
ws = Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit")
print("workspace scripts files:", sorted(x.name for x in ws.iterdir() if "clock" in x.name or "run_" in x.name))
'''


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE))
    finally:
        client.close()
    print(out)


if __name__ == "__main__":
    main()
