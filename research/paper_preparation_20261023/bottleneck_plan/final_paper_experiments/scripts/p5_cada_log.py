# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
echo ===CaDA===
tail -20 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck/out/1_CaDA.log
echo ===RouteFinder===
tail -12 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck/out/1_RouteFinder.log
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
