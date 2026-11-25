# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
B=/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5
echo ===stage03c worker log===
tail -14 $B/stage03c/out/0_ORDER_SPLIT_D.log 2>/dev/null
echo ===author3 worker log===
tail -10 $B/author3/out/7_MVMoE.log 2>/dev/null
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
