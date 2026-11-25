# -*- coding: utf-8 -*-
"""核验远端 external01/methods 与 cc_compare 环境 python。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
ls -la /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/methods/
echo ===vendor===
ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/methods/_vendor/ 2>/dev/null | head
echo ===cc_compare env===
ls -d /home/hzeng/envs/*/ 2>/dev/null
find /home/hzeng/envs -maxdepth 2 -name 'python' -path '*cc_compare*' 2>/dev/null | head -3
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
