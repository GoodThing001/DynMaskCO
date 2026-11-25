# -*- coding: utf-8 -*-
"""预检 p5_external_worker 作者模型调度（AM/POMO/DeepACO/CaR 各一格）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01
mkdir -p out methods
ls methods/
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"),
                  "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/unified_worker.py")
            with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/future_dispatch.py", "w") as f:
                f.write(wb)
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
