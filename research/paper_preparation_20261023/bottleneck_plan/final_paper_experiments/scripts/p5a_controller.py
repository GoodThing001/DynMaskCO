# -*- coding: utf-8 -*-
"""检查 P5A 冒烟控制器日志与进程。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, "tail -c 1200 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/smoke01/controller.log; echo ===; tmux ls")
        print(out)
    finally:
        client.close()

if __name__ == "__main__":
    main()
