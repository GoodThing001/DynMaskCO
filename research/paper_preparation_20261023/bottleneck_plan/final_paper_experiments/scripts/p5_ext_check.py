# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
tmux ls 2>&1 || true
echo ===RUNNERLOG===
cat /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/runner.log 2>/dev/null | head -20
echo ===STATUS===
cat /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/status.json 2>/dev/null | head -c 300 || true
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
