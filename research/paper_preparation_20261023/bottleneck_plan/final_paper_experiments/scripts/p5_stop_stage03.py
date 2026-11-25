# -*- coding: utf-8 -*-
"""停止 stage03 与 PyVRP 批次（车队约束修复后重跑）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
tmux kill-session -t p5a_stage03 2>/dev/null
tmux kill-session -t p5_external_pyvrp 2>/dev/null
sleep 2
tmux ls
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
