# -*- coding: utf-8 -*-
"""重启 p5b_full（ALNS 截止修复后全批重跑，保持单一代码版本）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"

SHELL = r"""
tmux kill-session -t p5b_full 2>/dev/null || true
sleep 1
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/vrptw_full01
rm -rf out status.json runner.log
tmux new-session -d -s p5b_full 'cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/vrptw_full01 && /home/hzeng/envs/MASKCO_env/bin/python -B unified_runner.py > runner.log 2>&1'
sleep 4
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
