# -*- coding: utf-8 -*-
"""启动 P5-A 各阶段工作区（stage02/stage03）——文件已由 p5a_deploy 部署。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"


def main(stage):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        cmd = ("cd " + P5 + "/" + stage + " && rm -rf out status.json runner.log && "
               "tmux new-session -d -s p5a_" + stage + " 'cd " + P5 + "/" + stage
               + " && " + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1])
