# -*- coding: utf-8 -*-
"""P5-B 正式批次部署与启动：vrptw_full01 工作区（56 例 × 3 种子 × 8 法 = 1344 格）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
WORKER = HERE.parent / "P5_standard_vrp/p5_worker.py"
PHASE = HERE.parent / "P5_standard_vrp/phase_vrptw_full.json"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            base = P5 + "/vrptw_full01"
            makedirs(base)
            wb = WORKER.read_bytes()
            s.put(str(WORKER), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            s.put(str(PHASE), base + "/phase.json")
        cmd = ("cp " + P5 + "/stage01/unified_runner.py " + P5 + "/vrptw_full01/ && cd " + P5
               + "/vrptw_full01 && rm -rf out status.json runner.log && tmux new-session -d -s p5b_full"
               " 'cd " + P5 + "/vrptw_full01 && " + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
