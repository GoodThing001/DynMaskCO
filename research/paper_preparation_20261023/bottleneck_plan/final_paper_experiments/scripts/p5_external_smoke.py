# -*- coding: utf-8 -*-
"""P5 外部方法批次：部署 external01 工作区（PyVRP 冒烟：CVRP + VRPTW 各一格）。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
WORKER = HERE.parent / "P5_standard_vrp/p5_external_worker.py"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"

SMOKE_PHASE = dict(
    name="p5_external_smoke", seconds=60, training_only=False, methods=["PyVRP"],
    interpreters={"PyVRP": PY},
    blocks=[
        dict(case="X-n101-k25", seed=68001, objective="cvrp", methods=["PyVRP"]),
        dict(case="C101", seed=69001, objective="vrptw", methods=["PyVRP"]),
    ],
    max_cores=2,
    schedule_note="P5 外部方法冒烟：PyVRP CVRP（取整口径）+ VRPTW（缩放口径）各一格")


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
            base = P5 + "/external01"
            makedirs(base)
            wb = WORKER.read_bytes()
            s.put(str(WORKER), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(SMOKE_PHASE, ensure_ascii=False, indent=2).encode("utf-8"))
        cmd = ("cp " + P5 + "/stage01/unified_runner.py " + P5 + "/external01/ && cd " + P5
               + "/external01 && rm -rf out status.json runner.log && tmux new-session -d -s p5_external_smoke"
               " 'cd " + P5 + "/external01 && " + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
