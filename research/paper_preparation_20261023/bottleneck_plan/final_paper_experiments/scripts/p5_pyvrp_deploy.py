# -*- coding: utf-8 -*-
"""P5 PyVRP 批次部署与启动（external01，phase_external_pyvrp，468 格）。

worker 与 phase 均兼容 PyVRP/作者两批（作者批另有 phase 切换 + runner 超时放宽）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"),
                  P5 + "/external01/unified_worker.py")
            with s.open(P5 + "/external01/future_dispatch.py", "w") as f:
                f.write(wb)
            s.put(str(HERE.parent / "P5_standard_vrp/phase_external_pyvrp.json"),
                  P5 + "/external01/phase.json")
        cmd = ("cd " + P5 + "/external01 && sed -i 's/timeout=1500/timeout=300/' unified_runner.py; "
               "rm -rf out status.json runner.log && tmux new-session -d -s p5_external_pyvrp 'cd "
               + P5 + "/external01 && " + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
