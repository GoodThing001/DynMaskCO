# -*- coding: utf-8 -*-
"""P5 作者模型批次部署与启动（external01 工作区，phase_external_author，696 格）。

- runner 子进程超时 300s → 1500s（RRNCO 作者默认单核 ~490s；构造式预算档 ≤1200s 已在 worker check 内）；
- 核验 8 个方法模块全部就位后启动；PyVRP 批次与作者批次共用 external01，
  按 phase 顺序切换（先 PyVRP 后作者；本脚本仅部署作者 phase）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
MODULES = ["am.py", "pomo.py", "deepaco.py", "car.py", "mvmoe.py", "cada.py", "routefinder.py", "rrnco.py"]

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01
ls methods/
sed -i 's/timeout=300/timeout=1500/' unified_runner.py
grep -n 'timeout=' unified_runner.py
"""


def main(launch=True):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"),
                  P5 + "/external01/unified_worker.py")
            with s.open(P5 + "/external01/future_dispatch.py", "w") as f:
                f.write(wb)
            s.put(str(HERE.parent / "P5_standard_vrp/phase_external_author.json"),
                  P5 + "/external01/phase.json")
        out = transport.remote_exec(client, SHELL)
        print(out)
        missing = [m for m in MODULES if m not in out]
        if missing:
            raise RuntimeError("missing method modules: " + ", ".join(missing))
        if launch:
            cmd = ("cd " + P5 + "/external01 && rm -rf out status.json runner.log && "
                   "tmux new-session -d -s p5_external_author 'cd " + P5 + "/external01 && "
                   + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
            print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main(launch=not (len(sys.argv) > 1 and sys.argv[1] == "nolaunch"))
