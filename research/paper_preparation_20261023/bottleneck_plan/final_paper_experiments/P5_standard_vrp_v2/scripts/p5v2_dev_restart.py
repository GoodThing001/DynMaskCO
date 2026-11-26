# -*- coding: utf-8 -*-
"""分步重启开发面板（每步单独 remote_exec，便于定位失败）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PY = "/home/hzeng/envs/MASKCO_env/bin/python"
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
RUN = P5V2 + "/dev01"


def step(client, cmd, label):
    try:
        out = transport.remote_exec(client, cmd)
        print("[%s] %s" % (label, (out or "").strip()[:400]))
        return out
    except Exception as exc:  # noqa: BLE001
        print("[%s] FAILED %s" % (label, exc))
        return None


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/unified_worker.py")
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/future_dispatch.py")
            for f in sorted((V2 / "standard_vrp").rglob("*.py")):
                rel = f.relative_to(V2).as_posix()
                s.put(str(f), P5V2 + "/" + rel)
        print("[upload] worker + package ok")
        step(client, "cd " + RUN + " && rm -f out/*.log status.json runner.log && ls out/*.json | wc -l",
             "clean")
        step(client, "cd " + RUN + " && setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 "
                     "< /dev/null & echo launched", "launch")
        time.sleep(8)
        step(client, "ps -eo pid,etime,args | grep unified_runner | grep -v grep | head -3", "ps")
        step(client, "ls " + RUN + "/status.json && head -c 200 " + RUN + "/status.json", "status")
    finally:
        client.close()


if __name__ == "__main__":
    main()
