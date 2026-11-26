# -*- coding: utf-8 -*-
"""开发面板重启（实例协议对齐 + 修正后的 V2 包/执行器 + CVRP 端点沿用）。

用法：python p5v2_dev_relaunch.py
"""
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
        print("[%s] %s" % (label, (out or "").strip()[:300]))
        return out
    except Exception as exc:  # noqa: BLE001
        print("[%s] FAILED %s" % (label, str(exc)[:300]))
        return None


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/unified_worker.py")
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/future_dispatch.py")
            for f in sorted((V2 / "standard_vrp").rglob("*.py")):
                s.put(str(f), P5V2 + "/" + f.relative_to(V2).as_posix())
            for f in sorted((V2 / "development/instances").glob("*.vrp")):
                s.put(str(f), P5V2 + "/development/instances/" + f.name)
            s.put(str(V2 / "development/manifest.json"), P5V2 + "/development/manifest.json")
        print("[upload] package + instances + worker ok")
        step(client, "cd " + RUN + " && for p in $(pgrep -f 'unified_runner[.]py'); do kill $p; done; sleep 1; echo killed", "kill")
        step(client, "cd " + RUN + " && rm -rf out status.json runner.log && echo cleaned", "clean")
        step(client, "cd " + RUN + " && setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 "
                     "< /dev/null & echo launched", "launch")
        time.sleep(10)
        step(client, "ps -eo pid,etime,args | grep unified_runner | grep -v grep | head -2", "ps")
        step(client, "cd " + RUN + " && head -c 160 status.json 2>/dev/null; tail -2 runner.log 2>/dev/null", "status")
    finally:
        client.close()


if __name__ == "__main__":
    main()
