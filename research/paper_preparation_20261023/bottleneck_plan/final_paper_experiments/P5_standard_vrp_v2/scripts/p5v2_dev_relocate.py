# -*- coding: utf-8 -*-
"""把开发面板迁到 /tmp（/home 已满），并清理 /home 上的中间产物。

步骤：
1. 远端确认 /home 容量；
2. 删除 dev01_stash 与 dev01/out（已下载到本地）；
3. 在 /tmp/p5v2_dev 重建运行目录（package 仍在 /home，只读）；
4. 启动 runner，输出写 /tmp。
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
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
RUN = "/tmp/p5v2_dev/dev01"


def step(client, cmd, label):
    try:
        out = transport.remote_exec(client, cmd)
        print("[%s] %s" % (label, (out or "").strip()[:300]))
        return out
    except Exception as exc:  # noqa: BLE001
        print("[%s] FAILED %s" % (label, str(exc)[:200]))
        return None


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        step(client, "df -h /home /tmp | tail -3", "space-before")
        step(client, "cd " + P5V2 + " && for p in $(pgrep -f 'unified_runner[.]py'); do kill $p; done; "
                     "sleep 1; rm -rf dev01_stash dev01/out dev01/status.json dev01/runner.log; "
                     "df -h /home | tail -1", "cleanup")
        step(client, "mkdir -p " + RUN, "mkdir")
        with client.open_sftp() as s:
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/unified_worker.py")
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/future_dispatch.py")
            s.put(str(V2 / "configs/phase_dev.json"), RUN + "/phase.json")
        step(client, "cp " + P5 + "/stage01/unified_runner.py " + RUN + "/ && ls " + RUN, "runner")
        step(client, "cd " + RUN + " && setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 "
                     "< /dev/null & echo launched", "launch")
        time.sleep(10)
        step(client, "ps -eo pid,etime,args | grep unified_runner | grep -v grep | head -2", "ps")
        step(client, "cd " + RUN + " && head -c 160 status.json 2>/dev/null; tail -2 runner.log 2>/dev/null", "status")
        step(client, "df -h /home /tmp | tail -3", "space-after")
    finally:
        client.close()


if __name__ == "__main__":
    main()
