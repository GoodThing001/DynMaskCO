# -*- coding: utf-8 -*-
"""开发面板断点续跑：上传修正后的执行器，清掉 .log，保留已成端点，重启 runner。

用法：python p5v2_dev_resume.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PY = "/home/hzeng/envs/MASKCO_env/bin/python"
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
RUN = P5V2 + "/dev01"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            for f in sorted((V2 / "standard_vrp").rglob("*.py")):
                rel = f.relative_to(V2).as_posix()
                target = P5V2 + "/" + rel
                cur = ""
                for part in target.rsplit("/", 1)[0].strip("/").split("/"):
                    cur += "/" + part
                    try:
                        s.mkdir(cur)
                    except OSError:
                        pass
                s.put(str(f), target)
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/unified_worker.py")
            s.put(str(HERE / "p5v2_worker.py"), RUN + "/future_dispatch.py")
        cmd = ("cd " + RUN + " && "
               "for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done; "
               "for p in $(pgrep -f 'unified_runner[.]py'); do kill $p 2>/dev/null; done; "
               "sleep 2; mkdir -p ../dev01_stash && cp -n out/*.json ../dev01_stash/ 2>/dev/null; "
               "echo \"[stashed] $(ls ../dev01_stash/*.json 2>/dev/null | wc -l)\"; "
               "rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 5; ps -eo pid,etime,args | grep 'unified_runner' | grep -v grep | head -2")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
