# -*- coding: utf-8 -*-
"""正式面板与诊断进度总览：各 stage 的 cells / failed / 运行进程数。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

DIRS = dict(cvrp="/tmp/p5v2_formal/cvrp",
            vrptw="/tmp/p5v2_formal/vrptw",
            norm="/tmp/p5v2_formal/norm",
            dev="/tmp/p5v2_dev/dev01")


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        cmd = ["echo \"[now] $(date -u +%H:%M:%S)\""]
        for name, d in DIRS.items():
            cmd.append("echo \"[%s] json=$(ls %s/out/*.json 2>/dev/null | wc -l) "
                       "failed=$(grep -o 'failed.: .[^]]*' %s/status.json 2>/dev/null | head -c 160)\""
                       % (name, d, d))
        cmd.append("echo '[workers]'; ps -eo etime,args | grep -c 'unified_worker' || true")
        print(transport.remote_exec(client, "; ".join(cmd)))
    finally:
        client.close()


if __name__ == "__main__":
    main()
