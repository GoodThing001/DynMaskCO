# -*- coding: utf-8 -*-
"""归档前远端收尾：确认无残留作业，并统计各运行目录的产物数量。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

RUNS = {
    "dev": "/tmp/p5v2_dev/dev01",
    "cvrp": "/tmp/p5v2_formal/cvrp",
    "vrptw": "/tmp/p5v2_formal/vrptw",
    "norm": "/tmp/p5v2_formal/norm",
    "ext1": "/tmp/p5v2_formal/ext_pyvrp",
    "ext2": "/tmp/p5v2_formal/ext_pyvrp2",
    "diag": "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01",
}


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        cmd = ["echo \"[now] $(date -u +%Y-%m-%dT%H:%M:%SZ)\"",
               "echo '[runners]'; pgrep -af 'unified_runner[.]py' || echo none",
               "echo '[workers]'; pgrep -af 'unified_worker[.]py' || echo none",
               "echo '[pyvrp]'; pgrep -af 'p5v2' || echo none",
               "echo '[space]'; df -h /home /tmp | tail -2"]
        for name, d in RUNS.items():
            cmd.append("echo \"[%s] json=$(ls %s/out/*.json 2>/dev/null | wc -l) "
                       "log=$(ls %s/out/*.log 2>/dev/null | wc -l)\"" % (name, d, d))
        print(transport.remote_exec(client, "; ".join(cmd)))
    finally:
        client.close()


if __name__ == "__main__":
    main()
