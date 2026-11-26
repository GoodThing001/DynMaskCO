# -*- coding: utf-8 -*-
"""归档补全：抓取各运行目录的逐端点日志与 runner 记录（远端 /tmp 为易失分区）。

落地：<local>/logs/*.log（与 out/ 并列），不进入 git，仅进本地归档包。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

JOBS = [
    ("/tmp/p5v2_dev/dev01", V2 / "development_results/dev01_final"),
    ("/tmp/p5v2_formal/cvrp", V2 / "formal_results/cvrp"),
    ("/tmp/p5v2_formal/vrptw", V2 / "formal_results/vrptw"),
    ("/tmp/p5v2_formal/norm", V2 / "formal_results/norm"),
    ("/tmp/p5v2_formal/ext_pyvrp", V2 / "external_results/pyvrp_vrptw"),
    ("/tmp/p5v2_formal/ext_pyvrp2", V2 / "external_results/pyvrp_vrptw"),
    ("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01",
     V2 / "diagnostics/dev01"),
]


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    total = 0
    try:
        with client.open_sftp() as s:
            for remote, local in JOBS:
                try:
                    names = s.listdir(remote + "/out")
                except OSError:
                    print("skip (no out):", remote)
                    continue
                dest = local / "logs"
                dest.mkdir(parents=True, exist_ok=True)
                got = 0
                for name in names:
                    if not name.endswith(".log"):
                        continue
                    try:
                        s.get(remote + "/out/" + name, str(dest / name))
                        got += 1
                    except OSError as exc:
                        print("skip", remote, name, exc)
                for extra in ("runner.log", "status.json", "phase.json"):
                    try:
                        s.get(remote + "/" + extra, str(local / ("_runner_" + extra)))
                    except OSError:
                        pass
                total += got
                print("%-58s logs=%d" % (str(local.relative_to(V2)), got))
    finally:
        client.close()
    print("total logs fetched:", total)


if __name__ == "__main__":
    main()
