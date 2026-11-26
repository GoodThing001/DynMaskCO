# -*- coding: utf-8 -*-
"""部署并启动固定候选数量诊断（24 开发实例 × 2 种子 = 48 单元）。

落地 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01。
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
RUN = P5V2 + "/diag01"
SEEDS = (71001, 71002)


def build(launch=True):
    dev = json.loads((V2 / "development/manifest.json").read_text(encoding="utf-8"))
    blocks = []
    for row in dev["instances"]:
        for seed in SEEDS:
            blocks.append(dict(case=row["name"], seed=seed,
                               objective="cvrp" if row["kind"] == "cvrp" else "vrptw",
                               methods=["DIAG"]))
    phase = dict(name="p5v2_fixed_candidates", seconds=60, training_only=False,
                 methods=["DIAG"], blocks=blocks, max_cores=8,
                 interpreters={"DIAG": PY},
                 schedule_note="固定候选数量诊断：24 实例 × 2 种子，4 种排序同候选集")
    out = V2 / "configs/phase_diag.json"
    out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print("blocks", len(blocks))
    if not launch:
        return phase

    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                cur = ""
                for part in path.strip("/").split("/"):
                    cur += "/" + part
                    try:
                        s.mkdir(cur)
                    except OSError:
                        pass

            makedirs(RUN)
            makedirs(RUN + "/instances")
            for f in sorted((V2 / "development/instances").glob("*.vrp")):
                s.put(str(f), RUN + "/instances/" + f.name)
            s.put(str(V2 / "development/manifest.json"), RUN + "/manifest.json")
            s.put(str(out), RUN + "/phase.json")
            s.put(str(HERE / "p5v2_diag_worker.py"), RUN + "/unified_worker.py")
            s.put(str(HERE / "p5v2_diag_worker.py"), RUN + "/future_dispatch.py")
            s.put(str(HERE / "p5v2_fixed_candidates.py"), RUN + "/p5v2_fixed_candidates.py")
        cmd = ("mkdir -p " + RUN + " && cd " + RUN + " && "
               "cp /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage01/unified_runner.py . "
               "2>/dev/null; rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 3; pgrep -fc 'diag01' || true")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()
    return phase


if __name__ == "__main__":
    build(not (len(sys.argv) > 1 and sys.argv[1] == "nolaunch"))
