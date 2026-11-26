# -*- coding: utf-8 -*-
"""外部基线补齐：PyVRP VRPTW 168 端点（56 例 × 种子 69001-69003）。

沿用冻结外部工作器（P5-V1 p5_external_worker.py，只读复制）与 V1 评价器口径：
整数时间矩阵 + MaxRuntime(56)，双精度复发评价，8 个截断实例允许 TRUNCATED_OK。
落地 /tmp/p5v2_formal/ext_pyvrp（/home 无空间）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
P5V1 = V2.parent / "P5_standard_vrp"
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PY = "/home/hzeng/envs/MASKCO_env/bin/python"
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
RUN = "/tmp/p5v2_formal/ext_pyvrp"
SEEDS = (69001, 69002, 69003)
METHOD = "PyVRP"


def build():
    names = sorted(p.stem for p in (V2 / "data/solomon/instances").glob("*.vrp"))
    assert len(names) == 56, len(names)
    blocks = []
    for name in names:
        for seed in SEEDS:
            blocks.append(dict(case=name, seed=seed, objective="vrptw", methods=[METHOD]))
    phase = dict(name="p5v2_external_pyvrp_vrptw", seconds=60, training_only=False,
                 methods=[METHOD], blocks=blocks, max_cores=8,
                 interpreters={METHOD: PY},
                 schedule_note="外部基线补齐：PyVRP VRPTW 56 例 × 3 种子")
    out = V2 / "configs/phase_external_pyvrp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print("blocks", len(blocks), "cells", len(blocks))
    return phase


def deploy(phase):
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
            s.put(str(V2 / "configs/phase_external_pyvrp.json"), RUN + "/phase.json")
            s.put(str(P5V1 / "p5_external_worker.py"), RUN + "/unified_worker.py")
            s.put(str(P5V1 / "p5_external_worker.py"), RUN + "/future_dispatch.py")
        print("deployed to", RUN)
    finally:
        client.close()


def launch():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        cmd = ("cd " + RUN + " && cp " + P5 + "/stage01/unified_runner.py . 2>/dev/null; "
               "rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 3; ps aux | grep -c 'ext_pyvrp'")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ph = build()
    if what in ("deploy", "all"):
        deploy(ph)
    if what in ("launch", "all"):
        launch()
