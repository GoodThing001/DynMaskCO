# -*- coding: utf-8 -*-
"""P5-V2 开发面板 phase 生成 + 部署 + 启动（240 端点：24 实例 × 5 版本 × 2 种子）。

用法：python p5v2_dev_deploy.py [nolaunch]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PY = "/home/hzeng/envs/MASKCO_env/bin/python"
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
VERSIONS = ["STD_OLD", "STD_BUGFIX", "STD_FIX", "STD_NORM", "STD_DIST"]
SEEDS = (71001, 71002)


def build_phase():
    dev = json.loads((V2 / "development/manifest.json").read_text(encoding="utf-8"))
    blocks = []
    for k, row in enumerate(dev["instances"]):
        objective = "cvrp" if row["kind"] == "cvrp" else "vrptw"
        for si, seed in enumerate(SEEDS):
            order = VERSIONS[si % len(VERSIONS):] + VERSIONS[:si % len(VERSIONS)]
            blocks.append(dict(case=row["name"], seed=seed, objective=objective,
                               path=P5V2 + "/development/instances/" + row["file"],
                               methods=list(order)))
    phase = dict(name="p5v2_dev", seconds=60, training_only=False,
                 versions=VERSIONS,
                 blocks=blocks, max_cores=12,
                 interpreters={v: PY for v in VERSIONS},
                 schedule_note="P5-V2 开发面板：24 实例 × 5 版本 × 2 种子 = 240 端点")
    out = V2 / "configs/phase_dev.json"
    out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print("blocks", len(blocks), "cells", len(blocks) * len(VERSIONS))
    return phase


def main(launch=True):
    build_phase()
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            import posixpath

            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass

            # V2 包
            makedirs(P5V2)
            src = V2 / "standard_vrp"
            for f in sorted(src.rglob("*.py")):
                rel = f.relative_to(V2).as_posix()
                cur = P5V2
                for part in rel.split("/")[:-1]:
                    cur = cur + "/" + part
                    makedirs(cur)
                s.put(str(f), P5V2 + "/" + rel)
            # 开发实例
            inst_dir = V2 / "development/instances"
            makedirs(P5V2 + "/development")
            makedirs(P5V2 + "/development/instances")
            for f in sorted(inst_dir.glob("*.vrp")):
                s.put(str(f), P5V2 + "/development/instances/" + f.name)
            # worker + phase
            makedirs(P5V2 + "/dev01")
            s.put(str(HERE / "p5v2_worker.py"), P5V2 + "/dev01/unified_worker.py")
            with s.open(P5V2 + "/dev01/future_dispatch.py", "w") as f:
                f.write((HERE / "p5v2_worker.py").read_bytes())
            s.put(str(V2 / "configs/phase_dev.json"), P5V2 + "/dev01/phase.json")
        cmd = ("mkdir -p " + P5V2 + "/dev01 && cd " + P5V2 + "/dev01 && "
               "cp " + P5 + "/stage01/unified_runner.py . 2>/dev/null; "
               "sed -i 's/timeout=300/timeout=300/' unified_runner.py 2>/dev/null; "
               "rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 5; ps aux | grep unified_runner | grep -v grep | wc -l")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main(not (len(sys.argv) > 1 and sys.argv[1] == "nolaunch"))
