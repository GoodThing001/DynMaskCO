# -*- coding: utf-8 -*-
"""续跑（nohup 版）：删除截断 JSON、计算缺失块、生成续跑 phase、setsid 后台启动。

用法: python p5_resume2.py <phase.json> <results_dir> <new_ws> [cores]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
INTERNAL = {"ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT", "ALNS", "VNS"}


def main(phase_src, results_dir, new_ws, cores="4"):
    phase = json.loads(Path(phase_src).read_text(encoding="utf-8"))
    internal = any(m in INTERNAL for m in phase["methods"])
    worker_file = "p5_worker.py" if internal else "p5_external_worker.py"
    out = Path(results_dir) / "out"
    have = set()
    corrupt = 0
    for f in out.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            f.unlink()
            corrupt += 1
            continue
        have.add((r["case"], r["seed"], r["method"]))
    missing = []
    for b in phase["blocks"]:
        need = [m for m in b["methods"] if (b["case"], b["seed"], m) not in have]
        if need:
            missing.append(dict(b, methods=need))
    cells = sum(len(b["methods"]) for b in missing)
    print("corrupt deleted:", corrupt, "missing blocks:", len(missing), "cells:", cells)
    if not missing:
        return
    resume = dict(phase, name="p5_resume_" + new_ws, blocks=missing,
                  max_cores=int(cores), schedule_note="续跑 %d 块" % len(missing))
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            base = P5 + "/" + new_ws
            makedirs(base)
            wb = (HERE.parent / "P5_standard_vrp" / worker_file).read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp" / worker_file), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(resume, ensure_ascii=False, indent=2).encode("utf-8"))
        cmd = ("cp " + P5 + "/stage03/unified_runner.py " + P5 + "/" + new_ws + "/ 2>/dev/null; "
               + ("sed -i 's/timeout=300/timeout=1500/' " + P5 + "/" + new_ws + "/unified_runner.py 2>/dev/null; "
                  if not internal else "")
               + "cd " + P5 + "/" + new_ws + " && rm -rf out status.json runner.log && "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 5; ps aux | grep unified_runner | grep -v grep | wc -l")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "4")
