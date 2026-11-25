# -*- coding: utf-8 -*-
"""通用续跑：按已回收结果计算缺失块，生成续跑 phase 并部署启动。

用法: python p5_resume_generic.py <源phase.json> <本地结果目录> <新工作区> [max_cores]
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


def main(phase_src, results_dir, new_ws, cores="6"):
    phase = json.loads(Path(phase_src).read_text(encoding="utf-8"))
    have = set()
    out = Path(results_dir) / "out"
    for f in out.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        have.add((r["case"], r["seed"], r["method"]))
    missing = []
    for b in phase["blocks"]:
        need = [m for m in b["methods"] if (b["case"], b["seed"], m) not in have]
        if need:
            missing.append(dict(b, methods=need))
    resume = dict(phase, name="p5_resume_" + new_ws,
                  blocks=missing, max_cores=int(cores),
                  schedule_note="续跑 %d 块（%s）" % (len(missing), new_ws))
    print("missing blocks:", len(missing),
          "cells:", sum(len(b["methods"]) for b in missing))
    if not missing:
        return
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
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(resume, ensure_ascii=False, indent=2).encode("utf-8"))
        cmd = ("cp " + P5 + "/stage03/unified_runner.py " + P5 + "/" + new_ws + "/ 2>/dev/null; "
               "cd " + P5 + "/" + new_ws + " && rm -rf out status.json runner.log && "
               "tmux new-session -d -s " + new_ws + " 'cd " + P5 + "/" + new_ws + " && " + PY
               + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "6")
