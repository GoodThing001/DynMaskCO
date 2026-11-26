# -*- coding: utf-8 -*-
"""统一重启：修复 D8 后重跑正式面板（cvrp/vrptw/norm）、开发面板、固定候选诊断。

所有输出目录都在 /tmp（/home 仅 100MB 余量）。清理只发生在本次运行目录内。
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
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
FORMAL = "/tmp/p5v2_formal"
DEV = "/tmp/p5v2_dev/dev01"
DIAG = P5V2 + "/diag01"
STAGES = [("cvrp", FORMAL + "/cvrp"), ("vrptw", FORMAL + "/vrptw"), ("norm", FORMAL + "/norm")]


def step(client, cmd, label):
    try:
        out = transport.remote_exec(client, cmd)
        print("[%s] %s" % (label, (out or "").strip()[:200]))
        return out
    except Exception as exc:  # noqa: BLE001
        print("[%s] FAILED %s" % (label, str(exc)[:200]))
        return None


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        step(client, "for p in $(pgrep -f 'unified_runner[.]py'); do kill $p 2>/dev/null; done; "
                     "for p in $(pgrep -f 'unified_worker[.]py'); do kill $p 2>/dev/null; done; "
                     "sleep 2; echo stopped", "stop")
        with client.open_sftp() as s:
            for f in sorted((V2 / "standard_vrp").rglob("*")):
                if f.suffix not in (".py", ".json"):
                    continue
                target = P5V2 + "/" + f.relative_to(V2).as_posix()
                cur = ""
                for part in target.rsplit("/", 1)[0].strip("/").split("/"):
                    cur += "/" + part
                    try:
                        s.mkdir(cur)
                    except OSError:
                        pass
                s.put(str(f), target)
            for name, d in STAGES:
                s.put(str(HERE / "p5v2_worker.py"), d + "/unified_worker.py")
                s.put(str(HERE / "p5v2_worker.py"), d + "/future_dispatch.py")
            s.put(str(HERE / "p5v2_worker.py"), DEV + "/unified_worker.py")
            s.put(str(HERE / "p5v2_worker.py"), DEV + "/future_dispatch.py")
            s.put(str(HERE / "p5v2_fixed_candidates.py"), DIAG + "/p5v2_fixed_candidates.py")
        print("[upload] package + workers ok")
        if what in ("all", "formal"):
            for name, d in STAGES:
                step(client, "cd " + d + " && rm -rf out status.json runner.log && "
                             "cp " + P5 + "/stage01/unified_runner.py . 2>/dev/null; echo clean-" + name,
                     "clean-" + name)
            for i, (name, d) in enumerate(STAGES):
                step(client, "cd " + d + " && setsid nohup " + PY + " -B unified_runner.py "
                             "> runner.log 2>&1 < /dev/null & echo launched-" + name, "launch-" + name)
                if i + 1 < len(STAGES):
                    time.sleep(25)
        if what in ("all", "dev"):
            step(client, "cd " + DEV + " && rm -rf out status.json runner.log && echo clean-dev", "clean-dev")
            step(client, "cd " + DEV + " && setsid nohup " + PY + " -B unified_runner.py "
                         "> runner.log 2>&1 < /dev/null & echo launched-dev", "launch-dev")
        if what in ("all", "diag"):
            step(client, "cd " + DIAG + " && rm -rf out status.json runner.log && echo clean-diag", "clean-diag")
            step(client, "cd " + DIAG + " && setsid nohup " + PY + " -B unified_runner.py "
                         "> runner.log 2>&1 < /dev/null & echo launched-diag", "launch-diag")
        time.sleep(12)
        step(client, "ps -eo pid,etime,args | grep unified_runner | grep -v grep | wc -l", "runners")
        step(client, "df -h /home /tmp | tail -2", "space")
    finally:
        client.close()


if __name__ == "__main__":
    main()
