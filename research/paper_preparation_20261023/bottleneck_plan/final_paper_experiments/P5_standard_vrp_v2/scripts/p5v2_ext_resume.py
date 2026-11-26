# -*- coding: utf-8 -*-
"""外部 PyVRP 面板断点续跑：只重跑缺失块（前一进程被误杀所致）。"""
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
SRC = "/tmp/p5v2_formal/ext_pyvrp"
RUN = "/tmp/p5v2_formal/ext_pyvrp2"


def main():
    phase = json.loads((V2 / "configs/phase_external_pyvrp.json").read_text(encoding="utf-8"))
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        listing = transport.remote_exec(client, "ls " + SRC + "/out 2>/dev/null || true")
        done = {int(n.split("_")[0]) for n in str(listing).split()
                if n.endswith(".json") and n.split("_")[0].isdigit()}
        missing = [i for i in range(len(phase["blocks"])) if i not in done]
        print("done", len(done), "missing", len(missing))
        if not missing:
            print("nothing to resume")
            return 0
        blocks = []
        for k, i in enumerate(missing):
            b = dict(phase["blocks"][i])
            blocks.append(b)
        sub = dict(phase, name="p5v2_external_pyvrp_resume", blocks=blocks, max_cores=10)
        out = V2 / "configs/phase_external_pyvrp_resume.json"
        out.write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
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
            s.put(str(out), RUN + "/phase.json")
            s.put(str(P5V1 / "p5_external_worker.py"), RUN + "/unified_worker.py")
            s.put(str(P5V1 / "p5_external_worker.py"), RUN + "/future_dispatch.py")
        cmd = ("cd " + RUN + " && cp " + P5 + "/stage01/unified_runner.py . 2>/dev/null; "
               "rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 3; pgrep -fc 'ext_pyvrp2' || true")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
