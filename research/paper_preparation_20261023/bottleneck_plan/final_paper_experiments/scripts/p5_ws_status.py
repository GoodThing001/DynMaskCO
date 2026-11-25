# -*- coding: utf-8 -*-
"""检查指定工作区状态（命令行参数：ws1 ws2 ...）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
python3 - <<'EOF'
import json, os
base = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/"
for ws in __WS__:
    p = base + ws + "/status.json"
    if not os.path.exists(p):
        print(ws, "no status")
        continue
    s = json.load(open(p))
    print(ws, "end=", s["end"], "cells=", s["cells"], "/", s["total_cells"], "failed=", len(s["failed"]))
EOF
"""


def main(wss):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL.replace("__WS__", repr(tuple(wss)))))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1:])
