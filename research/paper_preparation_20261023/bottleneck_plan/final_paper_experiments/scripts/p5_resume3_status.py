# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
python3 - <<'EOF'
import json, os, glob
base = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/"
for ws in ("stage03c", "pyvrp3c", "author3"):
    p = base + ws + "/status.json"
    if not os.path.exists(p):
        print(ws, "no status")
        continue
    s = json.load(open(p))
    print("%-10s end=%s cells=%s/%s failed=%d cores=%s"
          % (ws, s["end"], s["cells"], s["total_cells"], len(s["failed"]), s["cores"]))
EOF
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
