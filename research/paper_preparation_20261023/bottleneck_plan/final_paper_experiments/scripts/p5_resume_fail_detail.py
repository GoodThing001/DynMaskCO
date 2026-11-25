# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
python3 - <<'EOF'
import json
base = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/"
for ws in ("stage03c", "author3"):
    s = json.load(open(base + ws + "/status.json"))
    print("===", ws, "failed:", len(s["failed"]))
    for f in s["failed"][:3]:
        print(" ", f["block"], f["case"], f["arm"], ":", f["error"][:180])
EOF
tail -5 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage03c/runner.log 2>/dev/null
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
