# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
python3 - <<'EOF'
import json
s = json.load(open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/author4/status.json"))
print("cells", s["cells"], "failed", len(s["failed"]))
for f in s["failed"]:
    print(" ", f["block"], f["case"], f["arm"])
EOF
ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/author4/out/*.log 2>/dev/null | head -3
tail -12 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/author4/out/3_CaDA.log 2>/dev/null
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
