# -*- coding: utf-8 -*-
"""查看 external_precheck 结果。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck
ls out/1_*.json 2>/dev/null | wc -l
ls precheck_done 2>/dev/null || true
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("out/1_*.json")):
    r = json.load(open(f))
    print(r["method"], r["status"], "K=%d" % r["vehicles"], "D=%.2f" % r["distance"],
          "online=%.1f" % r["online_s"])
EOF
tail -3 out/1_RRNCO.log 2>/dev/null || true
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
