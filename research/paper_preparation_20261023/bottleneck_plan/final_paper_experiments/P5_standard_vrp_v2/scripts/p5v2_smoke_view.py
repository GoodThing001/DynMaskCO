# -*- coding: utf-8 -*-
"""查看 V2 冒烟结果。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01
ls out/*.json 2>/dev/null | wc -l
ls smoke_done 2>/dev/null || echo not_done
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("out/0_*.json")):
    r = json.load(open(f))
    st = r.get("stats", {})
    print(r["version"], r["case"], "D=%.1f" % r["distance"], "K=%d" % r["vehicles"],
          "online=%.1f" % r["online_s"], "epochs=%s" % st.get("epochs"),
          "fwd=%s" % (st.get("predictor", {}).get("success") if st.get("predictor") else None))
EOF
tail -6 smoke_logs/0_STD_FIX.log 2>/dev/null
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
