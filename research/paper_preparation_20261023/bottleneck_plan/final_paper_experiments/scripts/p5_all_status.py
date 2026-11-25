# -*- coding: utf-8 -*-
"""一次性查看全部在跑批次状态（tmux + status.json 摘要）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5
tmux ls 2>&1 || true
python3 - <<'EOF'
import json, os, glob
for ws in ("stage03", "external_author2", "external_pyvrp3"):
    p = ws + "/status.json"
    if not os.path.exists(p):
        print(ws, "no status")
        continue
    s = json.load(open(p))
    n = len(glob.glob(ws + "/out/*.json"))
    print("%-18s end=%s cells=%s/%s done_blocks=%s failed=%d out_json=%d"
          % (ws, s["end"], s["cells"], s["total_cells"], s["complete"], len(s["failed"]), n))
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
