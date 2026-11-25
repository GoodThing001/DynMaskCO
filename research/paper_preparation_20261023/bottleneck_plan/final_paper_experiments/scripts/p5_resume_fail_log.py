# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
for ws in stage03b pyvrp3b; do
  echo "=== $ws runner.log ==="
  tail -8 /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/$ws/runner.log 2>/dev/null
  echo "=== $ws failed ==="
  python3 -c "import json;s=json.load(open('/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/$ws/status.json'));print(s['failed'][:2], s['cores'], s['cells'])" 2>&1 | head -3
done
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
