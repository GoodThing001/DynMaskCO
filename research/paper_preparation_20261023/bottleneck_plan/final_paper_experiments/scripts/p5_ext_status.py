# -*- coding: utf-8 -*-
"""查看 external01 与 tmux 状态（PyVRP 是否完成）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5
tmux ls 2>&1 || true
echo ===STATUS===
python3 -c "import json; s=json.load(open('external01/status.json')); print({k: s[k] for k in ('end','cells','total_cells','failed','complete')})" 2>&1 || true
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
