# -*- coding: utf-8 -*-
"""杀死指定工作区的 runner/worker（保留 author3）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
pkill -f '_maskco_final_paper_p5/stage03b/' 2>/dev/null; echo killed_stage03b
pkill -f '_maskco_final_paper_p5/pyvrp3b/' 2>/dev/null; echo killed_pyvrp3b
pkill -f '_maskco_final_paper_p5/p5_resume_' 2>/dev/null || true
sleep 2
echo ===remaining===
ps aux | grep -E 'unified_runner|unified_worker' | grep -v grep | awk '{print $11, $12, $13, $14}' | sort | uniq -c | head -8
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
