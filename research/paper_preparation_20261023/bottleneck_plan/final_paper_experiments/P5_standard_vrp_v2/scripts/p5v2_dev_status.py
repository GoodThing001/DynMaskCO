# -*- coding: utf-8 -*-
"""远程 P5-V2 开发面板进度查询：python p5v2_dev_status.py [dir]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
run = sys.argv[1] if len(sys.argv) > 1 else "dev01"

transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = (
        "cd " + P5V2 + "/" + run + " 2>/dev/null || { echo NODIR; exit 0; }; "
        "echo \"[cells] $(ls out/*.json 2>/dev/null | wc -l)\"; "
        "echo \"[procs] $(ps aux | grep -c 'unified_worker\\.py$')\"; "
        "echo \"[runner] $(ps aux | grep -c 'unified_runner\\.py$')\"; "
        "echo \"[status] $(cat status.json 2>/dev/null | head -c 2000)\"; "
        "echo '[fail]'; grep -l 'error' out/*.json 2>/dev/null | head -5; "
        "echo '[log]'; tail -n 6 runner.log 2>/dev/null; "
        "echo '[disk]'; df -h /home | tail -1"
    )
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
