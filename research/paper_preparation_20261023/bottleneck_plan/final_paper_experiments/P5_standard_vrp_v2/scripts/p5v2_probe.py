# -*- coding: utf-8 -*-
"""远程探测：P5 数据目录、BKS 参考与外部基线仓库状态。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = "; ".join([
        "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5 2>/dev/null || { echo NOP5; exit 0; }",
        "echo '[cvrplib]'; ls cvrplib 2>/dev/null | head -8",
        "echo \"[cvrplib count] $(ls cvrplib 2>/dev/null | wc -l)\"",
        "echo '[sol]'; ls cvrplib/*.sol 2>/dev/null | wc -l",
        "echo '[solomon]'; ls solomon 2>/dev/null | head -5",
        "echo \"[solomon count] $(ls solomon 2>/dev/null | wc -l)\"",
        "echo '[bks files]'; find . -maxdepth 3 -iname '*bks*' -o -maxdepth 3 -iname '*.sol' 2>/dev/null | head -10",
        "echo '[v2]'; ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out 2>/dev/null | wc -l",
        "echo '[disk]'; df -h /home | tail -1",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
