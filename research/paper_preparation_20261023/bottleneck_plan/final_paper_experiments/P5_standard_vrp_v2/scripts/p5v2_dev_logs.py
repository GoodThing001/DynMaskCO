# -*- coding: utf-8 -*-
"""查看 V2 dev01 的首个 worker 日志与进程 CPU。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01
ls out/*.log 2>/dev/null | head -6
for f in $(ls out/*.log 2>/dev/null | head -2); do echo "=== $f"; tail -8 $f; done
echo ===ps===
ps aux | grep unified_worker | grep -v grep | awk '{print $3, $4, $12, $13, $14}' | head -12
uptime
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
