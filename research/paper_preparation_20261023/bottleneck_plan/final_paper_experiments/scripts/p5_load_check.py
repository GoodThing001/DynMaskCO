# -*- coding: utf-8 -*-
"""检查远端负载与 worker 进程。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
uptime
echo ===workers===
ps aux | grep unified_worker | grep -v grep | wc -l
ps aux | grep unified_runner | grep -v grep | wc -l
echo ===top===
ps aux --sort=-%cpu | head -8 | awk '{print $3, $4, $11, $12, $13, $14}'
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
