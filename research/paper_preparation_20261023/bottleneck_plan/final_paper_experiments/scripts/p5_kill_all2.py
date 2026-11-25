# -*- coding: utf-8 -*-
"""全量清理 P5 runner/worker 进程（锚定 pkill 模式避免自匹配）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
pkill -f 'unified_worker\.py [0-9]+ [A-Za-z_]+$' 2>/dev/null
pkill -f 'unified_runner\.py$' 2>/dev/null
sleep 2
ps aux | grep -E 'unified_worker|unified_runner' | grep -v grep | wc -l
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
