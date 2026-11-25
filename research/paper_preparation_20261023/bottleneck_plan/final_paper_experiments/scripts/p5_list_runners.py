# -*- coding: utf-8 -*-
"""列出全部 runner 命令行，仅保留 author3。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
ps aux | grep unified_runner | grep -v grep | sed -E 's#.*(_maskco_final_paper_p5/[a-z0-9_]+/unified_runner.py).*#\1#' | sort | uniq -c
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
