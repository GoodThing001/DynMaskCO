# -*- coding: utf-8 -*-
"""抓取远端冻结求解链关键源码，供本地设计 P1/P2/P3 工作器。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
PROJECT = "/home/hzeng/project/MASKCO-Main"

FILES = [
    "_maskco_perishable_external_20261119/carryprobe01/external_common.py",
    "_maskco_perishable_external_20261119/carryprobe01/pool_search.py",
    "_maskco_perishable_external_20261119/carryprobe01/block_split.py",
    "_maskco_perishable_external_20261119/orderdev01/block_search.py",
    "_maskco_cross_dataset_20261119/compare.py",
    "_maskco_unified_external_20261120/origin01/phase.json",
    "_maskco_unified_external_20261120/origin01/unified_worker.py",
    "_maskco_unified_external_20261120/retail01/phase.json",
]

LOCAL = HERE.parent / "audit" / "remote_sources"
LOCAL.mkdir(parents=True, exist_ok=True)


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            for f in FILES:
                name = f.replace("/", "__")
                try:
                    s.get(PROJECT + "/" + f, str(LOCAL / name))
                    print("OK", f)
                except Exception as e:
                    print("MISS", f, type(e).__name__)
    finally:
        client.close()


if __name__ == "__main__":
    main()
