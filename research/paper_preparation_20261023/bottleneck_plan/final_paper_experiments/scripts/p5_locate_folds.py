# -*- coding: utf-8 -*-
"""远端定位冻结链 fold/实例注册表（供 P5-B 泄漏审计精确标注）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
echo ===cross_dataset dir===
ls /home/hzeng/project/MASKCO-Main/_maskco_cross_dataset_20261119/ 2>/dev/null | head -30
echo ===find registries===
find /home/hzeng/project/MASKCO-Main/_maskco_cross_dataset_20261119 -maxdepth 2 -name '*.json' 2>/dev/null | head -20
echo ===grep r1 list in py files===
grep -rln 'r101' /home/hzeng/project/MASKCO-Main/_maskco_cross_dataset_20261119 --include='*.py' 2>/dev/null | head -10
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
