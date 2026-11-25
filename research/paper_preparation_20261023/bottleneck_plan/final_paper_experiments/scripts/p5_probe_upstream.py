# -*- coding: utf-8 -*-
"""远端查看 retail01 适配器与本地 upstream_extra 内容。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
echo ===retail01===
ls /home/hzeng/project/MASKCO-Main/_maskco_unified_external_20261120/retail01/ 2>/dev/null
echo ===upstream local?===
ls /home/hzeng/project/MASKCO-Main/CC_Compare/checkpoints/ 2>/dev/null
echo ===models===
ls /home/hzeng/project/MASKCO-Main/CC_Compare/models/ 2>/dev/null | head -20
"""


def main():
    print("LOCAL upstream_extra_20261021:")
    d = ROOT / "CC_Compare/checkpoints/upstream_extra_20261021"
    if d.exists():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                print(" ", p.relative_to(ROOT), p.stat().st_size)
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
