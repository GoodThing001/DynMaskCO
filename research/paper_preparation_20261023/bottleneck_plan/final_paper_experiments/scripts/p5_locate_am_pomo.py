# -*- coding: utf-8 -*-
"""远端定位 AM/POMO 仓库与权重。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
echo ===upstream_extra===
ls -d /home/hzeng/project/*upstream* /home/hzeng/project/MASKCO-Main/*upstream* 2>/dev/null
find /home/hzeng/project -maxdepth 5 -type d \( -name 'AttentionModel' -o -name 'POMO' \) 2>/dev/null | head -10
echo ===POMO checkpoint===
find /home/hzeng/project -maxdepth 8 -name 'checkpoint-30500.pt' 2>/dev/null | head -5
echo ===AM epoch===
find /home/hzeng/project -maxdepth 8 -name 'epoch-99.pt' 2>/dev/null | head -5
echo ===CaR content===
ls /home/hzeng/project/MASKCO-Main/CC_Compare/checkpoints/CaR/CVRP/CaR-POMO_100/ 2>/dev/null
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
