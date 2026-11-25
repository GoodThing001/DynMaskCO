# -*- coding: utf-8 -*-
"""远端定位作者模型权重与推理入口。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
echo ===AM/POMO/DeepACO/CaR===
find /home/hzeng/project -maxdepth 6 \( -name 'epoch-99.pt' -o -name 'checkpoint-30500.pt' -o -name 'cvrp100.pt' -o -path '*CaR-POMO_100*' \) 2>/dev/null | head -20
echo ===RECENT5===
find /home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets -maxdepth 4 \( -name '*.pt' -o -name '*.ckpt' \) 2>/dev/null | head -40
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
