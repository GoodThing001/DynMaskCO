# -*- coding: utf-8 -*-
"""远端查看 RECENT5 作者模型目录结构与推理入口（为外部批次实现做范围评估）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
A=/home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets
echo ===MVMoE===
ls $A/MVMoE/ | head; ls $A/MVMoE/pretrained/mvmoe_4e_n100/ | head
echo ===RRNCO===
ls $A/RRNCO/ | head; ls $A/RRNCO/checkpoints/ 2>/dev/null | head
echo ===neural_assets===
ls $A/neural_assets/; ls $A/neural_assets/CaDA/ $A/neural_assets/RouteFinder/ 2>/dev/null
echo ===models dir===
ls /home/hzeng/project/MASKCO-Main/CC_Compare/models/CaR/ /home/hzeng/project/MASKCO-Main/CC_Compare/models/DeepACO/ /home/hzeng/project/MASKCO-Main/CC_Compare/models/RouteFinder/ 2>/dev/null | head -30
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
