# -*- coding: utf-8 -*-
"""external01 部署完整 PyVRP 批次 phase（468 格，暂不启动——P5-A 阶段顺序执行后启动）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PHASE = HERE.parent / "P5_standard_vrp/phase_external_pyvrp.json"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            s.put(str(PHASE), P5 + "/external01/phase.json")
        print("external01 phase.json -> phase_external_pyvrp.json (468 blocks)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
