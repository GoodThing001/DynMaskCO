# -*- coding: utf-8 -*-
"""抓取八城输入 JSON（含节点坐标/需求/时窗/服务），供 P2 特征统计。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PROJECT = "/home/hzeng/project/MASKCO-Main"
UNI = PROJECT + "/_maskco_unified_external_20261120/retail01/inputs"

CITIES = ["rw_zg170_p100", "rw_br100_p100", "rw_cp150_p100", "rw_ru120_p100",
          "rw_sg200_p100", "rw_me340_p100", "rw_ca240_p100", "rw_sf300_p100"]

LOCAL = HERE.parent / "P2_retail_analysis" / "inputs"
LOCAL.mkdir(parents=True, exist_ok=True)


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            for c in CITIES:
                s.get(UNI + "/" + c + ".json", str(LOCAL / (c + ".json")))
                print("OK", c)
    finally:
        client.close()


if __name__ == "__main__":
    main()
