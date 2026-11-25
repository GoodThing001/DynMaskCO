# -*- coding: utf-8 -*-
"""P5 数据回收：远端 CVRPLIB/Solomon 数据拉回本地并核验 SHA。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

REMOTE = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
LOCAL = HERE.parent / "P5_standard_vrp" / "data"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            for kind in ("cvrplib", "solomon"):
                for sub in ("instances", "bks"):
                    rd = REMOTE + "/" + kind + "/" + sub
                    ld = LOCAL / kind / sub
                    ld.mkdir(parents=True, exist_ok=True)
                    try:
                        names = s.listdir(rd)
                    except FileNotFoundError:
                        print("SKIP", rd)
                        continue
                    for n in sorted(names):
                        if not n.endswith((".vrp", ".sol")):
                            continue
                        s.get(rd + "/" + n, str(ld / n))
                    print("pulled", kind, sub, len([x for x in names if x.endswith((".vrp", ".sol"))]))
                for mf in ("source_manifest.json",):
                    try:
                        s.get(REMOTE + "/" + kind + "/" + mf, str(LOCAL / kind / mf))
                        print("pulled", kind, mf)
                    except FileNotFoundError:
                        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
