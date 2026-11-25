# -*- coding: utf-8 -*-
"""监视 P4 冒烟完成并回收。"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
from campaign_transport import follow, SPECS

transport.record = lambda *a, **k: None

TARGET = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p4/smoke01/exit_code"


def main():
    spec = SPECS["p4"]
    for i in range(120):
        client, _ = transport.connect(ROOT / ".vscode/sftp.json")
        try:
            with client.open_sftp() as s:
                try:
                    s.stat(TARGET)
                    done = True
                except FileNotFoundError:
                    done = False
                if not done:
                    try:
                        with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p4/smoke01/status.json") as f:
                            import json
                            st = json.load(f)
                        print(f"[monitor p4-smoke] cells {st['cells']}/{st['total_cells']}, failed {len(st['failed'])}", flush=True)
                    except FileNotFoundError:
                        print("[monitor p4-smoke] staging", flush=True)
        finally:
            client.close()
        if done:
            break
        time.sleep(240)
    follow(spec, "p4", smoke=True)


if __name__ == "__main__":
    main()
