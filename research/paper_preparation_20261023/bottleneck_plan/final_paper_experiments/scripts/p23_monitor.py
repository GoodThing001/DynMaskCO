# -*- coding: utf-8 -*-
"""后台监视 P2/P3 完成并自动回收。用法: python p23_monitor.py p2|p3"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
from campaign_transport import follow, SPECS

transport.record = lambda *a, **k: None

name = sys.argv[1]
spec = SPECS[name]
TARGET = "/home/hzeng/project/MASKCO-Main/" + spec["remote"] + "/" + name + "/exit_code"
STATUS = "/home/hzeng/project/MASKCO-Main/" + spec["remote"] + "/" + name + "/status.json"


def main():
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
                        with s.open(STATUS) as f:
                            st = json.load(f)
                        print(f"[monitor {name}] {st['complete']}/{st['total']} blocks, cells {st['cells']}, failed {len(st['failed'])}", flush=True)
                    except FileNotFoundError:
                        print(f"[monitor {name}] staging", flush=True)
        finally:
            client.close()
        if done:
            break
        time.sleep(300)
    follow(spec, name, smoke=False)


if __name__ == "__main__":
    main()
