# -*- coding: utf-8 -*-
"""后台监视 P1 完成并自动回收 receipts。每 5 分钟检查一次，完成后退出。"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
from p1_transport import follow

transport.record = lambda *a, **k: None

TARGET = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p1/p1/exit_code"


def main():
    pkg = HERE.parent / "P1_original_baselines"
    for i in range(120):  # 最长 10 小时
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
                        with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p1/p1/status.json") as f:
                            import json
                            st = json.load(f)
                        print(f"[monitor] {st['complete']}/{st['total']} blocks, cells {st['cells']}, failed {len(st['failed'])}", flush=True)
                    except FileNotFoundError:
                        print("[monitor] staging", flush=True)
        finally:
            client.close()
        if done:
            break
        time.sleep(300)
    follow("p1", pkg, smoke=False)


if __name__ == "__main__":
    main()
