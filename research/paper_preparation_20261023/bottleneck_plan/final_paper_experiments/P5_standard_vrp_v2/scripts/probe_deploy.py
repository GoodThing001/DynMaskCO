# -*- coding: utf-8 -*-
"""远端部署探针：逐步骤定位 SFTP 失败点。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, "df -h /home | tail -1; ls -d " + P5V2 + " 2>&1; mkdir -p " + P5V2 + " && echo mkdir_ok; ls " + P5V2 + " 2>&1 | head -5"))
        with client.open_sftp() as s:
            for step, path in (("root", P5V2), ("svr", P5V2 + "/standard_vrp"),
                               ("core", P5V2 + "/standard_vrp/core"), ("dev", P5V2 + "/dev01")):
                try:
                    s.mkdir(path)
                except OSError as exc:
                    print("mkdir", step, "->", exc)
            probe = V2 / "standard_vrp/core/calibration.py"
            try:
                s.put(str(probe), P5V2 + "/standard_vrp/core/calibration.py")
                print("put probe OK")
            except Exception as exc:
                print("put probe FAIL:", type(exc).__name__, exc)
            print("list:", s.listdir(P5V2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
