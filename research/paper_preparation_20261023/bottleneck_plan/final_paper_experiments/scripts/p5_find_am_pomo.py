# -*- coding: utf-8 -*-
"""查找 POMO/AttentionModel 作者原始权重文件（远端 stage3 目录）。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import json
from pathlib import Path
base = Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022")
found = {}
for p in base.rglob("*.pt"):
    rel = str(p.relative_to(base))
    if any(k in rel for k in ("AttentionModel", "POMO", "AM", "pomo")):
        found[rel] = p.stat().st_size
for p in base.rglob("*.pkl"):
    rel = str(p.relative_to(base))
    if any(k in rel for k in ("AttentionModel", "POMO", "AM", "pomo")):
        found[rel] = p.stat().st_size
print(json.dumps(found, indent=1)[:1500])
# 也查 run10 recovery
b2 = Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run10_six_methods_dev72/recovery_preparation")
for p in b2.rglob("*"):
    if p.is_file() and p.suffix in (".pt", ".pth") and p.stat().st_size > 100000:
        print("RUN10:", str(p.relative_to(b2)), p.stat().st_size)
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE))
    finally:
        client.close()
    print(out)


if __name__ == "__main__":
    main()
