# -*- coding: utf-8 -*-
"""检查远端零售/基准批次目录中 P2 轨迹工作器所需文件。"""
import json
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
uni = Path("/home/hzeng/project/MASKCO-Main/_maskco_unified_external_20261120")
for batch in ("retail01", "benchmark01", "trajectory01"):
    d = uni/batch
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print(batch, ":", [f for f in files if f in ("run_dev72_clock.py","external_common.py","future_dispatch.py","trajectory_worker.py","locked_method.json","heldout_admission.json","heldout_sources.json","masked_repair.so","split_order.so","unified_worker.py")])
    if (d/"inputs").is_dir():
        print("   inputs:", len(list((d/"inputs").iterdir())))
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
