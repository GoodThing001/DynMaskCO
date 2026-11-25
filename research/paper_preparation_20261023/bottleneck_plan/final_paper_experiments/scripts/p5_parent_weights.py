# -*- coding: utf-8 -*-
"""查作者原始权重的来源路径（parent_loaded.json）与远端 CC_Compare 检查点。"""
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
st = Path("/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run04_cpu_resume/outputs")
for m in ("AttentionModel", "POMO", "DeepACO", "CaR"):
    p = st/m/"parent_loaded.json"
    if p.is_file():
        d = json.loads(p.read_text())
        print(m, ":", json.dumps(d)[:300])
# CC_Compare 检查点
cc = Path("/home/hzeng/project/MASKCO-Main/CC_Compare")
for f in cc.rglob("*.pt"):
    print("CC_PT:", str(f.relative_to(cc)), f.stat().st_size)
for f in cc.rglob("*.pth"):
    print("CC_PTH:", str(f.relative_to(cc)), f.stat().st_size)
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
