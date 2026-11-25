# -*- coding: utf-8 -*-
"""远端清单：作者原始权重与 RECENT5 资产（P5 方法支持矩阵核实）。"""
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
r = Path("/home/hzeng/project/MASKCO-Main")
out = {}
# 1) comparison_v2 环境中的作者权重
cv = r/"C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02"
for sub in ("checkpoints", "envs"):
    p = cv/sub
    if p.is_dir():
        found = {}
        for f in p.rglob("*.pt"):
            found[str(f.relative_to(p))] = f.stat().st_size
        for f in p.rglob("*.pth"):
            found[str(f.relative_to(p))] = f.stat().st_size
        out[sub] = found
# 2) stage3 输出（冷链适配 11400 步）
st = r/"C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run04_cpu_resume/outputs"
if st.is_dir():
    out["stage3_outputs"] = {x.name: sorted(f.name for f in x.glob("*") if f.is_file()) for x in st.iterdir() if x.is_dir()}
# 3) RECENT5 资产
ra = r/"_maskco_recent5_20261120/assets"
if ra.is_dir():
    out["recent5_assets"] = sorted(str(p.relative_to(ra)) for p in ra.rglob("*") if p.is_file())[:40]
print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])
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
