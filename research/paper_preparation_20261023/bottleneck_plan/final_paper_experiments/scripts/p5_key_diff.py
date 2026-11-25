# -*- coding: utf-8 -*-
"""查看 stage03 缺失键与 r_ 前缀文件的键。"""
import json
from pathlib import Path

P5 = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp")
phase = json.loads((P5 / "phase_stage03.json").read_text(encoding="utf-8"))
out = P5 / "results_stage03" / "out"
have = set()
for f in out.glob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    have.add((r["case"], r["seed"], r["method"]))
want = {(b["case"], b["seed"], m) for b in phase["blocks"] for m in b["methods"]}
print("missing keys:", sorted(want - have)[:20])
print("r_ files keys:", [ (json.loads(f.read_text(encoding='utf-8'))["case"], json.loads(f.read_text(encoding='utf-8'))["seed"], json.loads(f.read_text(encoding='utf-8'))["method"]) for f in out.glob("r_*.json")][:20])
