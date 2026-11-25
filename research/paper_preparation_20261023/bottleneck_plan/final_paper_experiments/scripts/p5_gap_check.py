# -*- coding: utf-8 -*-
"""诊断续跑缺口计算：对比 phase 块与已回收 JSON 的键集合（容忍磁盘满导致的截断文件）。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"

for ws, phase_file, res in (("stage03", "phase_stage03.json", "results_stage03"),
                            ("pyvrp3", "phase_external_pyvrp.json", "results_external_pyvrp3"),
                            ("author", "phase_external_author.json", "results_external_author")):
    phase = json.loads((P5 / phase_file).read_text(encoding="utf-8"))
    have = set()
    bad = []
    files = list((P5 / res / "out").glob("*.json"))
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bad.append(f.name)
            continue
        have.add((r["case"], r["seed"], r["method"]))
    want = set()
    for b in phase["blocks"]:
        for m in b["methods"]:
            want.add((b["case"], b["seed"], m))
    print("%-8s files=%d corrupt=%d want=%d have=%d missing=%d extra=%d"
          % (ws, len(files), len(bad), len(want), len(have), len(want - have), len(have - want)))
    if bad:
        print("   corrupt:", bad[:5])
