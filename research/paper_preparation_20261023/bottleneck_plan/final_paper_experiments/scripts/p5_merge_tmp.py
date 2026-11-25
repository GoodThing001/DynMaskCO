# -*- coding: utf-8 -*-
"""合并 tmp 回收目录到主结果目录（跳过截断 JSON）；随后重算缺口。"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"

PAIRS = [("external_tmp_s", "results_stage03"),
         ("external_tmp_1", "results_stage03"),
         ("external_tmp_p", "results_external_pyvrp3"),
         ("external_tmp_2", "results_external_pyvrp3"),
         ("external_tmp_a", "results_external_author"),
         ("external_tmp_b", "results_external_author"),
         ("external_tmp_3", "results_external_author")]

for src, dst in PAIRS:
    s = P5 / src / "out"
    if not s.exists():
        continue
    d = P5 / dst / "out"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(s.glob("*.json")):
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        shutil.copy2(f, d / f.name)
        n += 1
    print("%-18s -> %-26s %d valid" % (src, dst, n))
    shutil.rmtree(P5 / src, ignore_errors=True)
