# -*- coding: utf-8 -*-
"""修复生成器中被注入的字面量 `n。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_development_instances.py")
t = p.read_text(encoding="utf-8")
bad = "windows = [(0.0, 1e9)] * (n + 1)`n        longest = 0.0"
good = "windows = [(0.0, 1e9)] * (n + 1)\n        longest = 0.0"
assert bad in t, "pattern not found"
t = t.replace(bad, good)
p.write_text(t, encoding="utf-8")
print("fixed")
