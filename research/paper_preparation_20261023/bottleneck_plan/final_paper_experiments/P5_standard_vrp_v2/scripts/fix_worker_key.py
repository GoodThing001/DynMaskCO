# -*- coding: utf-8 -*-
"""修正 worker 的 block 键名（versions -> methods）。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\p5v2_worker.py")
t = p.read_text(encoding="utf-8")
old = 'if version not in VERSIONS or version not in block["versions"]:'
new = 'if version not in VERSIONS or version not in block["methods"]:'
assert old in t, "pattern"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("worker key fixed")
