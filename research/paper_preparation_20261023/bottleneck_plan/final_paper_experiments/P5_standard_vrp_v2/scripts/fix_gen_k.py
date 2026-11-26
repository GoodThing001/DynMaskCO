# -*- coding: utf-8 -*-
"""修正：构造解分组数不设上限（k 由构造解决定），避免可行性构造死循环。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_development_instances.py")
t = p.read_text(encoding="utf-8")
t = t.replace("    if len(groups) > k:\n        return None\n", "")
t = t.replace("            return \"RDEV-%s-s%d-r%d\" % (layout[0].upper(), seed, tag % 2), xs, demands, windows, service, capacity, 10, base",
              "            return \"RDEV-%s-s%d-r%d\" % (layout[0].upper(), seed, tag % 2), xs, demands, windows, service, capacity, len(base), base")
p.write_text(t, encoding="utf-8")
print("patched k/limit")
