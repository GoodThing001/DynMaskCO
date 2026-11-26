# -*- coding: utf-8 -*-
"""修正开发实例生成器的名称与配对（避免重复实例）。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_development_instances.py")
t = p.read_text(encoding="utf-8")
t = t.replace('return "XDEV-n%d-s%d" % (n, seed), coords, demands, cap',
              'return "XDEV-n%d-s%d-r%d" % (n, seed, tag % 2), coords, demands, cap')
t = t.replace('return "RDEV-%s-s%d" % (layout[0].upper(), seed), xs, demands, windows, service, capacity, 10, base',
              'return "RDEV-%s-s%d-r%d" % (layout[0].upper(), seed, tag % 2), xs, demands, windows, service, capacity, 10, base')
t = t.replace('specs.append(("cvrp", n, seed + m, len(specs)))',
              'specs.append(("cvrp", n, seed, m))')
t = t.replace('specs.append(("vrptw", layout, seed + m, len(specs)))',
              'specs.append(("vrptw", layout, seed, m))')
p.write_text(t, encoding="utf-8")
print("patched")
