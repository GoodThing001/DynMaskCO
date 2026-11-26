# -*- coding: utf-8 -*-
"""修正 VRPTW 生成器：以构造解本身校验窗口可行性（而非重新推导的分组）。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_development_instances.py")
t = p.read_text(encoding="utf-8")

helper = '''

def schedule_feasible(xs, windows, service, routes):
    """校验给定路线在窗口下可行（构造解自检）。"""
    n = len(xs) - 1
    for r in routes:
        t = windows[0][0]
        prev = 0
        for i in r:
            d = math.hypot(xs[prev][0] - xs[i][0], xs[prev][1] - xs[i][1])
            t += d
            if t < windows[i][0]:
                t = windows[i][0]
            if t > windows[i][1] + 1e-9:
                return False
            t += service[i]
            prev = i
        d = math.hypot(xs[prev][0] - xs[0][0], xs[prev][1] - xs[0][1])
        t += d
        if t > windows[0][1] + 1e-9:
            return False
    return True
'''

t = t.replace("\ndef _schedule_ok(", helper + "\n\ndef _schedule_ok(")
t = t.replace("        if _schedule_ok(xs, demands, windows, service, capacity) is not None:\n            return",
              "        if schedule_feasible(xs, windows, service, base):\n            return")
p.write_text(t, encoding="utf-8")
print("patched vrptw check")
