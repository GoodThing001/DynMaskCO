import ast
from pathlib import Path

p = Path("scripts/p5v2_audit.py")
t = p.read_text(encoding="utf-8")

old = '''def dist(inst, a, b):
    xa, ya = inst["xy"][a]
    xb, yb = inst["xy"][b]
    return math.hypot(xa - xb, ya - yb)'''
new = '''def dist(inst, a, b):
    """距离口径：CVRP 逐边整数舍入（EUC_2D，与官方 BKS Cost 一致）；VRPTW 双精度。"""
    xa, ya = inst["xy"][a]
    xb, yb = inst["xy"][b]
    d = math.hypot(xa - xb, ya - yb)
    if inst.get("kind") == "cvrp":
        return float(int(d + 0.5))
    return d'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("audit distance convention fixed (cvrp rounded)")
