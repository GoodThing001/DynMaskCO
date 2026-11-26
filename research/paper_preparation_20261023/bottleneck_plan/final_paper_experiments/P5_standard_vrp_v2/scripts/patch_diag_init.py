import ast
from pathlib import Path

p = Path("scripts/p5v2_fixed_candidates.py")
t = p.read_text(encoding="utf-8")
old = """    payload = dict(case=case, seed=seed, objective=objective, dimension=inst.dimension,
                   mask=mask, insertion_order=[int(x) for x in order],
                   initial=dict(distance=val0, vehicles=k0),"""
new = """    payload = dict(case=case, seed=seed, objective=objective, dimension=inst.dimension,
                   mask=mask, insertion_order=[int(x) for x in order],
                   initial=dict(distance=val0, vehicles=k0),
                   initial_routes=[[int(x) for x in r] for r in init],
                   initial_U=dict(distance=val0, vehicles=k0),"""
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("diag payload includes initial_routes/initial_U")
