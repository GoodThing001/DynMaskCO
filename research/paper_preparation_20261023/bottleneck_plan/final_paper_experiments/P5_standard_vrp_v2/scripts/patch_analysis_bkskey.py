import ast
from pathlib import Path

p = Path("scripts/p5v2_formal_analysis.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''    return ({r["name"]: r for r in cvrp}, {r["name"]: r for r in vrptw})''',
'''    return ({r.get("name") or r["instance"]: r for r in cvrp},
            {r.get("name") or r["instance"]: r for r in vrptw})''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("bks table key fixed")
