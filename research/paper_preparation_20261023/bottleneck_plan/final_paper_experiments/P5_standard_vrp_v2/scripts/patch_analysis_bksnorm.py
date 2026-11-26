import ast
import re
from pathlib import Path

p = Path("scripts/p5v2_formal_analysis.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''def bks_tables():
    cvrp = json.loads((V2 / "data/bks_derived.json").read_text(encoding="utf-8"))["cvrp"]
    vrptw = json.loads((V2 / "data/solomon_bks_derived.json").read_text(encoding="utf-8"))["rows"]
    return ({r.get("name") or r["instance"]: r for r in cvrp},
            {r.get("name") or r["instance"]: r for r in vrptw})''',
'''def bks_tables():
    import re as _re
    cvrp = json.loads((V2 / "data/bks_derived.json").read_text(encoding="utf-8"))["cvrp"]
    vrptw = json.loads((V2 / "data/solomon_bks_derived.json").read_text(encoding="utf-8"))["rows"]
    cv = {}
    for r in cvrp:
        name = r.get("name") or r["instance"]
        m = _re.search(r"-n(\\d+)-k(\\d+)", name)
        n = int(m.group(1)) - 1 if m else None     # 实例维度 - 1 个仓库
        cv[name] = dict(bks=r.get("rounded") or r.get("official_cost"),
                        k_sol=r.get("k_sol") or r.get("k"),
                        n=n)
    vt = {r.get("name") or r["instance"]: r for r in vrptw}
    return cv, vt''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("bks tables normalized")
