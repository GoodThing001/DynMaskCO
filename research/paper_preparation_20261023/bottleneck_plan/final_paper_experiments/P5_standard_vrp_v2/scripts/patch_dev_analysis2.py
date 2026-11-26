from pathlib import Path
p = Path("scripts/p5v2_dev_analysis.py")
t = p.read_text(encoding="utf-8")
old = """               per_version=agg, pairs=pairs, predictor=pred, mechanisms=mech,
               cell_table=cells)"""
new = """               per_version=agg, pairs=pairs, predictor=pred, mechanisms=mech,
               cell_table=[dict(case=k[0], seed=k[1], objective=k[2], n=v.get("n"),
                                versions=v["versions"]) for k, v in sorted(cells.items())])"""
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("patched")
