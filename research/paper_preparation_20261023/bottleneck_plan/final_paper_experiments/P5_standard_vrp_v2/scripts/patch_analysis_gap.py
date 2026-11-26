import ast
from pathlib import Path

p = Path("scripts/p5v2_formal_analysis.py")
t = p.read_text(encoding="utf-8")
old = '''        entry = dict(instance=case, seed=seed, bks=b.get("rounded"), bks_k=b.get("k_sol"),
                     n=b.get("n"))'''
new = '''        ref = b.get("bks") or b.get("rounded")
        entry = dict(instance=case, seed=seed, bks=ref, bks_k=b.get("k_sol"), n=b.get("n"))'''
assert old in t
t = t.replace(old, new)
t = t.replace('''                entry[m + "_gap%"] = (100.0 * (d[m]["distance"] - b["rounded"]) / b["rounded"]
                                      if b.get("rounded") else None)''',
'''                entry[m + "_gap%"] = (100.0 * (d[m]["distance"] - ref) / ref
                                      if ref else None)''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("gap computation fixed")
