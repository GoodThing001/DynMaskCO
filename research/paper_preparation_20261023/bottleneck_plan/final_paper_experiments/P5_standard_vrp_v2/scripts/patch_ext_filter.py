from pathlib import Path
p = Path("scripts/p5v2_ext_analysis.py")
t = p.read_text(encoding="utf-8")
t = t.replace('        if "phase" in f.name or "status" in f.name:\n            continue',
              '        if "phase" in f.name or "status" in f.name or "audit" in f.name:\n            continue')
p.write_text(t, encoding="utf-8")
print("filter widened")
