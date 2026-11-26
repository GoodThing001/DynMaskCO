import ast
from pathlib import Path

p = Path("scripts/p5v2_audit.py")
t = p.read_text(encoding="utf-8")
t = t.replace('        if name.name in ("status.json",) or name.name.startswith("phase"):\n            continue',
              '        if "phase" in name.name or "status" in name.name:\n            continue')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("skip filter fixed")
