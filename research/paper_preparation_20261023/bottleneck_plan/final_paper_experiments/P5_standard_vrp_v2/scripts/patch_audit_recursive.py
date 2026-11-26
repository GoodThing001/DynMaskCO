import ast
from pathlib import Path

p = Path("scripts/p5v2_audit.py")
t = p.read_text(encoding="utf-8")
t = t.replace('    for name in sorted(out.glob("*.json")):',
              '    for name in sorted(out.rglob("*.json")):')
t = t.replace('        if name.name in ("status.json",):\n            continue',
              '        if name.name in ("status.json",) or name.name.startswith("phase"):\n            continue')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("auditor recursive; syntax ok")
