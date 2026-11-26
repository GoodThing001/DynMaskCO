import ast
from pathlib import Path

p = Path("scripts/p5v2_formal_analysis.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''        for f in sorted(p.glob("*.json")):
            if f.name == "status.json":
                continue
            rows.append(json.loads(f.read_text(encoding="utf-8")))''',
'''        for f in sorted(p.rglob("*.json")):
            if any(k in f.name for k in ("phase", "status", "audit")):
                continue
            row = json.loads(f.read_text(encoding="utf-8"))
            if "objective" not in row:
                continue
            rows.append(row)''')
t = t.replace('''    norm_rows = load([norm_dir]) if norm_dir else []''',
'''    norm_rows = load([norm_dir]) if norm_dir and Path(norm_dir).exists() else []''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("load filter fixed")
