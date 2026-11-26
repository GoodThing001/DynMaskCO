import ast
from pathlib import Path

p = Path("scripts/p5v2_audit.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''    for row in rows:
        idx = None
        for i, b in enumerate(blocks):
            if (b["case"] == row["case"] and b["seed"] == row["seed"]
                    and row["version"] in b["methods"]):
                idx = i
                break''',
'''    for row in rows:
        ver = row.get("version") or row.get("method")
        row["_ver"] = ver
        idx = None
        for i, b in enumerate(blocks):
            if (b["case"] == row["case"] and b["seed"] == row["seed"]
                    and (ver in b["methods"] or not b.get("methods"))):
                idx = i
                break''')
t = t.replace('''        path = blocks[idx]["path"]
        if path not in cache:''',
'''        path = blocks[idx].get("path")
        if not path:  # 外部基线 phase 无实例路径：按 objective 回退到本地官方数据
            root = Path(__file__).resolve().parent.parent / "data"
            sub = "cvrplib/instances" if row.get("objective") == "cvrp" else "solomon/instances"
            path = str(root / sub / (row["case"] + ".vrp"))
        if path not in cache:''')
t = t.replace('results.append(dict(case=row["case"], version=row["version"],',
              'results.append(dict(case=row["case"], version=ver,')
t = t.replace('rec = dict(case=row["case"], seed=row["seed"], version=row["version"],',
              'rec = dict(case=row["case"], seed=row["seed"], version=ver,')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("auditor generalized; syntax ok")
