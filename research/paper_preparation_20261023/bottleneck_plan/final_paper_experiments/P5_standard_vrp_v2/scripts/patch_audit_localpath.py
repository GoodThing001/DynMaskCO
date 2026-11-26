import ast
from pathlib import Path

p = Path("scripts/p5v2_audit.py")
t = p.read_text(encoding="utf-8")
old = '''        path = blocks[idx].get("path")
        if not path:  # 外部基线 phase 无实例路径：按 objective 回退到本地官方数据
            root = Path(__file__).resolve().parent.parent / "data"
            sub = "cvrplib/instances" if row.get("objective") == "cvrp" else "solomon/instances"
            path = str(root / sub / (row["case"] + ".vrp"))'''
new = '''        path = blocks[idx].get("path")
        local = Path(__file__).resolve().parent.parent
        if not path or not Path(str(path)).exists():
            # 远端路径不可达：按 case 名在本地数据/开发实例中解析
            cands = [local / "development/instances" / (row["case"] + ".vrp"),
                     local / "data/cvrplib/instances" / (row["case"] + ".vrp"),
                     local / "data/solomon/instances" / (row["case"] + ".vrp")]
            path = None
            for cand in cands:
                if cand.exists():
                    path = str(cand)
                    break
            if path is None:
                raise FileNotFoundError("instance not found for " + str(row["case"]))'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("local path fallback added")
