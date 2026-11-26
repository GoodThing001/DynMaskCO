import ast
from pathlib import Path

p = Path("scripts/p5v2_worker.py")
t = p.read_text(encoding="utf-8")

# 1) 修 RV：必须传路径而不是已读取的文本
t = t.replace("""    text = Path(path).read_text(encoding="utf-8")
    if objective == "cvrp":
        return CVRPInstance.parse_vrp(text)
    return RV(text)""",
"""    if objective == "cvrp":
        return CVRPInstance.parse_vrp(Path(path).read_text(encoding="utf-8"))
    return RV(path)""")

# 2) 幂等幂等：已存在有效 json 时直接跳过（断点续跑；由 runner 复验）
t = t.replace("""    case, seed, objective = block["case"], block["seed"], block["objective"]
    inst = load_instance(block["path"], objective)""",
"""    case, seed, objective = block["case"], block["seed"], block["objective"]
    dest = HERE / "out" / f"{idx}_{version}.json"
    if dest.exists():
        try:
            done = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:
            dest.unlink()
        else:
            if done.get("case") == case and done.get("seed") == seed:
                return      # 断点续跑：保留既有端点（runner 会再次核验）
    inst = load_instance(block["path"], objective)""")
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("worker fixed (RV path + idempotent skip); syntax ok")
