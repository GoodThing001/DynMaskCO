import ast
import json
from pathlib import Path

p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
old = """    started = time.perf_counter()
    deadline = started + seconds - 0.8
    hard = started + seconds - (0.6 + 0.0008 * n)"""
new = """    started = time.perf_counter()
    # 内部截止：保守留出"单步不可分割操作"的余量（numpy O(n^2) 步在负载竞争下可达数秒）
    margin = 1.0 + 0.0015 * n
    deadline = started + seconds - max(0.8, margin)
    hard = started + seconds - (margin + 0.3)"""
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)

# 并发预算：正式三段 + 开发面板的 lane 上限（总核数 < 48 物理核）
caps = {"configs/phase_formal_cvrp.json": 16, "configs/phase_formal_vrptw.json": 12,
        "configs/phase_formal_norm.json": 8, "configs/phase_dev.json": 8,
        "configs/phase_diag.json": 6}
for rel, cap in caps.items():
    f = Path(rel)
    if not f.exists():
        continue
    cfg = json.loads(f.read_text(encoding="utf-8"))
    cfg["max_cores"] = cap
    f.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("max_cores", rel, cap)
print("margins patched")
