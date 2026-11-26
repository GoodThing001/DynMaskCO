import ast
from pathlib import Path

p = Path("scripts/p5v2_fixed_candidates.py")
t = p.read_text(encoding="utf-8")

old = '''    okk, valk, kk = adapter.evaluate(kept)
    if not okk:
        raise SystemExit("state after mask removal infeasible")'''
new = '''    # 掩码状态是"部分解"（缺 40 客户），不能用完整解评价（覆盖性必然不足）；
    # 只需保证每条剩余路线自身可行（容量/时窗），记录其路线代价之和。
    if any((not r) or (not adapter.route_ok(r)) for r in kept):
        raise SystemExit("state after mask removal infeasible")
    valk = sum(adapter.route_proxy(r) for r in kept)
    kk = len(kept)'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("fixed-candidate partial-state check fixed")
