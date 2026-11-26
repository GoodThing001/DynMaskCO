from pathlib import Path
p = Path("standard_vrp/solver/masked_repair.py")
t = p.read_text(encoding="utf-8")
t = t.replace("""    out = reconstruct(adapter, [list(r) for r in parent], mask, heat=heat, regret=regret,
                      log=log)""",
"""    out = reconstruct(adapter, [list(r) for r in parent], mask, heat=heat, regret=regret,
                      log=log, deadline=deadline)""")
p.write_text(t, encoding="utf-8")
print("wrapper deadline wired")
