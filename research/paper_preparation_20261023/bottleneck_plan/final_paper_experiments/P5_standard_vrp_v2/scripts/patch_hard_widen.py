from pathlib import Path
p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
t = t.replace("    hard = started + seconds - 0.05", "    hard = started + seconds - 0.6")
p.write_text(t, encoding="utf-8")
print("hard deadline widened to -0.6s")
